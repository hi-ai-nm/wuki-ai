#!/usr/bin/env bash
# ================= 容器入口：nginx（对外 8689）+ wuki-ai（只回环 9000） =================
#
# 一个容器跑两个进程，不装 supervisord 是为了不额外拖一层 Python 依赖。
# 这里要的「守护」其实只有两件事：
#   1. 收到 docker stop 的 SIGTERM 时，把两个进程都按顺序停掉；
#   2. 谁先退出都把另一个带走，容器整体退出 —— 避免出现「nginx 活着、后端早就没了」
#      的假象（那种情况 compose 的 restart 策略也不会帮我们重启）。
#
# wuki-ai 的参数由 CMD（或 compose 的 command）原样传进来：默认 "-S 9000 -C corpus.txt ..."。
# 注意 CMD 里必须带模式（-S / -R / -SR），只给 -C 会走 main 的默认重训分支。
set -uo pipefail

APP=/app/Bin/wuki-ai

# daemon off：让 nginx 留在前台，由本脚本管它的生命周期。
# 主上下文单独把 error_log 指到 stderr —— 配置写错 / 端口被占这类「还没读到
# conf.d 就报出来」的错误也才能在 docker logs 里看到。
nginx -g 'daemon off; error_log stderr warn;' &
NGINX_PID=$!

"$APP" "$@" &
APP_PID=$!

# 两个都停：先 TERM（nginx 快速退出、wuki-ai 直接终止），再等它们真的走完。
shutdown() {
    kill -TERM "$APP_PID" "$NGINX_PID" 2>/dev/null || true
}
trap shutdown TERM INT QUIT

# wait -n：任意一个退出就返回，$? 是它自己的退出码。
wait -n "$APP_PID" "$NGINX_PID"
status=$?

shutdown
wait "$APP_PID" 2>/dev/null || true
wait "$NGINX_PID" 2>/dev/null || true

# 把先挂的那个进程的退出码交给 docker：137/143 之类的信号码、训练失败的 1~5、
# 端口被占的 8，都能在 docker ps / compose logs 里一眼看出停在哪一步。
exit "$status"
