# ================= 阶段1：构建 =================
FROM ubuntu:24.04 AS builder

WORKDIR /app

# 只装编译器和 cmake：项目自己只用标准库 + vendored 的 src/external/json.hpp。
# 之前那串（drogon / trantor / jsoncpp / uuid / zlib / pq / sqlite / mariadb / brotli / hiredis / yaml-cpp）
# 全是给已经删掉的 WukiWeb 动态库用的，现在一个都不需要
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY Wuki-ai .

# 编译（CMake 会自动创建 /app/Bin 并放入所有产物）
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)

# ================= 阶段2：运行 =================
FROM ubuntu:24.04

# 运行阶段只多装一个 nginx：它是容器对外的唯一入口（8689），把请求反代给同容器里
# 只听回环的 wuki-ai（9000）。--no-install-recommends 是为了别把 geoip / perl / xslt
# 那一串扩展模块也拖进来；strip 掉默认站点（它 listen 80），免得以后有人误以为 80 也对外。
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

# 设置工作目录为 Bin，这样 $ORIGIN 的 rpath 才能找到同目录的动态库
WORKDIR /app/Bin

# 把整个 /app/Bin 目录（包含 exe 和三个 .so：WukiInclude / WukiLib / WukiComm）全部拷过来
COPY --from=builder /app/Bin /app/Bin

# 语料在运行阶段才需要（-C corpus.txt），所以只拷进最终镜像。
# 96 MB 的语料必须在构建上下文里 —— .dockerignore 里如果排除它，这句会直接
# "failed to compute cache key: not found"，这就是之前 docker build 挂掉的原因。
COPY Wuki-ai/src/main/llm/corpus.txt /app/Bin/corpus.txt

# 文字画文件
COPY Wuki-ai/Bin/capa.conf /app/Bin/capa.conf

# 反向代理配置：8689（对外）→ 127.0.0.1:9000（wuki-ai）。
# 顺手在 build 阶段跑一次 nginx -t：配置写错现在炸，别等到容器起来才发现。
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN nginx -t

# 入口脚本：先把 nginx 拉起来，再按 CMD 的参数拉起 wuki-ai；谁挂了都让容器整体退出
# （见脚本里 wait -n 那几行，不另装 supervisord）。
COPY docker-entrypoint.sh /usr/local/bin/wuki-entrypoint.sh
RUN chmod +x /usr/local/bin/wuki-entrypoint.sh

# 对外只发布 nginx 的 8689（docker-compose.yml 里映射的就是它）。
# 9000 是 wuki-ai 的端口，只监听容器内回环 —— 不 EXPOSE、不发布，
# 所以宿主上除了 8689 没有别的口能碰到模型服务。
EXPOSE 8689

# 注意：可执行文件的真实名字由 CMakeLists.txt 的 OUTPUT_NAME 决定，现在是 "wuki-ai"，
# 不是 target 名 wuki-ai-B4A。之前写成 "./wuki-ai-b4a" 会直接 runc: no such file or directory。
# -C 只是修饰符（换语料），不指定模式时 main 里 mode 为空 → 走默认重训，
# 所以 CMD 必须带一个模式：-S 起服务（compose 映射 8689 就是给它用的），-R 重训一遍。
#
# 端口分工：EXPOSE / compose 发布的是 nginx 的 8689；"-S 9000" 的 9000 是 main.cpp
# 里「孤立数字 = 服务端口」的参数写法，wuki-ai 只听容器内 127.0.0.1:9000，
# nginx.conf 的 proxy_pass 打的就是它。两边要一起改。
#
# 训练量的账（开销 = 轮数 × 样本数，轮数参数见 main.cpp 的 -Esg / -EPre / -EAll）：
#   - 轮数维持 main.cpp 给真语料的默认值（3 / 2 / 3）就够，不要再加 -E 参数。
#     以前 CMD 是 -Esg 1000 -EPre 1000 -EAll 10000 + 全量 55 万行 ≈ 数年，
#     实测容器跑 21 小时连 Skip-gram 第一阶段都没跑完。
#   - -N 5000：先截 5000 行（约 32 万 token）。实测 CPU 约 1 分 45 秒（rnn::Train 热循环
#     和 BBPE 学 merge 都优化过一版，墙钟随同机其他程序的负载浮动）。权重落盘后每次
#     启动秒级加载；想再快就 -N 调小或 -EAll 1。
#   - 全量 55 万行（约 3570 万 token）联合训练一轮 ≈ 半小时 CPU 时间 —— 这是这套
#     自研框架 CPU 实现的现实上限：扩大模型靠加 -N（样本量），不是靠加轮数。
# -S 不带 R：有权重文件就直接 Load（秒级启动），没有才训一遍再存（见 compose 的挂载）。
ENTRYPOINT ["/usr/local/bin/wuki-entrypoint.sh"]
CMD ["-S", "9000", "-C", "corpus.txt", "-N", "5000"]
