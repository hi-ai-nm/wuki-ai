FROM ubuntu:22.04 AS builder

# 工作目录
WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 复制代码
COPY Wuki-ai .

# 配置 + 编译（-S / -B 是幂等的：build 已经在也不会像 mkdir 那样报 File exists）
RUN cmake -S . -B build && cmake --build build -j"$(nproc)"

# 运行
# 用 ENTRYPOINT 而不是 CMD：CMD 会被 docker run 后面的参数整个替换掉（变成去 exec 那个路径），
# ENTRYPOINT 则是「固定跑 wuki-ai，后面的参数原样当 argv[1..] 传进去」——
# 所以 `docker run --rm -v "$PWD/Bin:/out" docker_run /out/drnn.model.json`
# 就是「把训练好的模型写到挂载进来的目录」，不带参数时和以前一样用默认路径
ENTRYPOINT ["./Bin/wuki-ai"]