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
CMD ["./Bin/wuki-ai"]