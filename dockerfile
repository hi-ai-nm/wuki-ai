# ================= 阶段1：构建 =================
FROM ubuntu:24.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libdrogon-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Wuki-ai .

# 编译（CMake 会自动创建 /app/Bin 并放入所有产物）
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)

# ================= 阶段2：运行 =================
FROM ubuntu:24.04

# 运行阶段只需要运行库，不需要 build-essential 和 cmake
RUN apt-get update && apt-get install -y \
    libdrogon-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录为 Bin，这样 $ORIGIN 的 rpath 才能找到同目录的动态库
WORKDIR /app/Bin

# 关键修复：把整个 /app/Bin 目录（包含 exe 和两个 .so）全部拷过来
COPY --from=builder /app/Bin /app/Bin

EXPOSE 8080

CMD ["./wuki-ai"]