# Wuki AI

一个基于 C++ 的 AI 框架 + 实现，用于构建和训练 AI 模型。

## 特点

- 使用纯 C++ 编写，性能高
- 使用 MIT 许可证，可以自由使用和修改
- 支持 SIMD 指令集加速

## 提示

- 本项目仍在开发中，可能存在一些问题
- **此项目使用自创架构和学习算法，*不保证稳定性***
- **切勿用于生产环境**，因为 性能+稳定性 都不行
- 本项目使用了 cpp-httplib 和 nlohmann/json，为**静态链接**
- **请勿用于任何违法用途**

## 使用方法

下载源码，使用 CMake 构建项目，然后编译和运行。

（编译后，在项目根目录运行./Wuki-ai/Bin/wuki-ai -H可获得详细用法）

## WebGPU 加速（可选）

优化层有两条路：`Simd::Runtime` / `Simd::StudyTime`（CPU，默认）和 `Wuki::Gpu`（WebGPU）。
GPU 后端是**按规模自动启用**的：只有单块矩阵 ≥ 16M 个元素才走 GPU，小矩阵继续用 SIMD
（一次 WebGPU 往返的固定开销约 0.4 ms，本项目默认模型太小，上 GPU 只会更慢）。

```bash
# 默认 AUTO：找得到 SDK 就编进去，找不到就自动降级成纯 SIMD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 强制 / 关闭
cmake -S . -B build -DWUKI_WEBGPU=ON    # 必须可用，否则配置失败
cmake -S . -B build -DWUKI_WEBGPU=OFF   # 完全不碰 WebGPU

# 运行时
WUKI_GPU=force ./Wuki-ai/Bin/wuki-ai -R   # 强制走 GPU（忽略规模阈值，调试用）
WUKI_GPU=off   ./Wuki-ai/Bin/wuki-ai -R   # 强制只用 SIMD
WUKI_GPU_MIN=4194304 ./Wuki-ai/Bin/wuki-ai -R  # 自己定阈值
./Wuki-ai/Bin/wuki-ai -V                  # 看适配器 / 模式 / 阈值
./Wuki-ai/Bin/wuki-gpu-bench              # SIMD vs GPU 实测，标定本机阈值
```

SDK 来源（按顺序）：`-DWUKI_WEBGPU_SDK_DIR=...` → 仓库里的 `third_party/wgpu-native/`
→ 下载固定版本的官方预编译包（按 SHA256 校验，缓存在 `build/_deps/`）。

## 致谢

- `cpp-httplib`
- `nlohmann/json`
- `wgpu-native`（WebGPU 实现，Apache-2.0 / MIT 双许可，本项目按 Apache-2.0 使用）
- `webgpu.h`（WebGPU C API 头文件，BSD-3-Clause，"WebGPU native" developers）

## 第三方许可证

| 组件 | 许可证 | 原文 |
| --- | --- | --- |
| cpp-httplib | MIT | `Wuki-ai/src/external/cpp-httplib-LICENSE.txt` |
| nlohmann/json | MIT | `Wuki-ai/src/external/nlohmann-json-LICENSE.txt` |
| wgpu-native | Apache-2.0（或 MIT，本项目选 Apache-2.0） | `Wuki-ai/src/external/wgpu-native-LICENSE.APACHE.txt`、`...-LICENSE.MIT.txt` |
| webgpu.h（webgpu-headers） | BSD-3-Clause | `Wuki-ai/src/external/webgpu-headers-LICENSE.txt` |

本项目自身代码仍为 MIT（见 `LICENSE`）。wgpu-native 以未修改的官方预编译包形式分发，
没有改动其源码；其 Apache-2.0 许可要求保留的版权 / 许可声明在 `src/external/` 下随包附带，
README 中也有对应署名。`Wuki::Gpu` 这一层只是调用它的 C API，属于本项目的新增代码。
