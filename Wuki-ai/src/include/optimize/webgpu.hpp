#pragma once
#ifndef WUKI_OPTIMIZE_WEBGPU_HPP
#define WUKI_OPTIMIZE_WEBGPU_HPP

#include <cstddef>
#include <cstdint>
#include <string>

/*
 * Wuki::Gpu —— WebGPU 计算后端（wgpu-native 的 C API）
 *
 * 定位：给「一次算一大片」的内核多一条路。本项目原来的优化层是 Simd::Runtime /
 * Simd::StudyTime（CPU 上 4 lane 的 SSE / NEON）。这一层不替换它，而是在调用方
 * 判断「这次的数据量够大、值得开 GPU」时接管，算完再把结果读回内存；任何一步
 * 出问题（没有适配器、驱动拒绝、超时、结果校验不过）就返回 false，调用方原样
 * 走回 SIMD —— 所以「有 GPU 更快」和「没 GPU 也能跑」是同一份代码。
 *
 * 为什么不是所有函数都无脑上 GPU：
 *   一次 WebGPU 调用 = 建 buffer + 写显存 + 提交 + 等 fence + 映射读回，
 *   固定开销在 0.1~1 ms 量级（见 wuki-gpu-bench 的实测）。而本项目的默认模型
 *   只有 8 维、30/60 个神经元，一个点积只要几十纳秒 —— 上 GPU 反而慢几十倍。
 *   所以这里所有入口都配合 ShouldUse(elements) 使用：Auto 模式下小规模直接返回
 *   false，调用方继续用 SIMD。想强制验证 GPU 路径可以 WUKI_GPU=force。
 *
 * 线程模型：所有入口内部串行（一把全局锁）。当前调用方（HTTP 服务 + 训练主线程）
 * 不是高并发场景，先保证正确；以后要做多流再拆锁。
 *
 * 许可证：本层依赖 wgpu-native（Apache-2.0 / MIT 双许可，本项目按 Apache-2.0 使用）
 * 和 webgpu.h（BSD-3-Clause）。许可证原文见 src/external/ 下对应文件，
 * 合规说明见 README 的「致谢 / 第三方许可证」。
 */
namespace Wuki::Gpu {

    /// 运行模式（可用 WUKI_GPU=off|auto|force 设置，见 ConfigureFromEnv）
    enum class Mode : unsigned char {
        Off = 0,    ///< 完全不碰 GPU，全部走 SIMD
        Auto = 1,   ///< 数据量 >= MinElements 才走 GPU（默认）
        Force = 2,  ///< 只要适配器可用就走 GPU（忽略规模阈值，调试 / 基准用）
    };

    /// 走 GPU 的调用统计（Info() / 基准程序里用）
    struct Stats {
        std::uint64_t ops = 0;    ///< 成功在 GPU 上完成的调用次数
        std::uint64_t bytes = 0;  ///< 这些调用搬运的字节数（上传 + 读回）
    };

    /**
     * 探测并初始化 WebGPU 设备（第一次调用时初始化，之后复用）
     *
     * 没有编译进 WebGPU 支持（WUKI_WEBGPU=OFF / 找不到 SDK）、没有可用适配器、
     * 设备请求失败时都返回 false，并把原因记下来（见 Error()）。
     * 这个函数是幂等的，失败之后不会每次重试（避免训练循环里反复卡初始化超时）。
     */
    bool Available();

    /// 上一次初始化失败的原因；可用时返回空串
    const char* Error();

    /// 一行状态：可用时是「适配器名 (后端)」，不可用时是失败原因
    std::string Info();

    Mode GetMode();
    void SetMode(Mode mode);

    /// Auto 模式的规模阈值（元素个数，默认见 .cpp 里 kDefaultMinElements）
    std::size_t MinElements();
    void SetMinElements(std::size_t elements);

    /**
     * 这次该不该试 GPU
     *
     * 只做「模式 + 阈值 + 设备可用」的快速判断，不保证后续内核一定成功；
     * 内核返回 false 时调用方仍然要回退 SIMD。
     */
    bool ShouldUse(std::size_t elements);

    /// 从环境变量读配置：WUKI_GPU=off|auto|force，WUKI_GPU_MIN=<元素数>
    void ConfigureFromEnv();

    Stats GetStats();
    void ResetStats();

    /// 释放设备 / 管线 / 缓冲（进程退出前可调，不调也不影响正确性）
    void Shutdown();

    // ========================================================================
    // 内核
    //
    // 约定和 Simd::Runtime 一致：指针由调用方保证有效、n / rows / cols 非零，
    // 函数不抛异常。返回值 = 「这次确实在 GPU 上算完了」，false = 没算 / 算失败，
    // 调用方必须自己再跑一遍 CPU 版本（不要读输出缓冲区）。
    // ========================================================================

    /// out = Σ a[i]·b[i]（单 workgroup 归约；n 很大时才有意义）
    bool Dot(const float* a, const float* b, std::size_t n, float* out);

    /// y[i] += alpha · x[i]
    bool Axpy(float* y, const float* x, float alpha, std::size_t n);

    /**
     * 一层前向：y[j] = b[j] + Σ_i W[i·cols + j] · x[i]
     *
     * 布局和 Simd::Runtime::layer_forward 相同：W 是 [rows][cols]，x 有 rows 个，
     * y / b 有 cols 个。b 可以传 nullptr（按全 0 偏置处理）。
     */
    bool Gemv(const float* W, const float* x, const float* b, float* y,
              std::size_t rows, std::size_t cols);

    /**
     * 目标回推：out[m] = scale[m] · Σ_r W[r·cols + m] · u[r]
     *
     * 对应 Simd::StudyTime::backprop_targets（nn 的 1/Σ|w| 就装在 scale 里）。
     * scale 可以传 nullptr（按全 1 处理）。
     */
    bool Gemt(const float* W, const float* u, const float* scale, float* out,
              std::size_t rows, std::size_t cols);

    /// 权重更新：W[r·cols + m] += lr · delta[r] · prev[m]（读改写，原地返回）
    bool Rank1(float* W, const float* delta, const float* prev, float lr,
               std::size_t rows, std::size_t cols);

    /// 一次算完「一个查询向量到整张表的平方距离」：out[j] = Σ_d (q[d] − table[j·cols + d])²
    bool DistanceAll(const float* q, const float* table, float* out,
                     std::size_t rows, std::size_t cols);

    /// Skip-gram 的成对更新：a += g·b，b += g·a（两边都用更新前的值）
    bool PairUpdate(float* a, float* b, float g, std::size_t n);

}  // namespace Wuki::Gpu

#endif  // WUKI_OPTIMIZE_WEBGPU_HPP
