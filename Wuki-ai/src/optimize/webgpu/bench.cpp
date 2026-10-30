/*
 * wuki-gpu-bench —— SIMD 与 WebGPU 的实测对比
 *
 * 目的不是刷一个好看的倍率，而是回答两个工程问题：
 *   1. 一次 WebGPU 往返的固定开销是多少（小矩阵上 GPU 到底慢多少）；
 *   2. 矩阵大到多少元素时 GPU 才开始赢（Auto 模式的阈值就按这个定）。
 *
 * CPU 侧故意不用 Simd::Runtime::layer_forward / backprop_targets —— 那几个函数
 * 已经接了 GPU 分派，拿它们当基线会自己和自己比。这里直接用 -O3 的标量循环，
 * 编译器会自动向量化，量级和 Simd 一致（略慢一点，对结论没有影响）。
 *
 * 用法：./Bin/wuki-gpu-bench [--quick]
 *   退出码：0 正常跑完；1 WebGPU 不可用（只打印原因）。
 */

#include "optimize/webgpu.hpp"
#include "optimize/simd.hpp"  // CPU 基线和线上同一条 SIMD 路径（不是另写一份标量循环）

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

    using Clock = std::chrono::steady_clock;

    std::vector<float> randomFloats(const std::size_t n, const float lo, const float hi) {
        std::mt19937 gen(20261030u);
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<float> out(n);
        for (auto& v : out) {
            v = dist(gen);
        }
        return out;
    }

    /// 跑 reps 次取平均（毫秒）
    template <typename Fn>
    double timeMs(const int reps, Fn&& fn) {
        // 预热一次，别把首次的缺页 / 缓存冷启动算进去
        fn();
        const auto start = Clock::now();
        for (int i = 0; i < reps; ++i) {
            fn();
        }
        const auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / reps;
    }

    /// 按元素个数定重复次数：总量控制在 3e7 左右，单次测量几十毫秒
    int repsFor(const std::size_t elements) {
        const double reps = 3.0e7 / static_cast<double>(std::max<std::size_t>(elements, 1));
        return std::clamp(static_cast<int>(reps), 1, 200);
    }

    double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
        double worst = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
        }
        return worst;
    }

    void report(const char* name, std::size_t elements, double cpuMs, double gpuMs, double diff) {
        const double speedup = gpuMs > 0.0 ? cpuMs / gpuMs : 0.0;
        std::printf("%-10s %10zu  %10.3f ms  %10.3f ms  %7.2fx  maxdiff=%.2e\n", name, elements,
                    cpuMs, gpuMs, speedup, diff);
    }

    // ---------------- CPU 基线 ----------------
    //
    // 直接调线上那几个 SIMD 内核，但先把 GPU 模式切成 Off（否则 layer_forward 自己会
    // 分派到 GPU，变成自己和自己比）。这样基线就是线上真实的 CPU 性能。

    struct CpuMode {
        CpuMode() {
            m_old = Wuki::Gpu::GetMode();
            Wuki::Gpu::SetMode(Wuki::Gpu::Mode::Off);
        }
        ~CpuMode() {
            Wuki::Gpu::SetMode(m_old);
        }
        Wuki::Gpu::Mode m_old;
    };

    void cpuGemv(const float* W, const float* x, const float* b, float* y, std::size_t rows,
                 std::size_t cols) {
        CpuMode guard;
        Simd::Runtime::layer_forward(W, x, b, y, rows, cols);
    }

    void cpuGemt(const float* W, const float* u, const float* scale, float* out,
                 std::size_t rows, std::size_t cols) {
        CpuMode guard;
        Simd::StudyTime::backprop_targets(W, u, scale, out, rows, cols);
    }

    void cpuRank1(float* W, const float* delta, const float* prev, float lr, std::size_t rows,
                  std::size_t cols) {
        CpuMode guard;
        Simd::StudyTime::accumulate_weights(W, delta, prev, lr, rows, cols);
    }

    void cpuDistance(const float* q, const float* table, float* out, std::size_t rows,
                     std::size_t cols) {
        // 线上 Nearest / Sample 干了什么，这里就干什么：逐行 distance_sq
        for (std::size_t j = 0; j < rows; ++j) {
            out[j] = Simd::Runtime::distance_sq(q, table + j * cols, cols);
        }
    }

    void cpuDot(const float* a, const float* b, std::size_t n, float* out) {
        *out = Simd::Runtime::dot(a, b, n);
    }

    void cpuAxpy(float* y, const float* x, float alpha, std::size_t n) {
        Simd::Runtime::axpy(y, x, alpha, n);
    }

    void benchMatrix(const bool quick) {
        struct Shape {
            std::size_t rows;
            std::size_t cols;
        };
        std::vector<Shape> shapes = {{256, 256}, {1024, 1024}, {2048, 2048}, {4096, 4096}};
        if (!quick) {
            shapes.push_back({8192, 8192});
        }

        std::printf("\n== 矩阵内核（n×m 个元素，单次调用含上传 + 读回）==\n");
        std::printf("%-10s %10s  %13s  %13s  %8s  %s\n", "op", "elements", "CPU", "GPU", "speedup",
                    "check");

        for (const Shape& s : shapes) {
            const std::size_t n = s.rows * s.cols;
            auto W = randomFloats(n, -0.5f, 0.5f);
            auto x = randomFloats(s.rows, -1.0f, 1.0f);
            auto b = randomFloats(s.cols, -0.1f, 0.1f);
            auto yCpu = std::vector<float>(s.cols, 0.0f);
            auto yGpu = std::vector<float>(s.cols, 0.0f);

            const double cpuMs = timeMs(repsFor(s.rows * s.cols), [&] { cpuGemv(W.data(), x.data(), b.data(), yCpu.data(), s.rows, s.cols); });
            const double gpuMs = timeMs(repsFor(s.rows * s.cols), [&] {
                std::fill(yGpu.begin(), yGpu.end(), 0.0f);
                if (!Wuki::Gpu::Gemv(W.data(), x.data(), b.data(), yGpu.data(), s.rows, s.cols)) {
                    std::printf("GPU Gemv 失败: %s\n", Wuki::Gpu::Error());
                    return;
                }
            });
            report("gemv", n, cpuMs, gpuMs, maxAbsDiff(yCpu, yGpu));

            const double cpuMs2 = timeMs(repsFor(s.rows * s.cols), [&] { cpuGemt(W.data(), x.data(), b.data(), yCpu.data(), s.rows, s.cols); });
            const double gpuMs2 = timeMs(repsFor(s.rows * s.cols), [&] {
                std::fill(yGpu.begin(), yGpu.end(), 0.0f);
                if (!Wuki::Gpu::Gemt(W.data(), x.data(), b.data(), yGpu.data(), s.rows, s.cols)) {
                    std::printf("GPU Gemt 失败: %s\n", Wuki::Gpu::Error());
                    return;
                }
            });
            report("gemt", n, cpuMs2, gpuMs2, maxAbsDiff(yCpu, yGpu));

            auto delta = randomFloats(s.rows, -0.01f, 0.01f);
            auto prev = randomFloats(s.cols, -1.0f, 1.0f);
            auto wCpu = W;
            auto wGpu = W;
            cpuRank1(wCpu.data(), delta.data(), prev.data(), 0.01f, s.rows, s.cols);
            if (!Wuki::Gpu::Rank1(wGpu.data(), delta.data(), prev.data(), 0.01f, s.rows, s.cols)) {
                std::printf("GPU Rank1 失败: %s\n", Wuki::Gpu::Error());
            }
            const double rankDiff = maxAbsDiff(wCpu, wGpu);
            const double cpuMs3 = timeMs(repsFor(s.rows * s.cols), [&] { cpuRank1(wCpu.data(), delta.data(), prev.data(), 0.01f, s.rows, s.cols); });
            const double gpuMs3 = timeMs(repsFor(s.rows * s.cols), [&] {
                if (!Wuki::Gpu::Rank1(wGpu.data(), delta.data(), prev.data(), 0.01f, s.rows, s.cols)) {
                    std::printf("GPU Rank1 失败: %s\n", Wuki::Gpu::Error());
                }
            });
            report("rank1", n, cpuMs3, gpuMs3, rankDiff);

            auto dCpu = std::vector<float>(s.rows, 0.0f);
            auto dGpu = std::vector<float>(s.rows, 0.0f);
            auto table = randomFloats(n, -1.0f, 1.0f);
            auto q = randomFloats(s.cols, -1.0f, 1.0f);
            const double cpuMs4 = timeMs(repsFor(s.rows * s.cols), [&] { cpuDistance(q.data(), table.data(), dCpu.data(), s.rows, s.cols); });
            const double gpuMs4 = timeMs(repsFor(s.rows * s.cols), [&] {
                std::fill(dGpu.begin(), dGpu.end(), 0.0f);
                if (!Wuki::Gpu::DistanceAll(q.data(), table.data(), dGpu.data(), s.rows, s.cols)) {
                    std::printf("GPU DistanceAll 失败: %s\n", Wuki::Gpu::Error());
                }
            });
            report("distance", n, cpuMs4, gpuMs4, maxAbsDiff(dCpu, dGpu));
        }
    }

    void benchVectors(const bool quick) {
        std::printf("\n== 向量内核（固定开销占比更大）==\n");
        std::printf("%-10s %10s  %13s  %13s  %8s  %s\n", "op", "elements", "CPU", "GPU", "speedup",
                    "check");

        std::vector<std::size_t> sizes = {4096, 262144, 1048576, 4194304};
        if (!quick) {
            sizes.push_back(16777216);
        }
        for (const std::size_t n : sizes) {
            auto a = randomFloats(n, -1.0f, 1.0f);
            auto c = randomFloats(n, -1.0f, 1.0f);

            float dotCpu = 0.0f;
            float dotGpu = 0.0f;
            const double dotCpuMs = timeMs(repsFor(n), [&] { cpuDot(a.data(), c.data(), n, &dotCpu); });
            const double dotGpuMs = timeMs(repsFor(n), [&] {
                dotGpu = 0.0f;
                if (!Wuki::Gpu::Dot(a.data(), c.data(), n, &dotGpu)) {
                    std::printf("GPU Dot 失败: %s\n", Wuki::Gpu::Error());
                }
            });
            report("dot", n, dotCpuMs, dotGpuMs, std::fabs(static_cast<double>(dotCpu) - dotGpu));

            auto yCpu = a;
            auto yGpu = a;
            cpuAxpy(yCpu.data(), c.data(), 0.5f, n);
            if (!Wuki::Gpu::Axpy(yGpu.data(), c.data(), 0.5f, n)) {
                std::printf("GPU Axpy 失败: %s\n", Wuki::Gpu::Error());
            }
            const double axpyDiff = maxAbsDiff(yCpu, yGpu);
            const double axpyCpuMs = timeMs(repsFor(n), [&] { cpuAxpy(yCpu.data(), c.data(), 0.5f, n); });
            const double axpyGpuMs = timeMs(repsFor(n), [&] {
                if (!Wuki::Gpu::Axpy(yGpu.data(), c.data(), 0.5f, n)) {
                    std::printf("GPU Axpy 失败: %s\n", Wuki::Gpu::Error());
                }
            });
            report("axpy", n, axpyCpuMs, axpyGpuMs, axpyDiff);
        }
    }

    /// 固定开销：一个 64 元素的调用跑 100 次，平均每次就是「往返 + 提交」的成本
    void benchOverhead() {
        constexpr std::size_t n = 64;
        auto a = randomFloats(n, -1.0f, 1.0f);
        auto b = randomFloats(n, -1.0f, 1.0f);
        auto y = a;
        const double ms = timeMs(100, [&] {
            if (!Wuki::Gpu::Axpy(y.data(), b.data(), 0.1f, n)) {
                std::printf("GPU Axpy 失败: %s\n", Wuki::Gpu::Error());
            }
        });
        std::printf("\n== 固定开销 ==\n");
        std::printf("一次 64 元素的 GPU 往返：%.3f ms（这就是小矩阵不该上 GPU 的原因）\n", ms);
    }

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    bool overheadOnly = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (std::strcmp(argv[i], "--overhead") == 0) {
            overheadOnly = true;  // 只量固定开销，调 wait 策略时用
        }
    }

    Wuki::Gpu::ConfigureFromEnv();
    std::printf("Wuki GPU 基准\n");
    std::printf("设备    : %s\n", Wuki::Gpu::Info().c_str());
    std::printf("模式    : %d（0=off 1=auto 2=force）\n", static_cast<int>(Wuki::Gpu::GetMode()));
    std::printf("阈值    : %zu 个元素（WUKI_GPU_MIN 可改）\n", Wuki::Gpu::MinElements());
    if (!Wuki::Gpu::Available()) {
        std::printf("WebGPU 不可用，无法做对比。原因：%s\n", Wuki::Gpu::Error());
        return 1;
    }

    // 基准里就是要每次都跑 GPU，不受 auto 阈值影响
    Wuki::Gpu::SetMode(Wuki::Gpu::Mode::Force);

    benchOverhead();
    if (!overheadOnly) {
        benchVectors(quick);
        benchMatrix(quick);
    }

    Wuki::Gpu::Shutdown();  // WUKI_GPU_TRACE=1 时这里会打印阶段耗时
    const auto stats = Wuki::Gpu::GetStats();
    std::printf("\nGPU 调用 %llu 次，搬运 %.1f MB\n",
                static_cast<unsigned long long>(stats.ops),
                static_cast<double>(stats.bytes) / (1024.0 * 1024.0));
    std::printf("提示：把上面第一次出现 speedup > 1 的元素数四舍五入，就是适合本机的 WUKI_GPU_MIN。\n");
    return 0;
}
