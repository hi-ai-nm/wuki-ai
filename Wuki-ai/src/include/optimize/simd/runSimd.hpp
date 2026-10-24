#pragma once
#ifndef RUNSIMD_HPP
#define RUNSIMD_HPP

#include "def.hpp"

/*
 * Simd::Runtime —— 本项目唯一的一组 SIMD 内核
 *
 * 为什么不用 <experimental/simd>：
 *   它是 GCC libstdc++ 的实验性 TS，macOS 的 libc++（Apple clang）和 MSVC 都
 *   没有这个头文件。本项目在 macOS 上开发、在 Ubuntu(GCC) 的容器里构建，
 *   写死它等于换台机器就编不过（实测 Apple clang 直接报
 *   "no member named 'experimental' in namespace 'std'"）。
 *   这里改用 GCC / Clang 都支持的 __attribute__((vector_size(16)))：
 *   x86-64 上落成 SSE，arm64 上落成 NEON，宽度固定 4 个 float；
 *   既不支持该扩展、又没被识别成 GCC/Clang 的编译器（如 MSVC）走标量回退，
 *   行为一致，只是不加速。
 *
 * 约定：
 *   - 所有函数 inline、无状态、不分配内存，指针由调用方保证有效；
 *   - n 是元素个数，尾部不足一个寄存器宽度的部分用标量收尾，不越界读写；
 *   - 指针不要求对齐：load/store 走 memcpy，编译器会生成对应的（非对齐）向量指令。
 *
 * 数值说明：
 *   SIMD 求和会改变浮点累加顺序（每 4 个一组、组内独立累加后横向相加），
 *   和标量逐个累加的结果可能差 1 个 ulp 量级。对训练 / 推理没有可观察影响，
 *   但不要拿它做「逐位复现」的断言。
 */
namespace Simd::Runtime {

    /**
     * 点积 Σ a[i]·b[i]
     *
     * 用 4 个累加器：一条 FMA 依赖链要等上一拍的延迟，4 条并行刚好把这个延迟藏住
     * （arm64 / x86 的 FMA 延迟通常是 3~4 拍）。其余部分走单个累加器，最后标量收尾。
     */
    inline float dot(const float* a, const float* b, const std::size_t n) noexcept {
        V acc0 = splat(0.0f);
        V acc1 = splat(0.0f);
        V acc2 = splat(0.0f);
        V acc3 = splat(0.0f);

        std::size_t i = 0;
        const std::size_t stride = width * 4;
        for (; i + stride <= n; i += stride) {
            acc0 += loadu(a + i) * loadu(b + i);
            acc1 += loadu(a + i + width) * loadu(b + i + width);
            acc2 += loadu(a + i + width * 2) * loadu(b + i + width * 2);
            acc3 += loadu(a + i + width * 3) * loadu(b + i + width * 3);
        }

        V acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i + width <= n; i += width) {
            acc += loadu(a + i) * loadu(b + i);
        }

        float s = hsum(acc);
        for (; i < n; ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    /// Σ |a[i]|（SGD 里按「连出去的权重绝对值和」归一用）
    inline float sum_abs(const float* a, const std::size_t n) noexcept {
        V acc = splat(0.0f);
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            acc += vabs(loadu(a + i));
        }

        float s = hsum(acc);
        for (; i < n; ++i) {
            s += a[i] < 0.0f ? -a[i] : a[i];
        }
        return s;
    }

    /// Σ a[i]²（向量长度 / 范数）
    inline float sum_sq(const float* a, const std::size_t n) noexcept {
        V acc = splat(0.0f);
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            const V v = loadu(a + i);
            acc += v * v;
        }

        float s = hsum(acc);
        for (; i < n; ++i) {
            s += a[i] * a[i];
        }
        return s;
    }

    /// Σ (a[i] - b[i])²（最近邻的欧氏距离平方；不用开方）
    inline float distance_sq(const float* a, const float* b, const std::size_t n) noexcept {
        V acc = splat(0.0f);
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            const V d = loadu(a + i) - loadu(b + i);
            acc += d * d;
        }

        float s = hsum(acc);
        for (; i < n; ++i) {
            const float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    /// y[i] += alpha · x[i]（权重更新 / 目标回填的主力）
    inline void axpy(float* y, const float* x, const float alpha, const std::size_t n) noexcept {
        const V va = splat(alpha);
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            storeu(y + i, loadu(y + i) + va * loadu(x + i));
        }
        for (; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    }

    /// y[i] -= x[i]（目标 → 误差：激活值统一相减）
    inline void sub_inplace(float* y, const float* x, const std::size_t n) noexcept {
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            storeu(y + i, loadu(y + i) - loadu(x + i));
        }
        for (; i < n; ++i) {
            y[i] -= x[i];
        }
    }

    /// y[i] *= alpha
    inline void scale_inplace(float* y, const float alpha, const std::size_t n) noexcept {
        const V va = splat(alpha);
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            storeu(y + i, loadu(y + i) * va);
        }
        for (; i < n; ++i) {
            y[i] *= alpha;
        }
    }

    /**
     * 一对向量的耦合更新（Skip-gram 的 delta rule）：
     *     a[i] = a[i] + g · b[i]
     *     b[i] = b[i] + g · a[i]（用更新前的 a[i]）
     *
     * 两边必须都用对方的旧值，所以先各取一份旧向量再写回 —— 和标量写法逐元素一致。
     */
    inline void paired_update(float* a, float* b, const float g, const std::size_t n) noexcept {
        const V vg = splat(g);
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            const V va = loadu(a + i);
            const V vb = loadu(b + i);
            storeu(a + i, va + vg * vb);
            storeu(b + i, vb + vg * va);
        }
        for (; i < n; ++i) {
            const float oldA = a[i];
            const float oldB = b[i];
            a[i] = oldA + g * oldB;
            b[i] = oldB + g * oldA;
        }
    }

    /**
     * 一层加权求和：y[j] = Σ_i W[i·m + j] · x[i] + b[j]，j = 0..m-1
     *
     * W 按 [n][m] 连续存储（第 i 行 = 第 i 个输入对所有神经元的权重）。
     * 这个布局和 nn / rnn 的 [输出][输入] 不一样（那两个是「一个神经元一行」），
     * 所以它们用的是上面的 dot / axpy；本函数保留给「输入在前」的调用方。
     */
    inline void layer_forward(const float* W, const float* x, const float* b,
                              float* y, const std::size_t n, const std::size_t m) noexcept {
        std::size_t j = 0;
        for (; j + width <= m; j += width) {
            V acc = splat(0.0f);
            for (std::size_t i = 0; i < n; ++i) {
                acc += loadu(W + i * m + j) * x[i];  // x[i] 标量自动广播
            }
            acc += loadu(b + j);
            storeu(y + j, acc);
        }

        // 尾部：不足一个 SIMD 宽度的神经元，标量回退
        for (; j < m; ++j) {
            float s = b[j];
            for (std::size_t i = 0; i < n; ++i) {
                s += W[i * m + j] * x[i];
            }
            y[j] = s;
        }
    }

}  // namespace Simd::Runtime

#endif  // RUNSIMD_HPP
