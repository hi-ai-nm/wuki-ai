#pragma once
#ifndef RUNSTUDY_HPP
#define RUNSTUDY_HPP

#include "def.hpp"
#include "runSimd.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

/*
 * Simd::StudyTime —— 对比学习算法（Study / Train）专用的 SIMD 层
 *
 * 这个文件只放「无状态、只吃指针」的数值内核；nn / rnn 的控制流、下标表
 * （aStart / bStart / wStart）、缓存布局、错误码全都留在各自类里。这样
 * runStudy.hpp 不依赖 nn::NeuNet / rnn::RNN 的定义，单独就能测。
 *
 * 四步算法 → 本文件内核的对应关系：
 *   前向       z = Wx + b                        → Simd::Runtime::dot（已有）+ activate
 *   第 1 步    输出目标铺开 + MSE                → mse（内部 distance_sq）
 *   第 2 步    t_prev[m] = Σ_r W[r][m]·u[r]      → backprop_targets
 *              （u 里已经乘好 1/Σ|w|；nn 用 outScale 乘 1/Σ|w|）
 *   第 3 步    误差 = 目标 - 实际激活            → 并入 delta
 *   第 4 步    δ = 误差·a'(z)；w += lr·δ·prev    → delta + accumulate_weights
 *              b += blr·δ                        → accumulate_bias
 *
 * 数值约定：
 *   - 激活函数用有界误差的向量近似（tanh / sigmoid / exp），默认开启；
 *     编译时定义 WUKI_SIMD_MATH=0 可退回逐 lane 的 std::tanh / std::exp。
 *   - SIMD 会改变浮点累加顺序；不要拿它做逐位复现的断言。
 *   - 所有函数 inline、不分配内存、尾部标量收尾、不要求指针对齐。
 */
namespace Simd::StudyTime {

    /// 激活函数种类。名字 → 枚举只在每层入口解析一次，不在神经元循环里比字符串。
    enum class ActKind : unsigned char { Scth, Relu, Silu, AltSigmoid };

    /**
     * 名字 → 枚举
     *
     * 未知 / 空串按 ReLU 处理 —— 和 nn / rnn 的 Run、Study 里那个默认分支一致。
     * nn::set_ActFun 仍然拒绝 "scth"，这里只是把 Run 已经支持的那四种都认下来。
     */
    inline ActKind ParseActKind(const std::string& name) noexcept {
        if (name == "scth") {
            return ActKind::Scth;
        }
        if (name == "relu") {
            return ActKind::Relu;
        }
        if (name == "silu") {
            return ActKind::Silu;
        }
        if (name == "alt-sigmoid") {
            return ActKind::AltSigmoid;
        }
        return ActKind::Relu;
    }

    // ========================================================================
    // 向量数学
    // ========================================================================

    // 标量参考实现定义在本文件中段；下面的精确回退分支要调用它们
    inline float activate_scalar(ActKind kind, float z) noexcept;
    inline float deriv_scalar(ActKind kind, float z, float a) noexcept;

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(WUKI_SIMD_MATH) || WUKI_SIMD_MATH)
    using VI = int __attribute__((vector_size(16)));

    /// 位模式转换（float 向量 ↔ int 向量），走 memcpy 避开严格别名问题
    inline VI to_vi(const V v) noexcept {
        VI bits;
        __builtin_memcpy(&bits, &v, sizeof(bits));
        return bits;
    }
    inline V to_v(const VI bits) noexcept {
        V v;
        __builtin_memcpy(&v, &bits, sizeof(v));
        return v;
    }

    /// 逐 lane 选择：mask 为全 1 的 lane 取 a，否则取 b
    ///
    /// mask 的类型不能写死成 VI：GCC 的向量比较返回 int 向量，Clang 返回的
    /// 却是「位模式为 0 / 全 1 的 float 向量」。这里统一按位拷贝成 VI —— 千万
    /// 不能做数值转换（NaN → int 是未定义行为，优化一开结果就飘）。
    template <typename M>
    inline V vselect(const M mask, const V a, const V b) noexcept {
        static_assert(sizeof(M) == sizeof(VI), "比较结果的宽度必须和一个 SIMD 寄存器一致");
        VI m;
        __builtin_memcpy(&m, &mask, sizeof(m));
        return to_v((to_vi(a) & m) | (to_vi(b) & ~m));
    }

    inline auto visnan(const V v) noexcept {
        return v != v;
    }

    /**
     * e^{-t}，要求 t >= 0
     *
     * 范围规约：t = k·ln2 + r，|r| <= ln2/2 ≈ 0.3466，e^{-r} 用 7 阶 Taylor；
     * 2^{-k} 用整数向量的指数位直接构造（k 限制在 [0,126]，t 先 clamp 到 87，
     * 再大就按 ~1.6e-38 处理 —— 和真正的下溢差 1e-38 量级，远小于任何验收阈值）。
     *
     * NaN：先换成 0 再算（NaN → int 是未定义行为），末尾 select 回 NaN。
     */
    inline V vexp_neg(const V t) noexcept {
        const auto nanMask = visnan(t);
        V x = vselect(nanMask, splat(0.0f), t);
        x = vselect(x > splat(87.0f), splat(87.0f), x);

        const V kf = x * splat(1.4426950408889634f) + splat(0.5f);  // t/ln2 四舍五入
        // 注意：GNU 向量扩展里 (VI)float向量 是**按位重解释**，不是逐元素转整数；
        // 这里必须用 __builtin_convertvector（GCC 9+ / Clang 都支持）。
        const VI ki = __builtin_convertvector(kf, VI);
        const V k = __builtin_convertvector(ki, V);
        const V r = x - k * splat(0.6931471805599453f);             // ln2

        // e^{-r} = 1 - r + r²/2 - r³/6 + ... - r⁷/5040
        V p = splat(-1.0f / 5040.0f);
        p = p * r + splat(1.0f / 720.0f);
        p = p * r + splat(-1.0f / 120.0f);
        p = p * r + splat(1.0f / 24.0f);
        p = p * r + splat(-1.0f / 6.0f);
        p = p * r + splat(0.5f);
        p = p * r + splat(-1.0f);
        p = p * r + splat(1.0f);

        const VI expBits = __builtin_convertvector(splat(127.0f) - k, VI) << 23;  // 2^{-k}
        const V e = p * to_v(expBits);
        return vselect(nanMask, splat(std::numeric_limits<float>::quiet_NaN()), e);
    }

    /**
     * σ(z) = 1/(1+e^{-z})
     *
     * 只用 e^{-|z|}，所以永远不会 exp 溢出：
     *   z >= 0 → 1/(1+e)，z < 0 → e/(1+e)
     */
    inline V vsigmoid(const V z) noexcept {
        const V e = vexp_neg(vabs(z));
        const V inv = splat(1.0f) / (splat(1.0f) + e);
        return vselect(z < splat(0.0f), e * inv, inv);
    }

    /**
     * 自创 tanh：a = 2·tanh(z/2)
     *
     * tanh(z/2) = (1-e^{-z})/(1+e^{-z})，取 e = e^{-|z|} 后按符号镜像：
     *   a = sign(z)·2·(1-e)/(1+e)
     * 这样饱和区（|z| 大）自然落到 ±2，和 std::tanh 的 ±1 再乘 2 一致。
     */
    inline V vscth(const V z) noexcept {
        const V e = vexp_neg(vabs(z));
        const V inv = splat(1.0f) / (splat(1.0f) + e);
        const V mag = splat(2.0f) * (splat(1.0f) - e) * inv;
        return vselect(z < splat(0.0f), -mag, mag);
    }

    /// ReLU；NaN → 0，和现有标量分支 `z > 0 ? z : 0` 一致
    inline V vrelu(const V z) noexcept {
        return vselect(z > splat(0.0f), z, splat(0.0f));
    }

    /// SiLU：z·σ(z)（向量近似路径不做除法，精度见文件头）
    inline V vsilu(const V z) noexcept {
        return z * vsigmoid(z);
    }

    /// 激活导数 a'(z)（按 lane）
    inline V vderiv(const ActKind kind, const V z, const V a) noexcept {
        switch (kind) {
            case ActKind::Scth: {  // a = 2·tanh(z/2) → a' = 1 - (a/2)²
                const V half = a * splat(0.5f);
                return splat(1.0f) - half * half;
            }
            case ActKind::Relu:
                return vselect(z > splat(0.0f), splat(1.0f), splat(0.0f));
            case ActKind::Silu: {  // a = z·σ(z) → a' = σ·(1 + z·(1-σ))
                const V s = vsigmoid(z);
                return s * (splat(1.0f) + z * (splat(1.0f) - s));
            }
            case ActKind::AltSigmoid:  // σ' = σ(1-σ)，a 就是 σ(z)
                return a * (splat(1.0f) - a);
        }
        return vselect(z > splat(0.0f), splat(1.0f), splat(0.0f));  // 未知按 ReLU
    }

#elif defined(__GNUC__) || defined(__clang__)
    // 有向量类型，但显式要求精确数学（WUKI_SIMD_MATH=0）：逐 lane 调 libm。
    // 循环宽度、尾部结构和近似分支完全一样，只是把超越函数换回 std::exp / std::tanh。
    inline V vexp_neg(const V t) noexcept {
        V r;
        for (std::size_t i = 0; i < width; ++i) {
            r[i] = std::exp(-t[i]);
        }
        return r;
    }
    inline V vsigmoid(const V z) noexcept {
        V r;
        for (std::size_t i = 0; i < width; ++i) {
            r[i] = 1.0f / (1.0f + std::exp(-z[i]));
        }
        return r;
    }
    inline V vscth(const V z) noexcept {
        V r;
        for (std::size_t i = 0; i < width; ++i) {
            r[i] = 2.0f * std::tanh(z[i] * 0.5f);
        }
        return r;
    }
    inline V vrelu(const V z) noexcept {
        V r;
        for (std::size_t i = 0; i < width; ++i) {
            r[i] = z[i] > 0.0f ? z[i] : 0.0f;
        }
        return r;
    }
    inline V vsilu(const V z) noexcept {
        V r;
        for (std::size_t i = 0; i < width; ++i) {
            r[i] = z[i] / (1.0f + std::exp(-z[i]));  // 和旧代码逐 lane 同一条公式
        }
        return r;
    }
    inline V vderiv(const ActKind kind, const V z, const V a) noexcept {
        V r;
        for (std::size_t i = 0; i < width; ++i) {
            r[i] = deriv_scalar(kind, z[i], a[i]);
        }
        return r;
    }

#else
    // 标量回退（MSVC）：V 就是 float，一行一个
    inline V vexp_neg(const V t) noexcept {
        return std::exp(-t);
    }
    inline V vsigmoid(const V z) noexcept {
        return 1.0f / (1.0f + std::exp(-z));
    }
    inline V vscth(const V z) noexcept {
        return 2.0f * std::tanh(z * 0.5f);
    }
    inline V vrelu(const V z) noexcept {
        return z > 0.0f ? z : 0.0f;
    }
    inline V vsilu(const V z) noexcept {
        return z / (1.0f + std::exp(-z));
    }
    inline V vderiv(const ActKind kind, const V z, const V a) noexcept {
        return deriv_scalar(kind, z, a);
    }
#endif

    // ========================================================================
    // 标量参考 / 单点实现
    // ========================================================================

    /// 单点激活（尾部 / 测试 / nn::ActDeriv 的公式来源）
    inline float activate_scalar(const ActKind kind, const float z) noexcept {
        switch (kind) {
            case ActKind::Scth:
                return std::tanh(z * 0.5f) * 2.0f;
            case ActKind::Relu:
                return z > 0.0f ? z : 0.0f;
            case ActKind::Silu:
                return z / (1.0f + std::exp(-z));
            case ActKind::AltSigmoid:
                return 1.0f / (1.0f + std::exp(-z));
        }
        return z > 0.0f ? z : 0.0f;
    }

    /// 单点激活导数（ActDeriv 的公式来源）
    inline float deriv_scalar(const ActKind kind, const float z, const float a) noexcept {
        switch (kind) {
            case ActKind::Scth: {
                const float half = a * 0.5f;
                return 1.0f - half * half;
            }
            case ActKind::Relu:
                return z > 0.0f ? 1.0f : 0.0f;
            case ActKind::Silu: {
                const float s = 1.0f / (1.0f + std::exp(-z));
                return s * (1.0f + z * (1.0f - s));
            }
            case ActKind::AltSigmoid:
                return a * (1.0f - a);
        }
        return z > 0.0f ? 1.0f : 0.0f;
    }

    // ========================================================================
    // 批量激活 / 导数
    // ========================================================================

    /// 整层激活：a[i] = f(z[i])
    inline void activate(const ActKind kind, const float* z, float* a, const std::size_t n) noexcept {
        std::size_t i = 0;
        switch (kind) {
            case ActKind::Scth:
                for (; i + width <= n; i += width) {
                    storeu(a + i, vscth(loadu(z + i)));
                }
                break;
            case ActKind::Relu:
                for (; i + width <= n; i += width) {
                    storeu(a + i, vrelu(loadu(z + i)));
                }
                break;
            case ActKind::Silu:
                for (; i + width <= n; i += width) {
                    storeu(a + i, vsilu(loadu(z + i)));
                }
                break;
            case ActKind::AltSigmoid:
                for (; i + width <= n; i += width) {
                    storeu(a + i, vsigmoid(loadu(z + i)));
                }
                break;
        }
        for (; i < n; ++i) {  // 尾部标量（用 libm 的精确写法，误差比向量路径更小）
            a[i] = activate_scalar(kind, z[i]);
        }
    }

    /**
     * 第 3 + 4 步的融合：out[i] = (err[i] - a[i]) · a'(z[i])
     *
     * 原来 nn::Study / rnn::Train 先在整张表上做「目标 - 激活」，再逐神经元
     * 乘导数；这里合成一趟，并且把导数改成整层向量算。
     *
     * @param err 未减激活的目标（nn 的 error / rnn 的 tg），out 可以和它同一个数组吗？
     *            不可以 —— 本函数按 lane 读到 err 后立刻写 out，虽然顺序上安全，
     *            但契约里明确要求 err 与 out 不重叠，避免以后改动踩坑。
     */
    inline void delta(const ActKind kind, const float* z, const float* a,
                      const float* err, float* out, const std::size_t n) noexcept {
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            const V av = loadu(a + i);
            const V zv = loadu(z + i);
            const V ev = loadu(err + i);
            storeu(out + i, (ev - av) * vderiv(kind, zv, av));
        }
        for (; i < n; ++i) {
            out[i] = (err[i] - a[i]) * deriv_scalar(kind, z[i], a[i]);
        }
    }

    // ========================================================================
    // 四步算法的数值内核
    // ========================================================================

    /// 第 1 步：MSE = Σ (goal[i] - act[i])²（尾部和距离平方共用同一个内核）
    inline float mse(const float* goal, const float* act, const std::size_t n) noexcept {
        return Simd::Runtime::distance_sq(goal, act, n);
    }

    /// out[i] = a[i] · b[i]（rnn 第 2 步：u = invWSum · tg）
    inline void mul_into(float* out, const float* a, const float* b, const std::size_t n) noexcept {
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            storeu(out + i, loadu(a + i) * loadu(b + i));
        }
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    /// y[i] += |x[i]|（nn 第 2 步：按列累计 Σ|w|，一行一行扫，避免转置）
    inline void accumulate_abs(float* y, const float* x, const std::size_t n) noexcept {
        std::size_t i = 0;
        for (; i + width <= n; i += width) {
            storeu(y + i, loadu(y + i) + vabs(loadu(x + i)));
        }
        for (; i < n; ++i) {
            y[i] += x[i] < 0.0f ? -x[i] : x[i];
        }
    }

    /**
     * 第 2 步：out[m] = Σ_r W[r·cols + m] · u[r]
     *
     * W 是 [rows][cols]（一个「当前层神经元」一行，正是 nn / rnn 的权重布局），
     * 所以这里沿 rows 方向累加、一次产出 width 个 m：
     *   - 4 个 row 累加器（跟 dot 的 4 累加器同理，藏 FMA 延迟）；
     *   - out 只写一次（旧实现是每个 row 对 tg 做 axpy，读改写 prevSize 次）；
     *   - outScale 非空时再逐 lane 乘 outScale[m]（nn 的 1/Σ|w|）。
     *
     * 约束：out 与 u / W 不重叠；out 与 outScale 不重叠。
     */
    inline void backprop_targets(const float* W, const float* u, const float* outScale,
                                 float* out, const std::size_t rows, const std::size_t cols) noexcept {
        std::size_t m = 0;
        for (; m + width <= cols; m += width) {
            V acc0 = splat(0.0f);
            V acc1 = splat(0.0f);
            V acc2 = splat(0.0f);
            V acc3 = splat(0.0f);

            std::size_t r = 0;
            for (; r + 4 <= rows; r += 4) {
                acc0 += loadu(W + (r + 0) * cols + m) * u[r + 0];
                acc1 += loadu(W + (r + 1) * cols + m) * u[r + 1];
                acc2 += loadu(W + (r + 2) * cols + m) * u[r + 2];
                acc3 += loadu(W + (r + 3) * cols + m) * u[r + 3];
            }

            V acc = (acc0 + acc1) + (acc2 + acc3);
            for (; r < rows; ++r) {
                acc += loadu(W + r * cols + m) * u[r];
            }

            if (outScale != nullptr) {
                acc *= loadu(outScale + m);
            }
            storeu(out + m, acc);
        }

        // 尾部：不足一个寄存器宽度的输出神经元
        for (; m < cols; ++m) {
            float s = 0.0f;
            for (std::size_t r = 0; r < rows; ++r) {
                s += W[r * cols + m] * u[r];
            }
            if (outScale != nullptr) {
                s *= outScale[m];
            }
            out[m] = s;
        }
    }

    /**
     * 第 4 步：dW[r·cols + m] += lr · delta[r] · prevAct[m]
     *
     * 一次更新 4 行：prevAct 的那个向量只载一次，喂给 4 条独立的 FMA 链。
     * 行的尾部退化成 Simd::Runtime::axpy（和原来逐行更新逐元素一致）。
     */
    inline void accumulate_weights(float* dW, const float* delta, const float* prevAct,
                                   const float lr, const std::size_t rows,
                                   const std::size_t cols) noexcept {
        std::size_t r = 0;
        for (; r + 4 <= rows; r += 4) {
            const V d0 = splat(delta[r + 0] * lr);
            const V d1 = splat(delta[r + 1] * lr);
            const V d2 = splat(delta[r + 2] * lr);
            const V d3 = splat(delta[r + 3] * lr);

            std::size_t m = 0;
            for (; m + width <= cols; m += width) {
                const V p = loadu(prevAct + m);

                float* w0 = dW + (r + 0) * cols + m;
                float* w1 = dW + (r + 1) * cols + m;
                float* w2 = dW + (r + 2) * cols + m;
                float* w3 = dW + (r + 3) * cols + m;

                storeu(w0, loadu(w0) + d0 * p);
                storeu(w1, loadu(w1) + d1 * p);
                storeu(w2, loadu(w2) + d2 * p);
                storeu(w3, loadu(w3) + d3 * p);
            }

            for (; m < cols; ++m) {
                const float p = prevAct[m];
                dW[(r + 0) * cols + m] += delta[r + 0] * lr * p;
                dW[(r + 1) * cols + m] += delta[r + 1] * lr * p;
                dW[(r + 2) * cols + m] += delta[r + 2] * lr * p;
                dW[(r + 3) * cols + m] += delta[r + 3] * lr * p;
            }
        }

        // 不足 4 行的尾巴
        for (; r < rows; ++r) {
            Simd::Runtime::axpy(dW + r * cols, prevAct, delta[r] * lr, cols);
        }
    }

    /// 第 4 步：dB[i] += blr · delta[i]（整层一条向量）
    inline void accumulate_bias(float* dB, const float* delta, const float blr,
                                const std::size_t n) noexcept {
        Simd::Runtime::axpy(dB, delta, blr, n);
    }

}  // namespace Simd::StudyTime

#endif  // RUNSTUDY_HPP
