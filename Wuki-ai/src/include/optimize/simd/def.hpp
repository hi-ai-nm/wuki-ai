#pragma once
#ifndef DEF_HPP
#define DEF_HPP

#include <cstddef>

#if defined(__GNUC__) || defined(__clang__)
    /// 一个 SIMD 寄存器（x86: SSE / arm64: NEON），装 4 个 float
    using V = float __attribute__((vector_size(16)));
    static_assert(sizeof(V) == 16, "SIMD 寄存器宽度必须是 16 字节");
    inline constexpr std::size_t width = sizeof(V) / sizeof(float);
#else
    /// 标量回退：宽度 1，下面所有内核照常工作
    inline constexpr std::size_t width = 1;
    using V = float;
#endif

#if defined(__GNUC__) || defined(__clang__)
    /// 非对齐加载 width 个 float
    inline V loadu(const float* p) noexcept {
        V v;
        __builtin_memcpy(&v, p, sizeof(V));
        return v;
    }

    /// 非对齐写出 width 个 float
    inline void storeu(float* p, const V v) noexcept {
        __builtin_memcpy(p, &v, sizeof(V));
    }

    /// 标量广播到所有 lane
    inline V splat(const float x) noexcept {
        return V{x, x, x, x};
    }

    /// 逐 lane 取绝对值（清符号位）
    inline V vabs(const V v) noexcept {
        using VI = int __attribute__((vector_size(16)));
        VI bits;
        __builtin_memcpy(&bits, &v, sizeof(bits));
        bits &= VI{0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
        V out;
        __builtin_memcpy(&out, &bits, sizeof(out));
        return out;
    }

    /// 横向求和：4 个 lane 加成 1 个标量
    inline float hsum(const V v) noexcept {
        return v[0] + v[1] + v[2] + v[3];
    }
#else
    inline V loadu(const float* p) noexcept {
        return *p;
    }
    inline void storeu(float* p, const V v) noexcept {
        *p = v;
    }
    inline V splat(const float x) noexcept {
        return x;
    }
    inline V vabs(const V v) noexcept {
        return v < 0.0f ? -v : v;
    }
    inline float hsum(const V v) noexcept {
        return v;
    }
#endif

#endif