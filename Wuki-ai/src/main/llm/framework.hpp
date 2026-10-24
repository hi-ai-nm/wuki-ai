#pragma once
#ifndef LLM_FRAMEWORK_HPP
#define LLM_FRAMEWORK_HPP

#include "Tools.hpp"
#include "includeMain.hpp"

/*
 * SIMD 内核（实现见 src/include/optimize/simd.hpp）
 *
 * nn / rnn 的前向、反向热循环，以及 wet 的 Skip-gram / 最近邻 / 归一化都走它。
 * 在 framework 这一层统一 include：framework 的各个模块（architecture / wet ...）
 * 都能直接用 Simd::Runtime，不用各自猜相对路径。宽度和可用性见那个头的说明
 * （x86-64 → SSE，arm64 → NEON，不支持的编译器自动标量回退）。
 */
#include "optimize/simd.hpp"

#endif
