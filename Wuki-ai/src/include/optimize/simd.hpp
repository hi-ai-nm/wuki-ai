#pragma once
#ifndef SIMD_HPP
#define SIMD_HPP

/*
 * SIMD 内核总入口
 *
 *   simd/def.hpp       基础类型：V / width / loadu / storeu / splat / vabs / hsum
 *   simd/runSimd.hpp   Simd::Runtime：通用内核（dot / axpy / sum_abs / ...）
 *   simd/runStudy.hpp  Simd::StudyTime：对比学习算法专用
 *                      （激活 / 导数 / 目标回推 / 权重更新）
 */
#include "simd/runSimd.hpp"
#include "simd/runStudy.hpp"

#endif  // SIMD_HPP
