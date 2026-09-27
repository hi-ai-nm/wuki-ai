#pragma once
#ifndef WUKI_EXPORT_HPP
#define WUKI_EXPORT_HPP

/*
 * 跨平台动态库导出 / 导入宏
 *
 * 用法：
 *   库的实现（CMake 里带 WUKI_INCLUDE_BUILD / WUKI_LIB_BUILD 宏编译）看到的是
 *   dllexport（Windows）或 visibility("default")（GCC / Clang），符号被导出；
 *   外部使用者（exe、其它库）看到的是 dllimport，直接链接导入库即可。
 *
 *   标记任意一个类 / 函数：
 *       class WUKI_INCLUDE_API Foo { ... };
 *       WUKI_LIB_API int bar();
 */

/* ---------- 平台底层宏 ---------- */
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
// Windows（MSVC / MinGW）：用 __declspec 控制导出
#define WUKI_DETAIL_EXPORT __declspec(dllexport)
#define WUKI_DETAIL_IMPORT __declspec(dllimport)
#define WUKI_DETAIL_LOCAL
#elif defined(__GNUC__) || defined(__clang__)
// GCC / Clang（Linux so、macOS dylib）：用 visibility 属性控制导出
#define WUKI_DETAIL_EXPORT __attribute__((visibility("default")))
#define WUKI_DETAIL_IMPORT __attribute__((visibility("default")))
#define WUKI_DETAIL_LOCAL __attribute__((visibility("hidden")))
#else
// 未知编译器：退化成空宏（等价于全部符号都可见）
#define WUKI_DETAIL_EXPORT
#define WUKI_DETAIL_IMPORT
#define WUKI_DETAIL_LOCAL
#endif

/* ---------- 静态库模式：不需要导出 / 导入 ---------- */
#if defined(WUKI_STATIC)
#define WUKI_INCLUDE_API
#define WUKI_LIB_API

/* ---------- 动态库模式 ---------- */
#else
#if defined(WUKI_INCLUDE_BUILD)
#define WUKI_INCLUDE_API WUKI_DETAIL_EXPORT
#else
#define WUKI_INCLUDE_API WUKI_DETAIL_IMPORT
#endif

#if defined(WUKI_LIB_BUILD)
#define WUKI_LIB_API WUKI_DETAIL_EXPORT
#else
#define WUKI_LIB_API WUKI_DETAIL_IMPORT
#endif
#endif

#endif  // WUKI_EXPORT_HPP
