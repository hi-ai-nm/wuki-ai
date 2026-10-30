# ============================================================================
#  WebGPU 依赖解析（wgpu-native 预编译包）
#
#  由顶层 CMakeLists include。做三件事：
#    1. 找一个可用的 wgpu-native SDK（顺序：WUKI_WEBGPU_SDK_DIR → third_party/
#       → 按固定版本下载预编译包，带 SHA256 校验）；
#    2. 建一个 IMPORTED 目标 wgpu_native（头文件 + 动态库）；
#    3. 把动态库在构建后拷到 Bin/，exe 用 @loader_path / $ORIGIN 就能找到。
#
#  许可证：wgpu-native 是 Apache-2.0 / MIT 双许可（本项目按 Apache-2.0 使用），
#  随包下载的 webgpu.h 是 BSD-3-Clause。原文见 src/external/ 下三个 LICENSE 文件。
#  下载的是官方 release 资产，没有做任何修改（Apache-2.0 §4(b) 的「modified files」不适用）。
# ============================================================================

set(WUKI_WEBGPU "AUTO" CACHE STRING "WebGPU 加速后端：AUTO（找到 SDK 就编）/ ON（必须可用）/ OFF")
set_property(CACHE WUKI_WEBGPU PROPERTY STRINGS AUTO ON OFF)

set(WUKI_WEBGPU_VERSION "v29.0.1.1" CACHE STRING "wgpu-native 版本（升级时同步改下面的 SHA256）")
set(WUKI_WEBGPU_SDK_DIR "" CACHE PATH "已有的 wgpu-native SDK 目录（含 include/webgpu 和 lib/）；空 = 自动找 / 下载")
option(WUKI_WEBGPU_DOWNLOAD "本地找不到 SDK 时自动下载 wgpu-native 预编译包" ON)

string(TOUPPER "${WUKI_WEBGPU}" _wuki_webgpu_mode)
set(WUKI_WEBGPU_AVAILABLE FALSE)

# 这个函数必须先定义：下面任何一条「降级 / 关闭」分支都会提前 return，
# 而顶层 CMakeLists 无条件调用它（不可用时它自己什么都不做）。

# 让某个目标链接 WebGPU，并在构建后把动态库拷到产物目录
function(wuki_webgpu_attach target)
    if(NOT WUKI_WEBGPU_AVAILABLE)
        return()
    endif()
    target_link_libraries(${target} PUBLIC wgpu_native)
    target_compile_definitions(${target} PRIVATE WUKI_HAS_WEBGPU=1)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${WUKI_WEBGPU_RUNTIME_LIB}" "$<TARGET_FILE_DIR:${target}>"
        COMMENT "拷贝 wgpu-native 动态库到产物目录")
endfunction()

if(_wuki_webgpu_mode STREQUAL "OFF")
    message(STATUS "WebGPU: 已关闭（WUKI_WEBGPU=OFF），优化层只用 SIMD")
    return()
endif()

# ---------------------------------------------------------------------------
# 平台 → 官方资产名 / SHA256 / 动态库文件名
#
# 只支持官方 release 里有的桌面组合；其它平台（比如 32 位 Linux）自动降级成 SIMD，
# 不会让构建失败。
# ---------------------------------------------------------------------------
set(_wuki_wgpu_asset "")
set(_wuki_wgpu_sha "")
set(_wuki_wgpu_lib "")
set(_wuki_wgpu_implib "")
set(_wuki_wgpu_libdir "lib")

if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(_wuki_wgpu_asset "wgpu-macos-aarch64-release.zip")
        set(_wuki_wgpu_sha "a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(_wuki_wgpu_asset "wgpu-macos-x86_64-release.zip")
        set(_wuki_wgpu_sha "8e2f7378548ddd0e2cf21e7d864dda46e953f0af724855a33778b85ead206d41")
    endif()
    set(_wuki_wgpu_lib "libwgpu_native.dylib")
elseif(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_wuki_wgpu_asset "wgpu-windows-x86_64-msvc-release.zip")
        set(_wuki_wgpu_sha "7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132")
        set(_wuki_wgpu_lib "wgpu_native.dll")
        set(_wuki_wgpu_implib "wgpu_native.lib")
    endif()
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_wuki_wgpu_asset "wgpu-linux-aarch64-release.zip")
        set(_wuki_wgpu_sha "015fcdf1dbae82e614a783cc38017e5399ae0927a889fe9b69c9b664bc61b47a")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(_wuki_wgpu_asset "wgpu-linux-x86_64-release.zip")
        set(_wuki_wgpu_sha "95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a")
    endif()
    set(_wuki_wgpu_lib "libwgpu_native.so")
endif()

if(NOT _wuki_wgpu_asset)
    if(_wuki_webgpu_mode STREQUAL "ON")
        message(FATAL_ERROR "WUKI_WEBGPU=ON，但当前平台（${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}）没有官方 wgpu-native 资产")
    endif()
    message(STATUS "WebGPU: 当前平台没有预编译包，自动降级为纯 SIMD")
    return()
endif()

# 允许用本地 SDK 覆盖（离线构建 / 自己编的 wgpu-native）
set(_wuki_wgpu_sdk "${WUKI_WEBGPU_SDK_DIR}")
if(NOT _wuki_wgpu_sdk AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/wgpu-native/include/webgpu/webgpu.h")
    set(_wuki_wgpu_sdk "${CMAKE_CURRENT_SOURCE_DIR}/third_party/wgpu-native")
    message(STATUS "WebGPU: 使用仓库内的 third_party/wgpu-native")
endif()

# 没有本地 SDK 就下载固定版本的预编译包（缓存在 build/_deps 里，换构建目录才会重下）
if(NOT _wuki_wgpu_sdk)
    if(NOT WUKI_WEBGPU_DOWNLOAD)
        if(_wuki_webgpu_mode STREQUAL "ON")
            message(FATAL_ERROR "WUKI_WEBGPU=ON 但没找到 SDK，且 WUKI_WEBGPU_DOWNLOAD=OFF")
        endif()
        message(STATUS "WebGPU: 没找到 SDK 且禁止下载，自动降级为纯 SIMD")
        return()
    endif()

    set(_wuki_wgpu_cache "${CMAKE_BINARY_DIR}/_deps/wgpu-native-${WUKI_WEBGPU_VERSION}")
    set(_wuki_wgpu_stamp "${_wuki_wgpu_cache}/.extracted")
    if(NOT EXISTS "${_wuki_wgpu_stamp}")
        set(_wuki_wgpu_url "https://github.com/gfx-rs/wgpu-native/releases/download/${WUKI_WEBGPU_VERSION}/${_wuki_wgpu_asset}")
        set(_wuki_wgpu_zip "${_wuki_wgpu_cache}.zip")
        message(STATUS "WebGPU: 下载 ${_wuki_wgpu_url}")
        # 注意：这里绝对不能把 EXPECTED_HASH 直接挂在 file(DOWNLOAD) 上。
        #   下载失败（容器里连不上 GitHub 就是这种情况）时，带 EXPECTED_HASH 的
        #   file(DOWNLOAD) 会**直接抛 CMake Error**（"cannot compute hash on failed
        #   download"），下面基于 STATUS 的降级分支根本走不到，AUTO 就变成了配置失败
        #   —— docker build 挂在这一步就是这个原因。
        #   改成：先下到文件 → 看 STATUS → 再自己算 SHA256 比对，任何一步失败都能降级。
        file(DOWNLOAD "${_wuki_wgpu_url}" "${_wuki_wgpu_zip}"
             TLS_VERIFY ON
             TIMEOUT 300
             SHOW_PROGRESS
             STATUS _wuki_wgpu_status)
        list(GET _wuki_wgpu_status 0 _wuki_wgpu_code)
        list(GET _wuki_wgpu_status 1 _wuki_wgpu_reason)
        if(NOT _wuki_wgpu_code EQUAL 0)
            file(REMOVE "${_wuki_wgpu_zip}")
            if(_wuki_webgpu_mode STREQUAL "ON")
                message(FATAL_ERROR "WebGPU: 下载 wgpu-native 失败（${_wuki_wgpu_reason}）")
            endif()
            message(STATUS "WebGPU: 下载失败（${_wuki_wgpu_reason}），自动降级为纯 SIMD")
            return()
        endif()

        file(SHA256 "${_wuki_wgpu_zip}" _wuki_wgpu_hash)
        if(NOT _wuki_wgpu_hash STREQUAL _wuki_wgpu_sha)
            file(REMOVE "${_wuki_wgpu_zip}")
            if(_wuki_webgpu_mode STREQUAL "ON")
                message(FATAL_ERROR "WebGPU: 下载的 ${_wuki_wgpu_asset} SHA256 不匹配（期望 ${_wuki_wgpu_sha}，实际 ${_wuki_wgpu_hash}）")
            endif()
            message(STATUS "WebGPU: 下载校验和不匹配（可能是代理 / 镜像返回了别的内容），自动降级为纯 SIMD")
            return()
        endif()

        file(ARCHIVE_EXTRACT INPUT "${_wuki_wgpu_zip}" DESTINATION "${_wuki_wgpu_cache}")
        file(WRITE "${_wuki_wgpu_stamp}" "wgpu-native ${WUKI_WEBGPU_VERSION} ${_wuki_wgpu_asset}\n")
    endif()
    set(_wuki_wgpu_sdk "${_wuki_wgpu_cache}")
endif()

# 校验 SDK 结构：两个头 + 一个动态库，缺一个就当没有
if(NOT EXISTS "${_wuki_wgpu_sdk}/include/webgpu/webgpu.h" OR
   NOT EXISTS "${_wuki_wgpu_sdk}/include/webgpu/wgpu.h" OR
   NOT EXISTS "${_wuki_wgpu_sdk}/${_wuki_wgpu_libdir}/${_wuki_wgpu_lib}")
    if(_wuki_webgpu_mode STREQUAL "ON")
        message(FATAL_ERROR "WebGPU: SDK 目录结构不完整：${_wuki_wgpu_sdk}")
    endif()
    message(STATUS "WebGPU: SDK 目录不完整（${_wuki_wgpu_sdk}），自动降级为纯 SIMD")
    return()
endif()

add_library(wgpu_native SHARED IMPORTED GLOBAL)
set_target_properties(wgpu_native PROPERTIES
    IMPORTED_LOCATION "${_wuki_wgpu_sdk}/${_wuki_wgpu_libdir}/${_wuki_wgpu_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wuki_wgpu_sdk}/include")
if(WIN32 AND _wuki_wgpu_implib)
    set_target_properties(wgpu_native PROPERTIES
        IMPORTED_IMPLIB "${_wuki_wgpu_sdk}/${_wuki_wgpu_libdir}/${_wuki_wgpu_implib}")
endif()

# 给顶层 CMakeLists 用的几个值（函数里看不到 include 作用域的局部变量）
set(WUKI_WEBGPU_AVAILABLE TRUE)
set(WUKI_WEBGPU_SDK_RESOLVED "${_wuki_wgpu_sdk}" CACHE INTERNAL "解析后的 wgpu-native SDK 目录")
set(WUKI_WEBGPU_RUNTIME_LIB "${_wuki_wgpu_sdk}/${_wuki_wgpu_libdir}/${_wuki_wgpu_lib}" CACHE INTERNAL "要拷到 Bin/ 的动态库")

message(STATUS "WebGPU: 已启用（wgpu-native ${WUKI_WEBGPU_VERSION}，SDK: ${_wuki_wgpu_sdk}）")
