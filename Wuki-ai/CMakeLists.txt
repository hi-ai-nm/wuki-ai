# ============================================================================
#  Wuki-ai 构建脚本
#
#  产物（全部输出到 Bin/）：
#    Bin/libWukiInclude.dylib  | Bin/WukiInclude.dll  | Bin/libWukiInclude.so
#        来源：src/include/（nn + rnn），编译单元 src/include/includeMain.cpp
#    Bin/libWukiLib.dylib      | Bin/WukiLib.dll      | Bin/libWukiLib.so
#        来源：src/Lib（配置 + 工具），编译单元 src/Lib/Tools.cpp
#    Bin/libWukiComm.dylib     | Bin/WukiComm.dll     | Bin/libWukiComm.so
#        来源：src/comm（和 Flask 通信的 HTTP 服务），编译单元 src/comm/fskio.cpp
#    Bin/wuki-ai               | Bin/wuki-ai.exe
#        来源：src/main/main.cpp，链接上面三个动态库
#
#  同一个 CMakeLists 会自动适配：Windows→dll、macOS→dylib、Linux→so
# ============================================================================

# 设置最低版本（原 4.4.2，因 Docker 系统原因，降成 3.22）
cmake_minimum_required(VERSION 3.22)

# 声明工程的名称
project(Wuki LANGUAGES CXX)


# ---------------------------------------------------------------------------
# 全局设置
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ON：动态库（dll / dylib / so）；OFF：静态库（exe 会把库静态链进去）
option(BUILD_SHARED_LIBS "构建动态库（Windows: dll / macOS: dylib / Linux: so）" ON)

if(BUILD_SHARED_LIBS)
    set(WUKI_LIBRARY_TYPE SHARED)
else()
    set(WUKI_LIBRARY_TYPE STATIC)
endif()

# 所有产物统一丢到 Bin/
set(WUKI_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${WUKI_BIN_DIR}")  # exe、Windows 的 dll
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${WUKI_BIN_DIR}")  # so、dylib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${WUKI_BIN_DIR}")  # 静态库、Windows 导入库 .lib

# 多配置生成器（Visual Studio / Xcode）会额外套一层 Debug/、Release/，这里一并压平
foreach(_wuki_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_wuki_cfg} "${WUKI_BIN_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_wuki_cfg} "${WUKI_BIN_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_wuki_cfg} "${WUKI_BIN_DIR}")
endforeach()
unset(_wuki_cfg)

# 让 Bin/ 里的 exe 直接从自己所在的目录找动态库（Linux: $ORIGIN，macOS: @loader_path）
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(_wuki_origin "@loader_path")
elseif(UNIX)
    set(_wuki_origin "$ORIGIN")
endif()

if(DEFINED _wuki_origin)
    set(CMAKE_INSTALL_RPATH "${_wuki_origin}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    unset(_wuki_origin)
endif()

# ---------------------------------------------------------------------------
# 1) WukiInclude：src/include/ 以 includeMain.cpp 为中心 → 动态库
#    nn、rnn 是 header-only 实现，全部由这一个编译单元收拢进库
# ---------------------------------------------------------------------------
add_library(WukiInclude ${WUKI_LIBRARY_TYPE} src/include/includeMain.cpp)

target_compile_options(WukiInclude PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -flto>
)

target_include_directories(WukiInclude
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/include")

# 库内编译时导出符号（dllexport），外部使用时自动切成导入（dllimport）
target_compile_definitions(WukiInclude PRIVATE WUKI_INCLUDE_BUILD)

set_target_properties(WukiInclude PROPERTIES
    OUTPUT_NAME "WukiInclude"
    POSITION_INDEPENDENT_CODE ON
    # 只导出 WUKI_INCLUDE_API 标记的接口，内部 header-only 符号留在库里
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# 2) WukiLib：src/Lib 以 Tools.cpp 为中心 → 动态库
# ---------------------------------------------------------------------------
add_library(WukiLib ${WUKI_LIBRARY_TYPE} src/Lib/Tools.cpp)

target_compile_options(WukiLib PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -flto>
)

target_include_directories(WukiLib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/Lib"
    # 只为了找到 WukiExport.hpp（导出宏）
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/include")

target_compile_definitions(WukiLib PRIVATE WUKI_LIB_BUILD)

set_target_properties(WukiLib PROPERTIES
    OUTPUT_NAME "WukiLib"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# 3) WukiComm：src/comm 以 fskio.cpp 为中心 → 动态库
#    cpp-httplib 是 header-only（src/external/httplib.h），只在这一个编译单元里展开，
#    所以 httplib 的符号不会漏到 exe 里；exe 看到的只是 FSKIO_HPP 里的接口
# ---------------------------------------------------------------------------
add_library(WukiComm ${WUKI_LIBRARY_TYPE} src/comm/fskio.cpp)

target_compile_options(WukiComm PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -flto>
)

target_include_directories(WukiComm
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/comm"
    # 只为了找到 WukiExport.hpp（导出宏）
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/include")

target_compile_definitions(WukiComm PRIVATE WUKI_COMM_BUILD)

# httplib 的监听线程要线程库（Linux 上是 pthread；macOS 上是 libSystem 自带的，
# Threads::Threads 在那里是空的，写进来不影响）
find_package(Threads REQUIRED)
target_link_libraries(WukiComm PUBLIC Threads::Threads)

if(WIN32)
    # Windows 上 socket 在 ws2_32 里
    target_link_libraries(WukiComm PUBLIC ws2_32)
endif()

set_target_properties(WukiComm PROPERTIES
    OUTPUT_NAME "WukiComm"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# 4) wuki-ai：src/main/main.cpp → exe / exec（链接上面三个库）
# ---------------------------------------------------------------------------
add_executable(wuki-ai src/main/main.cpp)

target_compile_options(wuki-ai PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -flto>
)

target_link_libraries(wuki-ai PRIVATE WukiInclude WukiLib WukiComm)

set_target_properties(wuki-ai PROPERTIES OUTPUT_NAME "wuki-ai")

# Windows：exe 无法链接 dll（只能链接导入库），这里提醒一下链接关系
if(WIN32)
    message(STATUS "Windows 下运行前请确保 Bin 里的 WukiInclude.dll、WukiLib.dll、WukiComm.dll 和 exe 在同一目录")
endif()

# ---------------------------------------------------------------------------
# 安装：cmake --install . 也会把这一套放进 <prefix>/Bin
# ---------------------------------------------------------------------------
install(TARGETS WukiInclude WukiLib WukiComm wuki-ai
    RUNTIME DESTINATION Bin
    LIBRARY DESTINATION Bin
    ARCHIVE DESTINATION Bin)

message(STATUS "构建产物输出目录: ${WUKI_BIN_DIR}")
