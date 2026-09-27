#ifndef TOOLS_CPP
#define TOOLS_CPP

/*
 * WukiLib 动态库的唯一编译单元
 *
 * 把 MyConfig.hpp（配置解析）和 MyTool.hpp（工具函数）收拢成一个编译单元
 * → Bin/libWukiLib.{dll,dylib,so}，对外只暴露 Tools.hpp 中的 wuki:: 接口。
 */

#include "MyConfig.hpp"
#include "MyTool.hpp"

#include "Tools.hpp"

#include <cstdlib>  // realpath
#include <fstream>  // std::ifstream

// 取可执行文件路径的平台接口
#if defined(_WIN32)
#include <windows.h>  // GetModuleFileNameA
#elif defined(__APPLE__)
#include <mach-o/dyld.h>  // _NSGetExecutablePath
#elif defined(__linux__)
#include <unistd.h>  // readlink
#endif

namespace wuki {

    // ========================================================================
    //  Config：包装 MyConfig::MC
    // ========================================================================
    struct Config::Impl {
        MyConfig::MC mc;
    };

    Config::Config() : impl_(new Impl()) {}

    Config::~Config() {
        delete impl_;
        impl_ = nullptr;
    }

    bool Config::parse(const std::string& fileName) {
        // 先自己探一次文件，能打开才交给 MC::parse（它内部失败只会打印提示）
        std::ifstream probe(fileName);
        if (!probe.is_open()) {
            return false;
        }
        probe.close();

        impl_->mc.parse(fileName);
        return true;
    }

    std::vector<std::string> Config::list(const std::string& key) const {
        auto it = impl_->mc.data.listData.find(key);
        if (it == impl_->mc.data.listData.end()) {
            return std::vector<std::string>{};
        }
        return it->second;
    }

    std::string Config::var(const std::string& key) const {
        auto it = impl_->mc.data.varData.find(key);
        if (it == impl_->mc.data.varData.end()) {
            return std::string{};
        }
        return it->second;
    }

    bool Config::hasList(const std::string& key) const {
        return impl_->mc.data.listData.find(key) != impl_->mc.data.listData.end();
    }

    bool Config::hasVar(const std::string& key) const {
        return impl_->mc.data.varData.find(key) != impl_->mc.data.varData.end();
    }

    // ========================================================================
    //  工具函数：复用 MyTool.hpp 里已有的实现
    // ========================================================================
    std::vector<std::size_t> vs_to_vs(const std::vector<std::string>& input) {
        return ::vs_to_vs(input);
    }

    // ========================================================================
    //  工具函数：可执行文件所在目录
    // ========================================================================
    std::string exeDir() {
        std::string path;

#if defined(_WIN32)
        char buf[MAX_PATH] = {0};
        DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
        if (len == 0 || len >= MAX_PATH) {
            return std::string{};
        }
        path.assign(buf, static_cast<std::size_t>(len));
#elif defined(__APPLE__)
        char raw[4096] = {0};
        uint32_t size = sizeof(raw);
        if (_NSGetExecutablePath(raw, &size) != 0) {
            return std::string{};  // 缓冲区不够（此处一般不会）
        }
        char resolved[4096] = {0};
        if (::realpath(raw, resolved) != nullptr) {
            path = resolved;  // 解析掉符号链接 / 相对路径
        } else {
            path = raw;
        }
#elif defined(__linux__)
        char buf[4096] = {0};
        ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len <= 0) {
            return std::string{};
        }
        path.assign(buf, static_cast<std::size_t>(len));
#else
        return std::string{};  // 其它平台：不支持
#endif

        // 去掉最后的文件名，只留目录
        std::size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return std::string{};
        }
        return path.substr(0, pos);
    }

}  // namespace wuki

#endif  // TOOLS_CPP
