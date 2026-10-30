#pragma once
#ifndef WORKDIR_HPP
#define WORKDIR_HPP

#include <filesystem>
#include <vector>
#include <string>

#if defined(_WIN32)
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <climits>
#else
#  include <unistd.h>
#  include <climits>
#endif

namespace fs = std::filesystem;

fs::path getExecutableDir() {
#if defined(_WIN32)
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        DWORD n = ::GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
        if (n == 0) return {};
        if (n < buf.size()) { buf.resize(n); break; }
        buf.resize(buf.size() * 2);          // 路径被截断，扩容重试
    }
    return fs::path(buf).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buf(size + 1, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    // 该 API 返回的可能含 symlink / 相对段，用 canonical 解析
    std::error_code ec;
    fs::path p = fs::canonical(buf.data(), ec);
    return ec ? fs::path(buf.data()).parent_path() : p.parent_path();
#else
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return fs::path(buf).parent_path();      // /proc/self/exe 已解析 symlink
#endif
}

bool chdirToExecutableDir() {
    fs::path dir = getExecutableDir();
    if (dir.empty()) return false;
    std::error_code ec;
    fs::current_path(dir, ec);
    return !ec;
}

#endif // WORKDIR_HPP