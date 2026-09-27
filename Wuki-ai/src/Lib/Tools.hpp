#pragma once
#ifndef TOOLS_HPP
#define TOOLS_HPP

/*
 * WukiLib 动态库的公开接口
 *
 *   实现：src/Lib/Tools.cpp  →  Bin/libWukiLib.{dll,dylib,so}
 *   内容：配置文件解析（MyConfig::MC）+ 工具函数（MyTool.hpp）
 *
 * MyConfig.hpp、MyTool.hpp 是实现细节，外部代码只需要 #include 本头文件并链接 WukiLib。
 */

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <vector>   // std::vector

#include "WukiExport.hpp"  // WUKI_LIB_API

namespace wuki {

    /**
     * 配置文件解析（对应内部的 MyConfig::MC）
     *
     * 配置格式：
     *   var  <键> <值...>
     *   list <键> <值1> <值2> ...
     *   # 注释
     */
    class WUKI_LIB_API Config {
      public:
        Config();
        ~Config();

        Config(const Config&) = delete;
        Config& operator=(const Config&) = delete;

        /// 解析配置文件；文件打不开返回 false
        bool parse(const std::string& fileName);

        /// 取 list 项，例如 list("neur") → {"2", "3", "2", "1"}
        std::vector<std::string> list(const std::string& key) const;

        /// 取 var 项，键不存在时返回空串
        std::string var(const std::string& key) const;

        /// 键是否存在
        bool hasList(const std::string& key) const;
        bool hasVar(const std::string& key) const;

      private:
        struct Impl;  // 内部实现（只存在于动态库里）
        Impl* impl_;
    };

    /**
     * 字符串列表 → 数值列表（对应内部的 vs_to_vs，非法元素直接忽略）
     */
    WUKI_LIB_API std::vector<std::size_t> vs_to_vs(const std::vector<std::string>& input);

    /**
     * 当前可执行文件所在目录（跨平台：Windows / macOS / Linux；取不到时返回空串）
     *
     * 用来定位和 exe 放在一起的资源文件（例如 Bin/config.conf），
     * 这样程序无论从哪个工作目录启动都能找到，不依赖相对路径。
     */
    WUKI_LIB_API std::string exeDir();

}  // namespace wuki

#endif  // TOOLS_HPP
