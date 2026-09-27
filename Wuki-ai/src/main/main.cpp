#include <iostream>

#include "Tools.hpp"        // WukiLib：配置解析 + 工具函数
#include "includeMain.hpp"  // WukiInclude：nn / rnn 神经网络

using mc = wuki::Config;
using wuki::vs_to_vs;

int main() {
    // 解析配置文件
    mc config;
    if (!config.parse("config.conf")) {
        // 回退：exe 同目录（从别的工作目录启动、或双击运行时也能找到 Bin/config.conf）
        if (!config.parse(wuki::exeDir() + "/config.conf")) {
            std::cerr << "config.conf 打开失败" << std::endl;
        }
    }

    // 创建神经网络
    wuki::RNN net;
    net.set_neur(vs_to_vs(config.list("neur")));  // 设置网络结构

    auto out = net.Run({1.0f, 0.5f});
    std::cout << "输出: ";
    for (auto v : out) std::cout << v << " ";
    std::cout << std::endl;
    return 0;
}
