#include <iostream>
#include <vector>

#include "Tools.hpp"        // WukiLib：配置解析 + 工具函数
#include "includeMain.hpp"  // WukiInclude：nn / rnn 神经网络

using wcf = wuki::Config;
using wuki::vs_to_vs;

int main() {
    wuki::RNN net;
    net.set_neur({ 1 , 2 , 1 }, 3);

    std::vector<std::vector<float>> targets = { { 4.0f }, { 3.0f }, { 2.0f }, { 1.0f } };
    std::vector<std::vector<float>> error;

    for (size_t t = 0; t < 20; t++) {
        net.CleanContext();

        for (size_t i = 0; i < targets.size(); i++) { net.Run(targets[i]); }

        error.push_back(net.Study(targets, 0.5f, 0.4f));
    }

    std::cout << "训练结束了" << std::endl;
    for (auto& t : error) {
        for (auto& tmp : t) {
            std::cout << tmp << std::endl;
        }

        std::cout << "一个 Step 结束了" << std::endl;
    }

    return 0;
}
