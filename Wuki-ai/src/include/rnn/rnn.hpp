#pragma once
#ifndef RNN_HPP
#define RNN_HPP

#include <vector>  // std::vector
#include <string>  // std::string
#include <random>  // std::mt19937, std::random_device, std::uniform_real_distribution<float>

namespace rnn {
    // 工具函数
    namespace InteUtiFun {  // 内部工具函数

        inline float Random(std::mt19937& gen) {
            return (gen() >> 8) * (1.0f / 16777216.0f);
        }

        inline float RandomRange(std::mt19937& gen, float lo, float hi) {
            return lo + Random(gen) * (hi - lo);
        }

        /**
 * 计算输入向量中所有元素的和
 * @param input 包含size_t类型元素的向量
 * @return 返回向量中所有元素的总和
 */
        inline size_t SumUp(std::vector<size_t>& input) {
            size_t output = 0;  // 用于存储总和的变量，初始化为0
                                // 使用范围for循环遍历输入向量中的每个元素
            for (size_t& ccv : input) {
                output += ccv;  // 将当前元素的值累加到总和中
            }
            return output;  // 返回计算得到的总和
        }

        inline size_t SumOfProducts(std::vector<size_t>& input) {
            size_t output = 0;
            for (size_t layerNumber = 1; layerNumber < input.size(); layerNumber++) {
                output += input[layerNumber - 1] * input[layerNumber];
            }
            return output;
        }
        inline void ThereAreLessThan(std::vector<size_t>& dL, size_t& dC) {
            // 遍历向量中的每个元素
            for (auto& cNOD : dL) {
                // 检测是否超过 dC
                if (cNOD > dC) dC = cNOD;  // 更改 dC
            }

            return;  // 没发现不动
        }
    }  // namespace InteUtiFun
}  // namespace rnn

#endif  // RNN_HPP
