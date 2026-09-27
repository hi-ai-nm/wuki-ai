#pragma once
#ifndef NN_HPP
#define NN_HPP

#include <vector>  // std::vector
#include <string>  // std::string
#include <random>  // std::mt19937, std::random_device, std::uniform_real_distribution<float>

namespace nn {
    // 工具函数
    namespace InteUtiFun {  // 内部工具函数
        inline float Random(std::mt19937& gen) {
            return (gen() >> 8) * (1.0f / 16777216.0f);
        }

        inline float RandomRange(std::mt19937& gen, float lo, float hi) {
            return lo + Random(gen) * (hi - lo);
        }
        inline size_t SumUp(std::vector<size_t>& input) {
            size_t output = 0;
            for (size_t& ccv : input) {
                output += ccv;
            }
            return output;
        }

        inline size_t SumOfProducts(std::vector<size_t>& input) {
            size_t output = 0;
            for (size_t layerNumber = 1; layerNumber < input.size(); layerNumber++) {
                output += input[layerNumber - 1] * input[layerNumber];
            }
            return output;
        }

        inline void ThereAreLessThan(std::vector<size_t>& dL, size_t& dC) {
            for (auto& cNOD : dL) {
                // 检测是否超过 dC
                if (cNOD > dC) dC = cNOD;  // 更改 dC
            }

            return;  // 没发现不动
        }
    }  // namespace InteUtiFun
}  // namespace nn

#endif  // NN_HPP
