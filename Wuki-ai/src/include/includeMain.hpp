#pragma once
#ifndef INCLUDEMAIN_HPP
#define INCLUDEMAIN_HPP

/*
 * WukiInclude 动态库的公开接口
 *
 *   实现：src/include/includeMain.cpp  →  Bin/libWukiInclude.{dll,dylib,so}
 *   内容：nn（NeuNet）+ rnn（RNN）
 *
 * nn/nn.cpp、rnn/rnn.cpp 是 header-only 的实现细节，外部代码只需要 #include 本头文件、
 * 并链接 WukiInclude，不必（也不应该）直接 #include 它们的 .cpp。
 */

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <vector>   // std::vector

#include "WukiExport.hpp"  // WUKI_INCLUDE_API

namespace wuki {

    /**
     * 全连接神经网络（对应内部的 nn::NeuNet）
     */
    class WUKI_INCLUDE_API NN {
      public:
        NN();
        ~NN();

        // 句柄型对象，禁止拷贝（避免两个对象指向同一份内部实现）
        NN(const NN&) = delete;
        NN& operator=(const NN&) = delete;

        /// 设置每层神经元数量，autoInit = true 时立即随机初始化 w / b
        void set_neur(const std::vector<std::size_t>& neur, bool autoInit = true);

        /// 前向传播
        std::vector<float> Run(const std::vector<float>& input);

        /// 对比学习，返回 MSE 误差
        float Study(const std::vector<float>& goal, float wlr = 0.01f, float blr = 0.001f);

        /// 设置激活函数："relu" / "silu" / "alt-sigmoid"
        void set_ActFun(const std::string& content);

        /// 保存模型到文件（JSON）；失败返回 false，原因见 errorCode() / errorInfo()
        bool SaveModel(const std::string& fileName);

        /// 从文件读取模型（会校验文件签名和尺寸）；失败返回 false，原因见 errorCode() / errorInfo()
        bool LoadModel(const std::string& fileName);

        /// 最近一次的错误码 / 错误归属（无错误时为 0x00 / "not error"）
        std::size_t errorCode() const;
        std::string errorInfo() const;

      private:
        struct Impl;  // 内部实现（只存在于动态库里）
        Impl* impl_;
    };

    /**
     * 带上下文累积的循环神经网络（对应内部的 rnn::RNN）
     */
    class WUKI_INCLUDE_API RNN {
      public:
        RNN();
        ~RNN();

        RNN(const RNN&) = delete;
        RNN& operator=(const RNN&) = delete;

        /// 设置每层神经元数量，autoInit = true 时立即随机初始化 w / b
        void set_neur(const std::vector<std::size_t>& neur, bool autoInit = true);

        /// 前向传播（autoAttenuation = true 时自动衰减上下文）
        std::vector<float> Run(const std::vector<float>& input);

        /// 对比学习，返回 MSE 误差
        std::vector<float> Study(const std::vector<std::vector<float>>& goal, float wlr = 0.01f, float blr = 0.001f);

        /// 设置激活函数："relu" / "silu" / "alt-sigmoid"
        void set_ActFun(const std::string& content);

        /// 保存模型到文件（JSON）；失败返回 false，原因见 errorCode() / errorInfo()
        bool SaveModel(const std::string& fileName);

        /// 从文件读取模型（会校验文件签名和尺寸）；失败返回 false，原因见 errorCode() / errorInfo()
        bool LoadModel(const std::string& fileName);

        /// 清空上下文（瞬时参数 a / z）
        void CleanContext();

        /// 最近一次的错误码 / 错误归属（无错误时为 0x00 / "not error"）
        std::size_t errorCode() const;
        std::string errorInfo() const;

      private:
        struct Impl;  // 内部实现（只存在于动态库里）
        Impl* impl_;
    };

}  // namespace wuki

#endif  // INCLUDEMAIN_HPP
