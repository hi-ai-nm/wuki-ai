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

        /*
         * 句柄可以移动，但仍然不能复制（两个对象不能指向同一份内部实现）
         *
         * 为什么要能移动：双层架构（Arch::DRNN）的「读模型」要先把两个子网读进临时对象，
         * 全部校验通过之后才换进目标模型 —— 一次性换进去才是「失败不留半截状态」。
         * 加载完之后，移动出来的源对象不是 nullptr，而是一个「空的合法对象」：
         * 再调用它只会得到「未初始化」的错误，不会崩。
         *
         * 这里故意不写 noexcept：移动要给源对象新建一个空实现，new 有可能抛 bad_alloc，
         * 标了 noexcept 就会直接 terminate，反而不安全。
         */
        RNN(RNN&& other);
        RNN& operator=(RNN&& other);

        /**
         * 设置每层神经元数量，autoInit = true 时立即随机初始化 w / b
         *
         * hSize 是隐状态大小：输入层会被撑成「input ‖ h」、输出层撑成「goal ‖ h」，
         * 所以 par.neur 里存的是「加过 hSize 之后」的尺寸。
         * 注意 hSize 一定要显式传 —— 它和 autoInit 挨着，漏传会被 bool 悄悄顶上。
         */
        void set_neur(const std::vector<std::size_t>& neur, size_t hSize = 16, bool autoInit = true);

        /// 前向传播（autoAttenuation = true 时自动衰减上下文）
        std::vector<float> Run(const std::vector<float>& input);

        /**
         * 整段序列的对比学习（自创，时间展开）：goals[t] / inputs[t] 是第 t 步的目标与输入
         *
         * 内部先把整段 Run 完（每步的 a / z / x 都缓存），再从最后一步倒着推目标：
         * 第 t 步喂进来的是 h(t-1)、产出的是 h(t)，所以第 t+1 步反推到输入层 h 段上的
         * 那个目标，正好就是第 t 步该产出的 h 的目标 —— 倒着走能立刻用上。
         * 这一步跨时间的回填是 RNN 和 NN 的根本区别，单步调用给不了，所以接口是整段。
         *
         * 每次调用都从 h = 0 开始跑一条新序列（内部会 CleanContext）。
         *
         * @return 每一步的 MSE 误差（长度 = 步数）；出错返回空 vector，
         *         正数 MSE 和错误分得开，原因见 errorCode() / errorInfo()
         */
        std::vector<float> Train(const std::vector<std::vector<float>>& goals,
                                 const std::vector<std::vector<float>>& inputs,
                                 const float wlr = 0.01f, const float blr = 0.01f);

        /// 设置激活函数："relu" / "silu" / "alt-sigmoid"
        void set_ActFun(const std::string& content);

        /// 保存模型到文件（JSON）；失败返回 false，原因见 errorCode() / errorInfo()
        bool SaveModel(const std::string& fileName);

        /// 从文件读取模型（会校验文件签名和尺寸）；失败返回 false，原因见 errorCode() / errorInfo()
        bool LoadModel(const std::string& fileName);

        /// 每层神经元数量（含输入层 / 输出层尾部的 h 段；未初始化时为空）
        std::vector<std::size_t> neurons() const;

        /// 隐状态大小（hSize）
        std::size_t hSize() const;

        /**
         * 把模型序列化成 JSON 字符串（格式与 SaveModel 写出的文件完全相同）
         *
         * 给「一个文件装多个子网」的架构用（见 src/main/llm/DRNNSL.hpp）：
         * 库外拿不到 w / b，只能靠它把子网内容交出去。
         *
         * @return 成功返回 JSON 文本；失败返回空串，原因见 errorCode() / errorInfo()
         */
        std::string ModelToJson();

        /// 从 JSON 字符串恢复模型（校验和 LoadModel 完全一样）；失败返回 false
        bool ModelFromJson(const std::string& content);

        /// 清空上下文：隐状态 h 归零、瞬时参数 a / z / x 清空（换一条新序列时用）
        void CleanContext();

        std::vector<float> obtainInputTarget();

        /**
         * 上一次 Train 里「每一步」反推回输入层的目标（obtainInputTarget 的整段版本）
         *
         * 长度 = 步数，每格宽度 = 「输入层宽度 - hSize」；没 Train 过 / Train 失败 /
         * CleanContext 之后是空列表。
         *
         * 给「一个网络喂另一个网络」的架构用（Arch::DRNN 的 Train）：下游网络整段的
         * 输入目标就是上游网络该产出的那一整块 —— 一块切成好几片时，只拿第 0 步那份
         * 是拼不满宽度的。
         */
        std::vector<std::vector<float>> obtainInputTargets();

        /// 最近一次的错误码 / 错误归属（无错误时为 0x00 / "not error"）
        std::size_t errorCode() const;
        std::string errorInfo() const;

      private:
        struct Impl;  // 内部实现（只存在于动态库里）
        Impl* impl_;
    };

}  // namespace wuki

#endif  // INCLUDEMAIN_HPP
