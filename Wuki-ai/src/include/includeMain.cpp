#ifndef INCLUDEMAIN_CPP
#define INCLUDEMAIN_CPP

/*
 * WukiInclude 动态库的唯一编译单元
 *
 * src/include/ 下的 nn、rnn 都是 header-only 实现（类直接定义在 .cpp/.hpp 里），
 * 这里把整个目录收拢成一个编译单元 → Bin/libWukiInclude.{dll,dylib,so}，
 * 对外只暴露 includeMain.hpp 中的 wuki::NN / wuki::RNN 接口。
 */

#include "nn/nn.cpp"
#include "rnn/rnn.cpp"

#include "includeMain.hpp"

namespace wuki {

    // ========================================================================
    //  NN：包装 nn::NeuNet
    // ========================================================================
    struct NN::Impl {
        nn::NeuNet net;
    };

    NN::NN() : impl_(new Impl()) {
    }

    NN::~NN() {
        delete impl_;
        impl_ = nullptr;
    }

    void NN::set_neur(const std::vector<std::size_t>& neur, bool autoInit) {
        impl_->net.set_neur(neur, autoInit);
    }

    std::vector<float> NN::Run(const std::vector<float>& input) {
        return impl_->net.Run(input);
    }

    float NN::Study(const std::vector<float>& goal, float wlr, float blr) {
        return impl_->net.Study(goal, wlr, blr);
    }

    void NN::set_ActFun(const std::string& content) {
        impl_->net.set_ActFun(content);
    }

    bool NN::SaveModel(const std::string& fileName) {
        return impl_->net.SaveModel(fileName);
    }

    bool NN::LoadModel(const std::string& fileName) {
        return impl_->net.LoadModel(fileName);
    }

    std::size_t NN::errorCode() const {
        return impl_->net.par.errorCode;
    }

    std::string NN::errorInfo() const {
        return impl_->net.par.errorBlgig;
    }

    // ========================================================================
    //  RNN：包装 rnn::RNN
    // ========================================================================
    struct RNN::Impl {
        rnn::RNN net;
    };

    RNN::RNN() : impl_(new Impl()) {
    }

    RNN::~RNN() {
        delete impl_;
        impl_ = nullptr;
    }

    void RNN::set_neur(const std::vector<std::size_t>& neur, bool autoInit) {
        impl_->net.set_neur(neur, autoInit);
    }

    std::vector<float> RNN::Run(const std::vector<float>& input) {
        return impl_->net.Run(input);
    }

    std::vector<float> RNN::Study(const std::vector<std::vector<float>>& goal, float wlr, float blr) {
        return impl_->net.Study(goal, wlr, blr);
    }

    void RNN::set_ActFun(const std::string& content) {
        impl_->net.set_ActFun(content);
    }

    bool RNN::SaveModel(const std::string& fileName) {
        return impl_->net.SaveModel(fileName);
    }

    bool RNN::LoadModel(const std::string& fileName) {
        return impl_->net.LoadModel(fileName);
    }

    void RNN::CleanContext() {
        impl_->net.CleanContext();
    }

    std::size_t RNN::errorCode() const {
        return impl_->net.par.errorCode;
    }

    std::string RNN::errorInfo() const {
        return impl_->net.par.errorBlgig;
    }

}  // namespace wuki

#endif  // INCLUDEMAIN_CPP
