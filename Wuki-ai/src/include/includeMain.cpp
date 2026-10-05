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

    // 移动：把内部实现整个交出去，然后给源对象留一个「新建的空实现」而不是 nullptr
    // （nullptr 的话，移动过的对象一调用方法就崩；空的实现只会报「未初始化」）
    RNN::RNN(RNN&& other) : impl_(other.impl_) {
        other.impl_ = new Impl();
    }

    RNN& RNN::operator=(RNN&& other) {
        if (this == &other) {
            return *this;  // 自移动：直接返回，别先把自己的实现删了
        }
        delete impl_;  // 换掉之前先把旧的还回去（不然就漏了）
        impl_ = other.impl_;
        other.impl_ = new Impl();
        return *this;
    }

    void RNN::set_neur(const std::vector<std::size_t>& neur, size_t hSize, bool autoInit) {
        impl_->net.set_neur(neur, hSize, autoInit);
    }

    std::vector<float> RNN::Run(const std::vector<float>& input) {
        return impl_->net.Run(input);
    }

    std::vector<float> RNN::Train(const std::vector<std::vector<float>>& goals,
                                  const std::vector<std::vector<float>>& inputs,
                                  const float wlr, const float blr) {
        return impl_->net.Train(goals, inputs, wlr, blr);
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

    // par.neur / par.hSize 是 Init 加过 h 之后的结构，双层架构（DRNNSL）靠它们把
    // 「结构参数」反推回来，不用去翻 rnn 的 json 内部字段
    std::vector<std::size_t> RNN::neurons() const {
        return impl_->net.par.neur;
    }

    std::size_t RNN::hSize() const {
        return impl_->net.par.hSize;
    }

    // 内存版存取：和 SaveModel / LoadModel 共用 rnn 的 toJson / fromJson，
    // 所以格式、校验、错误码全都一致，只是不落文件
    std::string RNN::ModelToJson() {
        rnn::json outJson;
        if (!impl_->net.toJson(outJson)) {  // 错误码 / 归属已经写进 par
            return {};
        }
        return outJson.dump(4);
    }

    bool RNN::ModelFromJson(const std::string& content) {
        rnn::json inJson;
        try {
            inJson = rnn::json::parse(content);
        } catch (const rnn::json::exception&) {  // 不是合法 JSON：和 LoadModel 同一套错误码
            impl_->net.par.errorCode = 0x01;
            impl_->net.par.errorBlgig = "load";
            return false;
        }
        return impl_->net.fromJson(inJson);
    }

    void RNN::CleanContext() {
        impl_->net.CleanContext();
    }

    std::vector<float> RNN::obtainInputTarget() {
        return impl_->net.obtainInputTarget();
    }

    std::size_t RNN::errorCode() const {
        return impl_->net.par.errorCode;
    }

    std::string RNN::errorInfo() const {
        return impl_->net.par.errorBlgig;
    }

}  // namespace wuki

#endif  // INCLUDEMAIN_CPP
