#pragma once
#ifndef ENCAP_HPP
#define ENCAP_HPP

#include "llm.hpp"  // LLM（顺带拿到 framework / architecture / DRNNSL / bbpe / wet）

#include <cmath>     // std::sqrt（余弦相似度）
#include <iostream>  // std::cout
#include <map>       // std::map（InitAll 的结果比对）
#include <string>    // std::string
#include <utility>   // std::pair
#include <vector>    // std::vector

/*
 * encap：main.cpp 里直写的几段流水线逻辑，收进来让 main 只剩「参数 + 调用顺序」
 *
 * 每个函数自己打印过程信息，返回值沿用 main 的退出码语义（0 成功，非 0 直接 return 出去）：
 *   1 词嵌入初始化 / 训练失败 | 2 DRNN 初始化失败 | 3 语料编码失败
 *   4 训练失败 | 5 Save 失败 | 6 Load 失败 | 7 存 / 取往返不一致
 */

namespace encap {

    // 余弦相似度（Skip-gram 学没学动的体检指标）
    inline float Cosine(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (size_t j = 0; j < a.size() && j < b.size(); ++j) {
            dot += a[j] * b[j];
            na += a[j] * a[j];
            nb += b[j] * b[j];
        }
        return (na == 0.0f || nb == 0.0f) ? 0.0f : dot / (std::sqrt(na) * std::sqrt(nb));
    }

    /*
     * 模型文件路径：命令行给了就用命令行（容器里配 -v 挂载的场景），
     * 否则放 exe 旁边，从哪个工作目录启动都一样（exeDir 取不到就退化成当前目录）
     */
    inline std::string ModelPath(const char* arg /* nullptr = 命令行没给 */) {
        if (arg != nullptr) {
            return arg;
        }
        const std::string binDir = wuki::exeDir();
        return (binDir.empty() ? std::string(".") : binDir) + "/drnn.model.json";
    }

    /*
     * LLM 两步初始化：分词器先立词表学 merge，词表大小定了再初始化词嵌入和 DRNN
     * （分两步的原因见 llm.hpp 的类注释：vec 的行数 = 词表大小，merge 学完才存在）
     *
     * @param par 调用方填好除 tokenSizeVEC 以外的字段（那是本函数的产出）
     * @return 0 成功；1 vec 初始化失败；2 DRNN 初始化失败
     */
    inline size_t InitLLM(LLM& llm, LLM::InitPar& par,
                          const std::vector<std::string>& corpus, const size_t numMerges) {
        par.tokenSizeVEC = llm.InitToken(par, corpus, numMerges);
        std::cout << "词表大小: " << par.tokenSizeVEC
                  << "（256 字节 + " << par.specialTokensTOKEN.size()
                  << " 特殊 token + BBPE merge，实际学到的没那么多）\n";

        const std::map<std::string, size_t> initResult = llm.InitAll(par);
        for (const auto& [name, okCode] : LLM::_INIT_OK) {  // 哪个部件非 0 就报哪个，退出码分开
            if (initResult.at(name) != okCode) {
                std::cout << "初始化失败: " << name << " 错误码 " << initResult.at(name);
                if (name == "drnn") {
                    std::cout << " (" << llm.data.drnn.errorBlgig << ")";  // 结构参数不合法的归属
                }
                std::cout << "\n";
                return name == "vec" ? 1 : 2;
            }
        }
        std::cout << "词向量: " << llm.data.vec.tokenSize() << " × " << llm.data.vec.embdSize() << "\n";
        return 0;
    }

    /*
     * 词嵌入的 Skip-gram 训练 + 体检：训练完打印 hello·wuki vs hello·small 的余弦，
     * 前者在语料里老挨着、应该明显高于后者 —— 没区别就是没学动（学习率 / 轮数不够）。
     *
     * （探测词写死了 hello / wuki / small，和 main 的演示语料绑在一起；换语料得换这里。）
     *
     * @return 0 成功；1 词嵌入还没初始化就训练（InitAll 没成功就调了）
     */
    inline size_t TrainEmbdCheck(LLM& llm, const std::vector<std::string>& corpus,
                                 const size_t window, const size_t epochs,
                                 const size_t negNum, const float lr) {
        if (llm.TrainEmbd(corpus, window, epochs, negNum, lr) != 0) {
            std::cout << "词嵌入还没初始化就训练（InitAll 没成功就调了 TrainEmbd）\n";
            return 1;
        }
        std::cout << "skip-gram: window " << window << "，epochs " << epochs
                  << "，负样本 " << negNum << "\n";

        const size_t idHello = llm.data.tokenizer.Encode("hello")[0];  // 常用词都被 merge 成了单 token，取第一个 id 即可
        const size_t idWuki = llm.data.tokenizer.Encode("wuki")[0];
        const size_t idSmall = llm.data.tokenizer.Encode("small")[0];
        std::cout << "余弦(hello, wuki) = " << Cosine(llm.data.vec.Get(idHello), llm.data.vec.Get(idWuki))
                  << "，余弦(hello, small) = " << Cosine(llm.data.vec.Get(idHello), llm.data.vec.Get(idSmall)) << "\n";
        return 0;
    }

    // 一个滑窗训练样本：inputs 是 token 流里连续的 window 个 id，target 是下一个 id
    struct WindowSample {
        std::vector<size_t> inputs;
        size_t target = 0;
    };

    /*
     * 滑窗训练样本：语料编成一条 token 流（句与句之间插 <END>），流里每个位置
     * 当一次「下一个」，前面 window 个 token 当上下文。
     *
     * 滑窗可以跨过 <END>：「<END> 后面跟着某句首词」正是想让它学的转移之一。
     * <END> 的 id 从分词器现查（Encode 特殊 token 串本身），不写死 256。
     * 候选 token 也包含 <END> —— 生成时模型可以学「到此为止」。
     *
     * @param window 上下文宽度（几个 token，时序深度，和词向量维度无关）
     * @return 0 成功；3 编码失败 / 凑不出任何样本（stream / samples 内容不可靠）
     */
    inline size_t MakeStream(LLM& llm, const std::vector<std::string>& corpus, const size_t window,
                             std::vector<size_t>& stream,
                             std::vector<WindowSample>& samples,
                             std::vector<size_t>& candidates) {
        stream.clear();
        samples.clear();
        candidates.clear();

        const std::vector<size_t> eosIds = llm.data.tokenizer.Encode("@__SYSTEM<END>__");
        if (eosIds.empty()) {  // 特殊 token 不在词表里（specialTokensTOKEN 被改过、没带它）
            std::cout << "<END> 不在词表里：specialTokensTOKEN 里得有 \"@__SYSTEM<END>__\"\n";
            return 3;
        }
        const size_t eos = eosIds[0];

        std::vector<bool> seen(llm.data.vec.tokenSize(), false);
        auto remember = [&seen, &candidates](const size_t id) {
            if (!seen[id]) {  // 按出现顺序记一遍，去重
                seen[id] = true;
                candidates.push_back(id);
            }
        };

        for (const std::string& sentence : corpus) {
            const std::vector<size_t> ids = llm.data.tokenizer.Encode(sentence);
            if (ids.empty()) {  // Encode 编不出来就返回空（不会给半截）
                std::cout << "编码失败: " << sentence << "\n";
                return 3;
            }
            for (const size_t id : ids) {
                stream.push_back(id);
                remember(id);
            }
            stream.push_back(eos);  // 句与句之间隔一个 <END>
            remember(eos);
        }

        for (size_t p = window; p < stream.size(); ++p) {  // 每个位置当一次预测目标
            WindowSample s;
            s.inputs.assign(stream.begin() + (p - window), stream.begin() + p);
            s.target = stream[p];
            samples.push_back(std::move(s));
        }

        std::cout << "token 流: " << stream.size() << "，滑窗样本: " << samples.size()
                  << "，语料覆盖 token: " << candidates.size() << "\n";
        if (samples.empty()) {  // 流比窗还短，什么都构不出来
            std::cout << "语料太短：凑不出一个滑窗样本\n";
            return 3;
        }
        return 0;
    }

    /*
     * DRNN 训练：epochs 轮 × 整份滑窗样本
     *
     * 每个样本是一条 window 步的序列（inputs 逐 token 喂进去，allRNN 块间 h 连续），
     * 目标是下一个 token 的词向量（只有最后一片吃它，见 llm.hpp 的 TrainWindow）。
     *
     * @param logEvery 隔多少轮打印一次平均 loss（0 = 只打印最后一轮）
     * @return 0 成功；4 训练失败（细节在 llm.data.drnn.errorCode / errorBlgig）
     */
    inline size_t TrainDRNN(LLM& llm, const std::vector<WindowSample>& samples,
                            const size_t epochs, const float wlr, const float blr,
                            const size_t logEvery = 2000) {
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            float total = 0.0f;
            for (const auto& sample : samples) {
                auto [mse, code] = llm.TrainWindow(sample.inputs, sample.target, wlr, blr);
                if (code != 0) {
                    std::cout << "Train error: " << code << " (" << llm.data.drnn.errorBlgig << ")\n";
                    return 4;
                }
                total += mse;
            }
            if (epoch % logEvery == 0 || epoch + 1 == epochs) {  // 隔一会儿看一眼误差有没有在降
                std::cout << "epoch " << epoch << " 平均 loss "
                          << (total / static_cast<float>(samples.size())) << "\n";
            }
        }
        return 0;
    }

    /*
     * 存 / 取往返校验：存下来 → 读进另一个 LLM → 同一输入两个 DRNN 的输出逐位相同
     *
     * 注意文件里只有 DRNN 的两个子网（见 DRNNSL.hpp 的格式说明），词表和词嵌入不在
     * 里面，所以这里比较的是「同一个词向量分别喂给两个 DRNN」的结果 —— 这正好也是
     * 存 / 取唯一能承诺的东西：同一份权重、同一套前向，输出逐位相同。
     *
     * @param probe 同一个输入向量（一般从原模型的词嵌入里取一个）
     * @return 0 成功；5 Save 失败；6 Load 失败；7 往返不一致
     */
    inline size_t RoundTrip(LLM& llm, const std::string& modelPath, const std::vector<float>& probe) {
        if (!llm.Save(modelPath)) {
            std::cout << "Save 失败: " << llm.errCode << " (" << llm.errBlgig << ")\n";
            return 5;
        }
        std::cout << "Save ok: " << modelPath << "\n";

        LLM loaded;  // 新 LLM，除了这次 Load 之外没碰过训练
        if (!loaded.Load(modelPath)) {
            std::cout << "Load 失败: " << loaded.errCode << " (" << loaded.errBlgig << ")\n";
            return 6;
        }
        std::cout << "Load ok\n";

        // 同一份权重、同一套前向 → 应当逐位相同（json 里的 float 是「能精确还原」的写法）
        auto [result, error] = llm.data.drnn.Run({ probe });
        auto [result2, error2] = loaded.data.drnn.Run({ probe });

        bool same = (error == 0x00 && error2 == 0x00 && result.size() == result2.size());
        if (same) {
            for (size_t i = 0; i < result.size(); i++) {
                if (result[i] != result2[i]) {
                    same = false;
                    break;
                }
            }
        }
        std::cout << "往返一致: " << (same ? "yes" : "no") << "\n";
        return same ? 0 : 7;
    }

}  // namespace encap

#endif  // ENCAP_HPP
