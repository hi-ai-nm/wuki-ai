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
        const size_t n = a.size() < b.size() ? a.size() : b.size();
        const float dot = Simd::Runtime::dot(a.data(), b.data(), n);
        const float na = Simd::Runtime::sum_sq(a.data(), n);
        const float nb = Simd::Runtime::sum_sq(b.data(), n);
        return (na == 0.0f || nb == 0.0f) ? 0.0f : dot / (std::sqrt(na) * std::sqrt(nb));
    }

    /*
     * 模型文件路径：固定放 exe 旁边，从哪个工作目录启动都一样
     * （exeDir 取不到就退化成当前目录）。程序每次启动都重训，路径不需要可选。
     */
    inline std::string ModelPath() {
        const std::string binDir = wuki::exeDir();
        return (binDir.empty() ? std::string(".") : binDir) + "/model.wfm";
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
     * 语料文件 → 一行一句（UTF-8，空行跳过，CRLF 的 \r 会被剪掉）
     *
     * 为什么要有它：原来语料是硬编码在 Template::Init / Study 的默认参数里的 7 句
     * "hello world"，那是给机制演示用的。真语料只能是文件 —— 一百多万行不可能写成
     * 默认值，也不该编进二进制。
     *
     * 文件格式按仓库里那份 corpus.txt：每行一句、空格分好词、句尾自带 </end>
     * （所以 MakeStream 那个"句间补一个 END"在真语料上要关掉，见 insertEnd）。
     *
     * @param maxLines 最多读多少行（0 = 全读）。真语料有 110 万行，先截一段再训练是常规做法：
     *                 训练开销是 epochs × 样本数，词表大小也会跟着涨
     * @return 0 成功；1 文件打不开；2 一行都没读到（空文件 / 全是空行）
     */
    inline size_t LoadCorpus(const std::string& path, std::vector<std::string>& corpus, const size_t maxLines = 0) {
        corpus.clear();
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return 1;
        }

        std::string line;
        while ((maxLines == 0 || corpus.size() < maxLines) && std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {  // CRLF 文件：留着会让 </end> 编不出来
                line.pop_back();
            }
            if (line.empty()) {
                continue;  // 空行跳过：GetLine 出来的空串进 Encode 是空，白占一个样本位
            }
            corpus.push_back(std::move(line));
        }
        return corpus.empty() ? 2 : 0;
    }

    /*
     * 生成时的候选 token：语料里出现过的 id（按首次出现顺序去重）+ 那个结束符
     *
     * 生成只在候选里做最近邻：全词表找的话，没经过训练的 token（字节 / 特殊 token）会搅局，
     * 生成出来就是乱码（见 llm.hpp 的 GenerateStream）。候选里含结束符，
     * 所以模型学没学会「到此为止」，从生成结果里看得出来。
     *
     * MakeStream（训练时顺手把候选带出来）和 Template::Run（直接生成时现扫一份）共用这里，
     * 免得「候选是什么」出现两份实现。
     *
     * 结束符的串由 endTOKEN 传进来，不再在这里写死字面量（原来这是 END 的第三份拷贝，
     * 改特殊 token 时得同时改 llm.hpp 的 InitPar、GenerateStream 的默认参和这里）。
     * 玩具语料的结束符是 "@__SYSTEM<END>__"；真语料自己带 "</end>"，直接用它。
     *
     * @param eos      顺带把结束符的 id 交出去（MakeStream 要往流里插它）；不要就传 nullptr
     * @param endTOKEN 结束符的串，必须和 specialTokensTOKEN 里那个、以及 GenerateStream 的 end 一致
     * @return 0 成功；3 结束符不在词表里 / 某句编码不出来（candidates 内容不可靠）
     */
    inline size_t Candidates(LLM& llm, const std::vector<std::string>& corpus,
                             std::vector<size_t>& candidates, size_t* eos = nullptr,
                             const std::string& endTOKEN = "@__SYSTEM<END>__") {
        candidates.clear();

        const std::vector<size_t> eosIds = llm.data.tokenizer.Encode(endTOKEN);
        if (eosIds.empty()) {  // 特殊 token 不在词表里（specialTokensTOKEN 被改过、没带它）
                               // 也可能是「表里有、串不一样」：这里的串和 GenerateStream 的 end 必须是同一个
            std::cout << "结束符不在词表里：specialTokensTOKEN 里得有 \"" << endTOKEN << "\"\n";
            return 3;
        }
        if (eos != nullptr) {
            *eos = eosIds[0];
        }

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
                remember(id);
            }
            remember(eosIds[0]);  // 句间那个 <END> 也进候选（生成时模型可以学「到此为止」）
        }
        return 0;
    }

    /*
     * 滑窗训练样本：语料编成一条 token 流（句与句之间插结束符），流里每个位置
     * 当一次「下一个」，前面 window 个 token 当上下文。
     *
     * 滑窗可以跨过结束符：「结束符后面跟着某句首词」正是想让它学的转移之一。
     * 结束符的 id 和候选 token 都由 Candidates 扫出来（那份注释里有 endTOKEN 的说明）。
     *
     * insertEnd 为什么要有：真语料的每一行**自己就以 </end> 结尾**（见 LoadCorpus），
     * 再插一个就成了每句两个结束符 —— 训练目标会变成「吐完句子先吐 END、再吐一个 END」，
     * 生成时也会莫名其妙多停一步。所以真语料传 false，玩具语料（"hello world" 这种
     * 不带标记的句子）传 true。
     *
     * @param window    上下文宽度（几个 token，时序深度，和词向量维度无关）
     * @param endTOKEN  结束符的串（要和 specialTokensTOKEN、GenerateStream 的 end 一致）
     * @param insertEnd 是否在每句末尾补一个结束符；语料自带就传 false
     * @return 0 成功；3 编码失败 / 凑不出任何样本（stream / samples 内容不可靠）
     */
    inline size_t MakeStream(LLM& llm, const std::vector<std::string>& corpus, const size_t window,
                             std::vector<size_t>& stream,
                             std::vector<WindowSample>& samples,
                             std::vector<size_t>& candidates,
                             const std::string& endTOKEN = "@__SYSTEM<END>__",
                             const bool insertEnd = true) {
        stream.clear();
        samples.clear();

        size_t eos = 0;
        if (const size_t code = Candidates(llm, corpus, candidates, &eos, endTOKEN)) {  // 候选（含结束符）和它的 id
            return code;
        }

        for (const std::string& sentence : corpus) {
            const std::vector<size_t> ids = llm.data.tokenizer.Encode(sentence);  // 上面扫候选时已经确认编得出
            for (const size_t id : ids) {
                stream.push_back(id);
            }
            if (insertEnd) {
                stream.push_back(eos);  // 句与句之间隔一个结束符（语料自带的话就不用补）
            }
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
                          << (total / static_cast<float>(samples.size()))
                          << "，allRNN 块0输出范数 " << llm.DebugAllNorm0(samples.front().inputs) << "\n";
            }
        }
        return 0;
    }

    /*
     * 预训练 allRNN：沿 token 流滑动取连续 preWindow+1 个 token，
     * 正序喂、倒序挂目标（见 llm.hpp 的 PretrainOne），让它先学会「预测后面的所有」；
     * 联合训练（TrainDRNN）时 allRNN 只按微调分数小步走。
     *
     * @return 0 成功；4 预训练失败（细节在 llm.data.drnn.errorCode / errorBlgig）
     */
    inline size_t PretrainAllRNN(LLM& llm, const std::vector<size_t>& stream,
                                 const size_t preWindow, const size_t epochs,
                                 const float wlr, const float blr,
                                 const size_t logEvery = 500) {
        if (stream.size() < preWindow + 2) {  // 至少得有一个完整 slice（preWindow+1 个 token）
            std::cout << "token 流太短：不够预训练窗口\n";
            return 4;
        }
        const size_t nSamples = stream.size() - preWindow;  // 起点 t = 0 .. nSamples-1
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            float total = 0.0f;
            for (size_t t = 0; t < nSamples; ++t) {
                const std::vector<size_t> slice(stream.begin() + t, stream.begin() + t + preWindow + 1);
                auto [mse, code] = llm.PretrainOne(slice, wlr, blr);
                if (code != 0) {
                    std::cout << "预训练 error: slice " << t << "，码 " << code
                              << "，drnn 码 " << llm.data.drnn.errorCode
                              << " (" << llm.data.drnn.errorBlgig << ")\n";
                    return 4;
                }
                total += mse;
            }
            if (epoch % logEvery == 0 || epoch + 1 == epochs) {  // 隔一会儿看一眼误差有没有在降
                std::cout << "pretrain epoch " << epoch << " 平均 loss "
                          << (total / static_cast<float>(nSamples))
                          << "，allRNN 块0输出范数 " << llm.DebugAllNorm0({ stream[0] }) << "\n";
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

    // 把尺寸列表打成 "{ 8, 16, 8 }"（Init 打印解析后的结构用）
    inline std::string ShowSizes(const std::vector<size_t>& sizes) {
        if (sizes.empty()) {
            return "{ }";
        }
        std::string s = "{ ";
        for (size_t i = 0; i < sizes.size(); i++) {
            s += std::to_string(sizes[i]);
            if (i + 1 < sizes.size()) {
                s += ", ";
            }
        }
        return s + " }";
    }

    /*
     * 提示词 → 新生成的文本（网页 / 服务端用：src/comm 的 FlaskIO 拿它当回调）
     *
     * 和 Template::Run 是同一条生成路径（候选来自语料、<END> 的串从词表解出来、温度采样），
     * 区别只在交出去的形态：Run 打印过程并交 id 序列，这里直接把「模型新接上的那部分」
     * 解成文本，不含提示词本身 —— 网页上要显示的是回复，不是「提示词 + 回复」。
     *
     * 提示词编不出来 / 候选为空都返回空串（调用方按「这次没生成出东西」处理，不算错误）；
     * 想知道是哪一步卡住的就用 Template::Run，原因会打在 stdout 上。
     *
     * 这个重载要调用方把 candidates / eos 传进来：候选集合只跟 (语料, 词表) 有关、随请求不变，
     * 而扫一遍 55 万行的真语料要一两分钟 —— 每个请求都扫一遍，服务就不可用了。
     * serve 启动时用 Candidates() 算一次，回调里反复复用（见 main.cpp 的 serve 分支）。
     *
     * @param candidates Candidates() 扫出来的候选 token（空 = 上游没成功，直接交回空串）
     * @param eos        结束符的 id（Candidates() 的第二个产出）
     * @param genLen      最多再接多少个 token（默认值和 Run 一样）
     * @param temperature 采样温度（0 = 贪心；见 llm.hpp 的 GenerateStream / WorEmbd::Sample）
     */
    inline std::string ReplyText(LLM& llm, const std::string& prompt,
                                 const std::vector<size_t>& candidates, const size_t eos,
                                 const size_t genLen = 24, const float temperature = 0.05f) {
        if (candidates.empty()) {
            return std::string{};
        }

        // end 参数也不在这抄字面量：结束符的串直接从词表解出来（和 Run 一致）
        const std::vector<size_t> ids = llm.GenerateStream(prompt, genLen, candidates,
                                                           llm.data.tokenizer.Decode({ eos }), temperature);
        if (ids.empty()) {
            return std::string{};  // 提示词编不出来（GenerateStream 返回空）
        }

        // 整段解码再切掉提示词：提示词那段的字节长度就是前缀长度。
        // 比按 token 个数切安全 —— 切在 merge 出来的 token 中间会把多字节字符劈开
        const std::string text = llm.data.tokenizer.Decode(ids);
        const std::string head = llm.data.tokenizer.Decode(llm.data.tokenizer.Encode(prompt));
        if (text.size() <= head.size()) {
            return std::string{};
        }

        /*
         * 尾巴上的空白剪掉。
         *
         * 为什么会有：语料是 7 句 "hello world"，而 MakeStream 是按整句 Encode 建 token 流的，
         * 句中/句尾那个空格也一并进了训练 —— 于是模型学到的就包括「空格 → 下一个词」。
         * 生成时接完 "world" 之后它会顺着这条路一直吐空格，直到 genLen 用完，
         * 结果就是「一个词 + 二十来个空格」（实测 24 个 token 里后 20 个是空格）。
         * 这不是 bug，是这份玩具语料本身的性质。
         *
         * 但对「网页上显示一句话」来说首尾空白没有信息量，这里直接剪掉。
         * 整条都是空白的情况（模型偶尔真的全程只吐空格）剪成空串，
         * 调用方按「这次没生成出东西」处理，前端会显示「（Wuki 没说话）」。
         *
         * 想从根上少见到空格，要在语料 / 训练那边下手（让模型学会用 <END> 收尾，
         * 或者给句尾一个明确的结束标记），不是这一层能解决的。
         */
        std::string reply = text.substr(head.size());
        const std::string::size_type last = reply.find_last_not_of(" \t\r\n");
        reply.erase(last == std::string::npos ? 0 : last + 1);  // 全是空白 → 剪成空串
        return reply;
    }

    /*
     * 便利重载：候选现扫。每调用一次就把整份语料 Encode 一遍，
     * 只适合单发场景 / 玩具语料（真语料请用上面那个，自己算一次候选再复用）。
     *
     * @param corpus   候选 token 的来源，必须和 Init / Study 用的是同一份
     * @param endTOKEN 结束符的串（必须和 specialTokensTOKEN 里那个一致）
     */
    inline std::string ReplyText(LLM& llm, const std::string& prompt,
                                 const std::vector<std::string>& corpus = {
                                     "hello world",
                                     "hello world",
                                     "hello world",
                                     "hello world",
                                     "hello world",
                                     "hello world",
                                     "hello world",
                                 },
                                 const size_t genLen = 24,
                                 const float temperature = 0.05f,
                                 const std::string& endTOKEN = "@__SYSTEM<END>__") {
        std::vector<size_t> candidates;
        size_t eos = 0;
        if (Candidates(llm, corpus, candidates, &eos, endTOKEN) != 0) {
            return std::string{};
        }
        return ReplyText(llm, prompt, candidates, eos, genLen, temperature);
    }

    class Template {
    public:
        /*
         * 把 Init 的一堆参数组装成 LLM::InitPar
         *
         * 「没指定」的哨兵：0 / 空表示「按 embdSize 推导」（维度 0 和空尺寸本身都是非法值，
         * 不会和真实取值撞车）；给了具体值就照用。推导规则见 Init 的注释。
         *
         * @param specialTokensTOKEN 空 = 沿用 LLM::InitPar 里的默认。
         *        END 字面量已经有四份拷贝了（见 InitPar 的说明），Init 不再抄第五份：
         *        这里留空就等于「谁定义谁负责」，改特殊 token 只需要改 InitPar 那一处
         * @param minVEC / maxVEC 词向量的初始化区间；默认值和 InitPar 一样，改 InitPar 记得同步
         */
        inline LLM::InitPar MakeInitPar(const size_t embdSize,
                                        const float allScale,
                                        const std::vector<std::string>& specialTokensTOKEN,
                                        const float minVEC, const float maxVEC,
                                        const std::vector<size_t>& outRnnNeurDRNN,
                                        const std::vector<size_t>& allRnnNeurDRNN,
                                        const size_t onceAllInputNumDRNN,
                                        const size_t onceAllOutputNumDRNN,
                                        const size_t outRnnHSizeDRNN,
                                        const size_t allRnnHSizeDRNN,
                                        const std::string& actFunDRNN) {
            LLM::InitPar par;
            if (!specialTokensTOKEN.empty()) {  // 空 = 用 InitPar 自带的默认（不让这里变成第二真相源）
                par.specialTokensTOKEN = specialTokensTOKEN;
            }
            par.tokenSizeVEC = 0;  // 产出：InitToken 学完 merge 才知道，调用方别自己猜
            par.outputSizeVEC = embdSize;
            par.minVEC = minVEC;
            par.maxVEC = maxVEC;

            // 空 / 0 = 按 embdSize 推导（为什么这么定，见 Init 的注释）；给了就用给的
            par.outRnnNeurDRNN = outRnnNeurDRNN.empty()
                ? std::vector<size_t>{ embdSize, embdSize * 2, embdSize } : outRnnNeurDRNN;
            par.allRnnNeurDRNN = allRnnNeurDRNN.empty()
                ? std::vector<size_t>{ embdSize * 2 } : allRnnNeurDRNN;
            par.onceAllInputNumDRNN = (onceAllInputNumDRNN == 0) ? embdSize : onceAllInputNumDRNN;
            par.onceAllOutputNumDRNN = (onceAllOutputNumDRNN == 0) ? embdSize : onceAllOutputNumDRNN;
            par.outRnnHSizeDRNN = (outRnnHSizeDRNN == 0) ? embdSize : outRnnHSizeDRNN;
            par.allRnnHSizeDRNN = (allRnnHSizeDRNN == 0) ? embdSize : allRnnHSizeDRNN;
            par.allTrainScaleDRNN = allScale;
            par.actFunDRNN = actFunDRNN;
            return par;
        }

        /*
         * 初始化：分词器立词表学 merge → 词表大小定了 → 词嵌入 + DRNN 一起初始化
         *
         * 这就是 Retrain 的第 1 步，单独抽出来是为了能被别的入口复用（比如以后 -W 起网页服务
         * 前先把模型备好：要么 Init 完接着 Load 权重，要么 Init 完从头训）。顺序不能颠倒，
         * 原因见 llm.hpp 顶上的类注释（InitAll 会读 tokenSizeVEC，而它只有 merge 学完才存在；
         * 反过来先 InitAll 再学 merge，BBPE::Init 会把刚学出来的 merge 全抹掉）。
         *
         * 参数全部带默认值，默认值就是 Retrain 调它时传的那套值（Retrain 的「0. 参数」：
         * 语料 7 句 / numMerges 64 / embdSize 8 / allScale 1.0），结构参数也是写死的数
         * （按 embdSize = 8 配的一套，见下）。想换配置就在调用时按位置覆盖：
         *   Init(llm);
         *   Init(llm, corpus, numMerges, embdSize, allScale);
         *   Init(llm, corpus, numMerges, embdSize, allScale, specialTokensTOKEN, -0.7f, 0.7f, { 4, 8, 4 }, ...);
         *
         * 默认参数不能引用别的参数（C++ 不让），所以结构那几个值只能写成具体的数：
         *   outRnnNeurDRNN   = { 8, 16, 8 }
         *       首项是 outRNN 的输入层宽度（= allRNN 一次产出的宽度），末项是 DRNN 最终输出宽度
         *   allRnnNeurDRNN   = { 40, 16, 32 }（不含输入层和输出层）
         *   onceAllInputNum / onceAllOutputNum = 8 = embdSize
         *       一次正好喂进一个 token 的词向量，产出也正好一个词向量
         *       → sliceWidth = outRnnNeur[0] = 8，sliceNum = onceAllOutputNum / sliceWidth = 1
         *         （不切片），于是「下一个 token 的词向量」可以直接当目标用，用不着再拆 / 拼
         *       这两个数还得能被 outRnnNeur[0] 整除（DRNN::Init 会校验），改了要一起看
         *   outRnnHSize / allRnnHSize = 30 / 60
         *       h 和词向量维度没有关系，纯粹是各层自己的宽度
         *   actFunDRNN = "scth"（自创 tanh，±2 饱和）—— 纯 tanh 只有 ±1，够不着 skip-gram 训完的
         *       词向量（±1.5 以上），目标永远差一截，权重会「够不着 → 持续顶格更新」地漂移
         *
         * 换了 embdSize 就得手改上面那六个（默认值不会跟着动）。懒得改就显式传 0 / 空：
         * 结构参数传 0 / {} 时按 embdSize 推导（MakeInitPar 里那份），等价于上面这套数。
         *
         * 注意词表大小不在参数里：它是分词器的产出（256 字节 + 特殊 token + 学到的 merge），
         * 由 InitLLM 从 InitToken 的返回值填进 InitPar —— 调用方想自己定只能靠 numMerges 和语料。
         *
         * @param corpus      学 merge 用的语料（决定词表里有哪些 token，也决定后面训练覆盖什么）
         * @param numMerges   BBPE 学的合并轮数（词表最多长大这么多，没东西可合就提前停）
         * @param embdSize    词向量维度，整套 DRNN 尺寸的锚
         * @param allTrainScaleDRNN 联合训练时 allRNN 的学习率微调分数（Retrain 里那份叫 allScale）
         * @param specialTokensTOKEN 进词表的特殊 token（默认只有 <END>；"@__CMD<PRINT>__" 没在这，
         *        它本来也只是占个 id 没人消费）。这个串是 END 字面量的又一份拷贝 ——
         *        改 llm.hpp 的 InitPar::specialTokensTOKEN 时这里要一起改，不然 MakeStream / Run
         *        扫候选时会以 3 号退出码停下（那是整套里唯一会「响亮失败」的一处）
         * @param minVEC / maxVEC 词向量的初始化区间；默认值和 InitPar 一样，改 InitPar 记得同步
         * @param outRnnNeurDRNN / allRnnNeurDRNN / onceAll*NumDRNN / *HSizeDRNN
         *        默认就是上面那套数；传空 / 0 = 按 embdSize 推导，传值 = 照用
         * @return 0 成功；1 词向量初始化失败（含 embdSize = 0）；2 DRNN 初始化失败
         *         （结构参数不合法的归属打印在 drnn.errorBlgig 上）
         */
        inline int Init(LLM& llm,
                        const std::vector<std::string>& corpus = {
                            "hello world",
                            "hello world",
                            "hello world",
                            "hello world",
                            "hello world",
                            "hello world",
                            "hello world",
                        },
                        const size_t numMerges = 64,
                        const size_t embdSize = 8,
                        const float allTrainScaleDRNN = 1.0f,
                        const std::vector<std::string>& specialTokensTOKEN = { "@__SYSTEM<END>__" },
                        const float minVEC = -0.7f, const float maxVEC = 0.7f,
                        const std::vector<size_t>& outRnnNeurDRNN = { 8, 16, 8 },
                        const std::vector<size_t>& allRnnNeurDRNN = { 40, 16, 32 },
                        const size_t onceAllInputNumDRNN = 8,
                        const size_t onceAllOutputNumDRNN = 8,
                        const size_t outRnnHSizeDRNN = 30,
                        const size_t allRnnHSizeDRNN = 60,
                        const std::string& actFunDRNN = "scth") {
            if (embdSize == 0) {  // 锚是 0 什么都推不出来，早点停下比让 vec.Init 报 1 号错误码清楚
                std::cout << "初始化失败: embdSize = 0（词向量维度是整套尺寸的锚）\n";
                return 1;
            }
            if (corpus.empty()) {  // 不致命：词表还是立得起来，只是学不到 merge
                std::cout << "语料为空：分词器只立基础词表（256 字节 + 特殊 token），学不到 merge\n";
            }

            LLM::InitPar par = MakeInitPar(embdSize, allTrainScaleDRNN, specialTokensTOKEN,
                                           minVEC, maxVEC, outRnnNeurDRNN, allRnnNeurDRNN,
                                           onceAllInputNumDRNN, onceAllOutputNumDRNN,
                                           outRnnHSizeDRNN, allRnnHSizeDRNN, actFunDRNN);
            // InitLLM 是 non-const 引用：它要塞 tokenSizeVEC（分词器的产出），所以要个可写的 par
            const int code = static_cast<int>(encap::InitLLM(llm, par, corpus, numMerges));
            if (code != 0) {  // InitLLM 里已经打印了是哪个部件、什么错误码
                return code;
            }

            // 把「推导之后」的结构打出来：默认参数推出来的到底是什么，这里一眼能看见
            std::cout << "结构: outRNN " << ShowSizes(par.outRnnNeurDRNN) << " h " << par.outRnnHSizeDRNN
                      << "，allRNN " << ShowSizes(par.allRnnNeurDRNN) << " h " << par.allRnnHSizeDRNN
                      << "，一次进 " << par.onceAllInputNumDRNN << " 出 " << par.onceAllOutputNumDRNN
                      << "，激活 " << par.actFunDRNN
                      << "，allRNN 微调分数 " << par.allTrainScaleDRNN << "\n";
            return 0;
        }

        /*
         * 训练：Skip-gram 学共现 → 语料编成 token 流 + 滑窗样本 → 预训练 allRNN → 联合训练
         *
         * 参数的默认值就是 Retrain 调它时传的那套值（原来写死在 Retrain 的「0. 参数」里）。
         * 调之前得先 Init：词表 / 词嵌入 / 两个子网都还没立起来的话，第一步就停下了。
         *
         * 四段各自在干什么：
         *   1) Skip-gram：中心词预测上下文，正样本拉近、负采样推远 —— 让词向量带上共现语义。
         *      学完这张表就冻住，DRNN 训练和生成时都只查表、不再更新；顺手量一下
         *      余弦(hello, wuki) vs 余弦(hello, small)，没区别就是没学动（见 TrainEmbdCheck）
         *   2) 滑窗样本：语料 → 一条 token 流（句间插 <END>）→ 每个位置当一次「下一个」，
         *      前面 window 个 token 当上下文（见 MakeStream）
         *   3) 预训练 allRNN：正序读、倒序挂目标（第 k 步读 slice[k]，目标 slice[n-k]，
         *      先吐最远的未来，从 t+n 到 t）—— 让它先学会「预测后面的所有」
         *   4) 联合训练：一条 window 步的序列进，预测下一个 token 的词向量；allRNN 直挂真目标
         *      （块 t 产出 = 下一个 token 的向量），outRNN 全速学精修
         *
         * @param corpus     训练语料（决定词表覆盖什么，也决定滑窗样本从哪来）
         * @param embdWindow / embdEpochs / embdNeg / embdLr   Skip-gram 的窗口 / 轮数 / 负样本数 / 学习率
         * @param preWindow / preEpochs / preWlr / preBlr      allRNN 预训练的窗口 / 轮数 / 权重与偏置学习率
         * @param window / epochs / wlr / blr                  联合训练的滑窗宽度 / 轮数 / 权重与偏置学习率
         * @return 0 成功；1 词嵌入训练失败；3 编码失败 / 凑不出样本；4 训练失败（含预训练）
         */
        inline int Study(LLM& llm,
                         const std::vector<std::string>& corpus = {
                             "hello world",
                             "hello world",
                             "hello world",
                             "hello world",
                             "hello world",
                             "hello world",
                             "hello world",
                         },
                         const size_t embdWindow = 2,
                         const size_t embdEpochs = 3500,
                         const size_t embdNeg = 5,
                         const float embdLr = 0.05f,
                         const size_t preWindow = 8,
                         const size_t preEpochs = 3000,
                         const float preWlr = 0.001f,
                         const float preBlr = 0.00025f,
                         const size_t window = 4,
                         const size_t epochs = 17000,
                         const float wlr = 0.002f,
                         const float blr = 0.0005f,
                         const std::string& endTOKEN = "@__SYSTEM<END>__",
                         const bool insertEnd = true) {
            if (const size_t code = encap::TrainEmbdCheck(llm, corpus, embdWindow, embdEpochs, embdNeg, embdLr)) {
                return static_cast<int>(code);
            }

            // 候选 token 这里用不上（Run 自己从语料扫一份），MakeStream 要就给它一个
            std::vector<size_t> stream;
            std::vector<encap::WindowSample> samples;
            std::vector<size_t> candidates;
            if (const size_t code = encap::MakeStream(llm, corpus, window, stream, samples, candidates,
                                                      endTOKEN, insertEnd)) {
                return static_cast<int>(code);
            }

            if (const size_t code = encap::PretrainAllRNN(llm, stream, preWindow, preEpochs, preWlr, preBlr)) {
                return static_cast<int>(code);
            }

            if (const size_t code = encap::TrainDRNN(llm, samples, epochs, wlr, blr)) {
                return static_cast<int>(code);
            }
            return 0;
        }

        /*
         * 生成（推理）：每步把「提示词 + 已生成」整段喂进去（h 在整段上连续），
         * 取最后输出做最近邻还原成 token，再接上 —— 和 TrainWindow 的训练形态一致
         *
         * 候选 token 从语料现扫（语料里出现过的 + <END>，见 Candidates）：只在候选里做最近邻，
         * 没经过训练的 token（字节 / 特殊 token）才不会搅局。模型吐出 <END> 就停下，
         * 所以「学没学会收尾」从生成结果里看得出来。
         *
         * 语料只有 7 句、滑窗样本只有几十个，生成大概率还是收敛进某个循环 ——
         * 这一步是为了把整条链路跑通，不是要它写出文章。
         *
         * @param prompt      提示词（编不出来就以 3 号退出码停下）
         * @param genLen      最多再接多少个 token
         * @param temperature 采样温度（见 llm.hpp 的 GenerateStream / WorEmbd::Sample）：
         *                    0 = 贪心；调大后按 softmax(−距离/T) 采样，量纲是平方距离，没有通用值
         * @param generated   要接着用生成结果就传它（存 / 取往返的探测、喂网页）；nullptr = 只要打印。
         *                    内容 = 提示词本身的 id + 新生成的 id（和 GenerateStream 的返回一致）
         * @return 0 成功；3 扫不出候选 / 提示词编码失败
         */
        inline int Run(LLM& llm,
                       const std::string& prompt = "hello",
                       const std::vector<std::string>& corpus = {
                           "hello world",
                           "hello world",
                           "hello world",
                           "hello world",
                           "hello world",
                           "hello world",
                           "hello world",
                       },
                       const size_t genLen = 24,
                       const float temperature = 0.05f,
                       std::vector<size_t>* generated = nullptr,
                       const std::string& endTOKEN = "@__SYSTEM<END>__") {
            std::vector<size_t> candidates;
            size_t eos = 0;
            if (const size_t code = encap::Candidates(llm, corpus, candidates, &eos, endTOKEN)) {
                return static_cast<int>(code);
            }

            // end 参数不在这抄字面量：结束符的串直接从词表解出来用（Decode 出来就是它）
            const std::vector<size_t> ids = llm.GenerateStream(prompt, genLen, candidates,
                                                               llm.data.tokenizer.Decode({ eos }), temperature);
            if (ids.empty()) {
                std::cout << "提示词编码失败: " << prompt << "\n";
                return 3;
            }
            if (generated != nullptr) {
                *generated = ids;  // 交出去：后面的存 / 取往返拿它当探测输入
            }

            const size_t head = llm.data.tokenizer.Encode(prompt).size();  // 前面这几格是提示词，后面才是生成的
            std::cout << "提示词: " << prompt << "\n";
            std::cout << "生成  : " << llm.data.tokenizer.Decode(ids)
                      << "（其中新生成 " << (ids.size() - head) << " 个 token）\n";
            return 0;
        }
    };

}  // namespace encap

#endif  // ENCAP_HPP
