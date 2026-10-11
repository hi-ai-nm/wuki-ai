#include "llm/framework.hpp"
#include "llm/architecture.hpp"
#include "llm/DRNNSL.hpp"  // 双层架构的存 / 取（要放在 architecture.hpp 之后）
#include "llm/bbpe.hpp"
#include "llm/wet.hpp"

#ifndef LLM_HPP
#define LLM_HPP

#include <map>
#include <string>
#include <utility>  // std::pair
#include <vector>

/*
 * LLM：把「BBPE 分词 → 词嵌入 → 双层 RNN」整条流水线收进一个类
 *
 * 初始化必须分两步，顺序不能乱：
 *   1) InitToken —— 先立分词器、用语料学 merge。词表大小是这一步的产出，
 *                   后面词嵌入的行数、DRNN 的尺寸全都围着它转
 *   2) InitAll   —— 拿词表大小初始化词嵌入和 DRNN
 *
 * 为什么不能一步：
 *   词嵌入的行数 = 词表大小，merge 没学完之前这个数字根本不存在；
 *   而 BBPE::Init 会清空整张词表，先 InitAll 再学 merge，等于把刚学出来的 merge 全抹掉。
 */
class LLM {
  public:
    struct Data {
        BBpe::BBPE tokenizer;  // 分词器（词表 + merge 规则）
        WorEmbd vec;           // 词嵌入表（先经 Skip-gram 学共现，见 TrainEmbd；DRNN 训练阶段冻住）
        Arch::DRNN drnn;       // 双层 RNN（整条流水线里唯一在训练的东西）
    } data;

    // 各部件「初始化成功」的错误码都是 0（分词器没有错误码，恒为 0），
    // InitAll 返回的 map 和它逐项比就知道有没有初始化成功
    inline const static std::map<std::string, size_t> _INIT_OK = {
        {"tokenizer", 0},
        {"vec", 0},
        {"drnn", 0}
    };

    // 初始化参数（全集中在这一个结构里，main 只填它）
    struct InitPar {
        // tokenizer 的（InitToken 用）
        std::vector<std::string> specialTokensTOKEN = { "@__SYSTEM<END>__" , "@__CMD<PRINT>__"};
        // vec 的
        size_t tokenSizeVEC = 0;    // 词表大小：吃 InitToken 的返回值，别自己猜
        size_t outputSizeVEC = 0;   // 词向量维度
        float minVEC = -0.7f; float maxVEC = 0.7f;
        // drnn 的
        std::vector<size_t> outRnnNeurDRNN = {}; /* outRNN 的完整 neur：含输入层和输出层，不含 h */
        std::vector<size_t> allRnnNeurDRNN = { 48 }; /* 不含输入层和输出层 */
        size_t onceAllInputNumDRNN = 32;
        size_t onceAllOutputNumDRNN = 32;
        size_t outRnnHSizeDRNN = 16; size_t allRnnHSizeDRNN = 32;
    };

    /*
     * 阶段 1：分词器。Init 立基础词表（256 字节 + 特殊 token），再用语料学 numMerges 轮 merge
     *
     * @return 学完之后词表的大小（256 + 特殊 token 数 + 实际学到的 merge 数），
     *         也就是 InitAll 里 tokenSizeVEC 该填的值
     */
    inline size_t InitToken(const InitPar& par, const std::vector<std::string>& corpus, const size_t numMerges = 0) {
        data.tokenizer.Init(par.specialTokensTOKEN);
        if (numMerges > 0) {
            data.tokenizer.Train(data.tokenizer.ConvertCorpus(corpus), numMerges);
        }
        return data.tokenizer.model.tokens.size();
    }

    /*
     * 阶段 2：词嵌入 + DRNN（词表定了之后才能定尺寸，所以排在 InitToken 后面）
     *
     * @return 每个部件的错误码，和 _INIT_OK 逐项比即可；vec / drnn 非 0 时细节看它们的错误归属
     */
    inline std::map<std::string, size_t> InitAll(const InitPar& par) { // 初始化 vec 和 drnn（tokenizer 在阶段 1 已完成）
        std::map<std::string, size_t> result = {{"tokenizer", 0}};  // 分词器是 void，走到这它就是成功的
        result["vec"] = data.vec.Init(par.tokenSizeVEC, par.outputSizeVEC, par.minVEC, par.maxVEC);
        result["drnn"] = data.drnn.Init(par.outRnnNeurDRNN, par.allRnnNeurDRNN, par.onceAllInputNumDRNN, par.onceAllOutputNumDRNN, par.outRnnHSizeDRNN, par.allRnnHSizeDRNN);
        return result;
    }

    /*
     * 词嵌入的 Skip-gram 训练：语料 → id 序列 → data.vec.Train（正样本拉近、负采样推远）
     *
     * 时序：InitAll 之后、DRNN 训练之前 —— 先让词向量带上共现语义（hello 离 wuki 近），
     * 再把表冻住，让 DRNN 学「词向量 → 下一个词向量」的转移。
     *
     * @return 0 成功；1 词嵌入还没初始化（InitAll 没成功就调它）
     */
    inline size_t TrainEmbd(const std::vector<std::string>& corpus, const size_t window = 2, const size_t epochs = 1, const size_t negNum = 5, const float lr = 0.05f) {
        std::vector<std::vector<size_t>> sentences;
        sentences.reserve(corpus.size());
        for (const auto& sentence : corpus) {
            const std::vector<size_t> ids = data.tokenizer.Encode(sentence);
            if (!ids.empty()) {
                sentences.push_back(ids);
            }
        }
        return data.vec.Train(sentences, window, epochs, negNum, lr);
    }

    /*
     * 滑窗训练：连续 w 个 token 当输入（w 步序列），预测下一个 token
     *
     * allRNN 把这段序列分块读完（块间 h 连续 —— 这才是双层架构的正经用法），
     * outRNN 逐步走，只有最后一片吃真目标（target 的词向量），前面的片拿实际输出
     * 当目标 + h 目标回填（监督结构见 architecture.hpp 的 Train）。
     * 词向量在 DRNN 训练阶段是冻住的（共现语义由 Skip-gram 阶段学好了）。
     *
     * @param inputs 连续的 w 个 token id（w ≥ 1）
     * @return { 最后一片（真正有监督那片）的 MSE, 错误码 }
     */
    inline std::pair<float, size_t> TrainWindow(const std::vector<size_t>& inputs, const size_t target, const float wlr, const float blr) {
        std::vector<std::vector<float>> seq;
        seq.reserve(inputs.size());
        for (const size_t id : inputs) {
            seq.push_back(data.vec.Get(id));
        }
        auto [loss, code] = data.drnn.Train(seq, data.vec.Get(target), wlr, blr);
        if (code != 0x00) {
            return { 0.0f, code };  // 细节看 data.drnn.errorCode / errorBlgig
        }
        if (loss.empty()) {  // Train 报了成功却没给 loss，正常不该发生
            return { 0.0f, 0x03 };
        }
        return { loss.back(), 0x00 };  // loss 是 outRNN 逐步的误差，只有最后一片有外部监督
    }

    /*
     * 流式生成：把「提示词 + 已生成」整段一次性喂进去，取最后一个输出做最近邻，
     * 还原成 token 接到序列末尾 —— 每一步都带着完整上下文（h 在整段上连续），
     * 和 TrainWindow 的训练形态一致。
     *
     * 只在 candidates（语料里出现过的 token）里做最近邻：全词表找的话，没经过
     * 训练的 token（字节 / 特殊 token）会搅局，生成出来就是乱码。
     * 模型吐出 end token（<END>）就到此为止：它学没学会「结束」从这里能看出来。
     *
     * @return 完整 id 序列（开头是提示词本身，后面才是生成的）；
     *         提示词编不出来返回空（调用方按「编码失败」处理）
     */
    inline std::vector<size_t> GenerateStream(const std::string& prompt, const size_t genLen, const std::vector<size_t>& candidates, const std::string end = "@__SYSTEM<END>__") {
        std::vector<size_t> ids = data.tokenizer.Encode(prompt);
        for (size_t i = 0; i < genLen && !ids.empty(); i++) {
            std::vector<std::vector<float>> seq;
            seq.reserve(ids.size());
            for (const size_t id : ids) {
                seq.push_back(data.vec.Get(id));
            }
            auto [out, code] = data.drnn.Run(seq);
            if (code != 0x00) {
                break;  // 生成失败就到此为止，把已经生成的部分交出去
            }
            const size_t next = data.vec.Nearest(out, candidates);
            if (next >= data.vec.tokenSize()) {
                break;  // 候选为空 / 维度对不上，还原不出来
            }

            if (data.tokenizer.Decode({ next }) == end) {
                break;  // 到达结束符就到此为止（<END> 本身不接入序列）
            }

            ids.push_back(next);
        }
        return ids;
    }

    /*
     * 存 / 取：只管 DRNN（文件格式里只有它的两个子网，词表和词嵌入不在里面 ——
     * 读回来之后要接着用，得用同一份语料 / 同一种子重建 tokenizer 和 vec）
     *
     * 失败时错误码 / 归属写在本类的 errCode / errBlgig 上（和 DRNN 一个习惯）
     */
    inline bool Save(const std::string& fileName) {
        errCode = 0x00;
        errBlgig = "not error";
        Arch::DRNNSL saveLoad;
        const bool ok = saveLoad.Save(data.drnn, fileName);
        errCode = saveLoad.errorCode;
        errBlgig = saveLoad.errorBlgig;
        return ok;
    }
    inline bool Load(const std::string& fileName) {
        errCode = 0x00;
        errBlgig = "not error";
        Arch::DRNNSL saveLoad;
        const bool ok = saveLoad.Load(data.drnn, fileName);
        errCode = saveLoad.errorCode;
        errBlgig = saveLoad.errorBlgig;
        return ok;
    }

    size_t errCode = 0x00;               // 最近一次 Save / Load 的错误码（0x00 表示没有错误）
    std::string errBlgig = "not error";  // 错误归属（哪个阶段出的错）
};

#endif // LLM_HPP
