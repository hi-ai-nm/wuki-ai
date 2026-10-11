#include "includeOne.hpp"  // llm.hpp（LLM 类）+ encap.hpp（流水线各阶段函数）

/*
 * wuki-ai 主程序：一条最小的「自造 LLM」流水线
 *
 *   BBPE 分词  →  Skip-gram 词嵌入（学共现）  →  双层 RNN（LLM 里的 drnn）滑窗预测下一个词
 *              →  生成的向量用最近邻还原成 token  →  解码回文本  →  存 / 取往返校验
 *
 * 各阶段的实现和原理说明都在 llm.hpp / encap.hpp / llm 子目录各头的注释里，这里只排顺序。
 *
 * 用法：
 *   wuki-ai [模型文件]
 *     不给参数：模型存在 exe 旁边（…/Bin/drnn.model.json），从哪个工作目录启动都一样
 *     给了参数：存在指定路径 —— 容器里跑的时候配 -v 挂载，才能把模型留到 host：
 *               docker run --rm -v "$PWD/Bin:/out" docker_run /out/drnn.model.json
 *               （镜像用仓库根目录的 dockerfile 构建：docker build -f dockerfile -t docker_run .）
 *
 * 尺寸为什么这么定（全都围着词向量维度 embdSize 转）：
 *   onceAllInputNum  = onceAllOutputNum = 词向量维度
 *       —— 一次喂进一个 token 的词向量（一步），DRNN 的输出也正好是词向量那么宽
 *   outRnnNeur = { 词向量维度, 隐层, 词向量维度 }
 *       —— sliceWidth = 词向量维度 → sliceNum = onceAllOutputNum / sliceWidth = 1（不切片），
 *          于是「下一 token 的词向量」可以直接当目标用，用不着再拆 / 拼
 *   hSize 和词向量维度取一样大只是为了省心，两者没有任何关系
 *
 * 返回值（每一条都是不同的退出码，脚本里一眼能看出停在哪一步）：
 *   0 成功 | 1 词嵌入初始化 / 训练失败 | 2 DRNN 初始化失败 | 3 语料 / 提示词编码失败
 *   4 训练失败 | 5 Save 失败 | 6 Load 失败 | 7 存 / 取往返结果不一致 | 8 参数太多
 */
int main(int argc, char** argv) {
    if (argc > 2) {  // 只认一个可选参数
        std::cout << "用法: " << argv[0] << " [模型文件路径]\n";
        return 8;
    }

    // ---------------------------------------------------------------------
    // 0. 参数（整条流水线的配置都集中在这）
    // ---------------------------------------------------------------------
    const std::vector<std::string> corpus = {
        "hello world",
        "hello wuki",
        "wuki is a small ai",
        "wuki learns to talk",
        "hello hello wuki",
        "a small ai learns to talk",
        "wuki talks to the world",
    };

    const size_t embdSize = 8;      // 词向量维度
    const size_t numMerges = 64;    // BBPE 学的合并轮数（词表最多长大这么多，没东西可合就提前停）
    const size_t embdWindow = 2;    // Skip-gram 左右各看几个 token 当上下文
    const size_t embdEpochs = 3000; // Skip-gram 训练轮数
    const size_t embdNeg = 5;       // 每对正样本配几个负样本（推远）
    const float embdLr = 0.05f;     // Skip-gram 学习率
    const size_t window = 4;        // 滑窗上下文宽度（几个 token 当输入，时序深度，和词向量维度无关）
    const size_t epochs = 12000;    // DRNN 训练轮数（每轮把整份滑窗样本过一遍）
    const size_t genLen = 24;       // 生成时最多再接多少个 token
    const std::string prompt = "hello";
    const float wlr = 0.002f;       // DRNN 权重学习率
    const float blr = 0.0005f;      // DRNN 偏置学习率（习惯上比权重小一个量级）
                                    // 这两个值是量出来的：同一套结构跑多次（词向量和权重
                                    // 都是随机初始化的），0.01 / 0.001 时 loss 有时不降，
                                    // 0.002 / 0.0005 每次都降、末值也最低

    // ---------------------------------------------------------------------
    // 1. 初始化：分词器立词表学 merge，词表大小定了再初始化词嵌入和 DRNN
    // ---------------------------------------------------------------------
    LLM llm;
    LLM::InitPar par;
    par.outputSizeVEC = embdSize;
    par.outRnnNeurDRNN = { embdSize, embdSize * 2, embdSize };
    par.allRnnNeurDRNN = { embdSize * 2 };
    par.onceAllInputNumDRNN = embdSize;   // 一步正好一个词向量
    par.onceAllOutputNumDRNN = embdSize;  // 输出也正好一个词向量 → sliceNum = 1，不切片
    par.outRnnHSizeDRNN = embdSize;       // 和词向量同宽只是省心，两者没有任何关系
    par.allRnnHSizeDRNN = embdSize;

    if (const size_t code = encap::InitLLM(llm, par, corpus, numMerges)) {
        return code;
    }

    // ---------------------------------------------------------------------
    // 2. 词嵌入：Skip-gram 学共现（中心词预测上下文，正样本拉近、负采样推远）
    //    学完这张表就冻住 —— DRNN 训练和生成时都只查表，不再更新它
    // ---------------------------------------------------------------------
    if (const size_t code = encap::TrainEmbdCheck(llm, corpus, embdWindow, embdEpochs, embdNeg, embdLr)) {
        return code;
    }

    // ---------------------------------------------------------------------
    // 3. 滑窗样本：语料 → 一条 token 流（句间插 <END>）→ 每个位置当一次「下一个」
    // ---------------------------------------------------------------------
    std::vector<size_t> stream;
    std::vector<encap::WindowSample> samples;
    std::vector<size_t> candidates;
    if (const size_t code = encap::MakeStream(llm, corpus, window, stream, samples, candidates)) {
        return code;
    }

    // ---------------------------------------------------------------------
    // 4. 训练 DRNN：一条 window 步的序列进，预测下一个 token 的词向量
    //    （只有最后一片吃真目标，前面的片靠实际输出 + h 目标回填）
    // ---------------------------------------------------------------------
    if (const size_t code = encap::TrainDRNN(llm, samples, epochs, wlr, blr)) {
        return code;
    }

    // ---------------------------------------------------------------------
    // 5. 生成：每步把「提示词 + 已生成」整段喂进去（h 在整段上连续），
    //    取最后输出做最近邻还原成 token，再接上
    //
    //    语料只有 7 句、滑窗样本只有几十个，生成大概率还是收敛进某个循环 ——
    //    这一步是为了把整条链路跑通，不是要它写出文章。
    // ---------------------------------------------------------------------
    const std::vector<size_t> generated = llm.GenerateStream(prompt, genLen, candidates);
    if (generated.empty()) {
        std::cout << "提示词编码失败: " << prompt << "\n";
        return 3;
    }
    const size_t head = llm.data.tokenizer.Encode(prompt).size();  // 前面这几格是提示词，后面才是生成的

    std::cout << "提示词: " << prompt << "\n";
    std::cout << "生成  : " << llm.data.tokenizer.Decode(generated)
              << "（其中新生成 " << (generated.size() - head) << " 个 token）\n";

    // ---------------------------------------------------------------------
    // 6. 存 / 取往返：存下来 → 读进另一个 LLM → 同一输入的输出必须逐位相同
    // ---------------------------------------------------------------------
    const std::string modelPath = encap::ModelPath(argc > 1 ? argv[1] : nullptr);
    const std::vector<float> probe = llm.data.vec.Get(generated.front());
    if (const size_t code = encap::RoundTrip(llm, modelPath, probe)) {
        return code;
    }
    return 0;
}
