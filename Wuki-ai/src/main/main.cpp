#include "llm.cpp"  // framework（Tools.hpp + includeMain.hpp）+ architecture + DRNNSL + bbpe + wet

#include <iostream>  // std::cout
#include <string>    // std::string
#include <utility>   // std::pair
#include <vector>    // std::vector

/*
 * wuki-ai 主程序：一条最小的「自造 LLM」流水线
 *
 *   BBPE 分词  →  词嵌入  →  双层 RNN（Arch::DRNN）预测下一个词的词向量
 *              →  生成的向量用最近邻还原成 token  →  解码回文本  →  存 / 取往返校验
 *
 * 用法：
 *   wuki-ai [模型文件]
 *     不给参数：模型存在 exe 旁边（…/Bin/drnn.model.json），从哪个工作目录启动都一样
 *     给了参数：存在指定路径 —— 容器里跑的时候配 -v 挂载，才能把模型留到 host：
 *               docker run --rm -v "$PWD/Bin:/out" docker_run /out/drnn.model.json
 *               （镜像用仓库根目录的 dockerfile 构建：docker build -f dockerfile -t docker_run .）
 *
 * 尺寸为什么这么定（全都围着词向量维度转）：
 *   onceAllInputNum  = onceAllOutputNum = 词向量维度
 *       —— 一次喂进一个 token 的词向量（一步），DRNN 的输出也正好是词向量那么宽
 *   outRnnNeur = { 词向量维度, 隐层, 词向量维度 }
 *       —— sliceWidth = 词向量维度 → sliceNum = onceAllOutputNum / sliceWidth = 1（不切片），
 *          于是「下一 token 的词向量」可以直接当目标用，用不着再拆 / 拼
 *   hSize 和词向量维度取一样大只是为了省心，两者没有任何关系
 *
 * 返回值（每一条都是不同的退出码，脚本里一眼能看出停在哪一步）：
 *   0 成功 | 1 词嵌入初始化失败 | 2 Init 失败 | 3 语料/提示词编码失败
 *   4 训练失败 | 5 Save 失败 | 6 Load 失败 | 7 存 / 取往返结果不一致 | 8 参数太多
 */

int main(int argc, char** argv) {
    if (argc > 2) {  // 只认一个可选参数
        std::cout << "用法: " << argv[0] << " [模型文件路径]\n";
        return 8;
    }

    // ---------------------------------------------------------------------
    // 0. 参数
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

    const size_t embdSize = 8;     // 词向量维度
    const size_t epochs = 12000;   // 训练轮数（每轮把整份训练对过一遍）
    const size_t genLen = 24;      // 生成时最多再接多少个 token
    const std::string prompt = "hello";
    const float wlr = 0.002f;      // 权重学习率
    const float blr = 0.0005f;     // 偏置学习率（习惯上比权重小一个量级）
                                   // 这两个值是量出来的：同一套结构跑多次（词向量和权重
                                   // 都是随机初始化的），0.01 / 0.001 时 loss 有时不降，
                                   // 0.002 / 0.0005 每次都降、末值也最低

    // ---------------------------------------------------------------------
    // 1. BBPE 分词器：先立模型（256 个字节 + 1 个起始 token），再用语料学 merge 规则
    //    Train 内部会自己 Init 一次，这里显式调是为了「语料为空」时词表也是完整的
    // ---------------------------------------------------------------------
    BBpe::BBPE tokenizer;
    tokenizer.Init();
    tokenizer.Train(tokenizer.ConvertCorpus(corpus));

    const size_t vocabSize = tokenizer.model.tokens.size();
    std::cout << "词表大小: " << vocabSize << "（含 256 字节 + 1 起始 token + BPE merge）\n";

    // ---------------------------------------------------------------------
    // 2. 词嵌入：每个 token id 一张随机向量
    //
    //    注意这张表是**冻住的**：WorEmbd 没有训练接口，DRNN 也不会回写它，整个训练里
    //    只有两个 RNN 的 w / b 在动。所以这里学的其实是「一个随机向量 → 下一个 token
    //    的那个随机向量」这个映射，而不是「词的意思」（数据量也远不够学出语义）。
    //    真要让嵌入一起学，得给 WorEmbd 加一个「按目标向量更新」的接口。
    // ---------------------------------------------------------------------
    WorEmbd embdInput;
    if (embdInput.Init(vocabSize, embdSize) != 0) {
        std::cout << "词嵌入初始化失败：tokenSize / outputSize 不能为 0\n";
        return 1;
    }
    std::cout << "词向量: " << embdInput.tokenSize() << " × " << embdInput.embdSize()
              << "（随机初始化，训练中不更新）\n";

    // ---------------------------------------------------------------------
    // 3. 双层 RNN：allRNN 先预测「后面所有的输出」，outRNN 再把每一片变成真正的输出
    //    输入层宽 = onceAllInputNum + allRnnHSize，所以调用方只用关心「一个词向量」那么宽
    // ---------------------------------------------------------------------
    const std::vector<size_t> outRnnNeur = { embdSize, embdSize * 2, embdSize };
    const std::vector<size_t> allRnnNeur = { embdSize * 2 };

    Arch::DRNN model;
    const size_t initCode = model.Init(outRnnNeur, allRnnNeur, embdSize, embdSize, embdSize, embdSize);
    std::cout << "Init: " << initCode << " (" << model.errorBlgig << ")\n";
    if (initCode != 0) {
        return 2;  // 结构参数不合法（比如 onceAllOutputNum 不能被 outRnnNeur[0] 整除）
    }

    // ---------------------------------------------------------------------
    // 4. 训练对：语料 → token id 序列 → (当前 token, 下一个 token)
    //    注意 BBPE 只对「词」学 merge，词之间的空格还是走字节 token，所以空格本身
    //    也是一个 token（(hello, 空格)、(空格, world) 都会被学出来）
    // ---------------------------------------------------------------------
    std::vector<std::pair<size_t, size_t>> pairs;
    std::vector<size_t> candidates;  // 语料里真正出现过的 token（生成时只在这里面挑）
    std::vector<bool> seen(vocabSize, false);

    for (const std::string& sentence : corpus) {
        const std::vector<size_t> ids = tokenizer.Encode(sentence);
        if (ids.empty()) {  // Encode 编不出来就返回空（不会给半截）
            std::cout << "编码失败: " << sentence << "\n";
            return 3;
        }
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            pairs.emplace_back(ids[i], ids[i + 1]);
        }
        for (const size_t id : ids) {
            if (!seen[id]) {  // 按出现顺序记一遍，去重
                seen[id] = true;
                candidates.push_back(id);
            }
        }
    }

    std::cout << "训练对: " << pairs.size() << "，语料覆盖 token: " << candidates.size() << "\n";
    if (pairs.empty()) {  // 语料短到没有任何相邻 token 对，训不出东西
        std::cout << "语料太短：编码后没有任何相邻 token 对\n";
        return 3;
    }

    // ---------------------------------------------------------------------
    // 5. 训练：目标是「下一个 token 的词向量」，一次喂一个 token（一步，不切片）
    //
    //    这里只能一对一对地训：DRNN::Train 一次吃一整条序列，但只有最后一步有外部目标
    //    （前面的步拿自己的实际输出当目标）。要把「当前 token → 下一个 token」这件事学出来，
    //    一个训练对就是一步序列。
    // ---------------------------------------------------------------------
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total = 0.0f;
        for (const auto& [cur, next] : pairs) {
            auto [loss, code] = model.Train({ embdInput.Get(cur) }, embdInput.Get(next), wlr, blr);
            if (code != 0) {
                std::cout << "Train error: " << code << " (" << model.errorBlgig << ")\n";
                return 4;
            }
            if (!loss.empty()) {
                total += loss[0];  // 这一步的 MSE
            }
        }

        if (epoch % 2000 == 0 || epoch + 1 == epochs) {  // 隔一会儿看一眼误差有没有在降
            std::cout << "epoch " << epoch << " 平均 loss "
                      << (total / static_cast<float>(pairs.size())) << "\n";
        }
    }

    // ---------------------------------------------------------------------
    // 6. 生成：喂进当前 token 的词向量 → 输出向量 → 最近邻还原成 token → 再喂回去
    //
    //    最近邻只在 candidates（语料里出现过的 token）里找：全词表找的话，256 个字节
    //    token 也各有一张随机向量，很容易挑到语料里从没出现过的控制字节（生成出来是乱码）。
    //    词向量是冻住的随机向量、训练对也只有几十个，所以生成结果会很快收敛到一个循环
    //    （学得最好的那个转移）—— 这一步是为了把整条链路跑通，不是要它写出文章。
    // ---------------------------------------------------------------------
    std::vector<size_t> generated = tokenizer.Encode(prompt);
    if (generated.empty()) {
        std::cout << "提示词编码失败: " << prompt << "\n";
        return 3;
    }
    const size_t head = generated.size();  // 前面这几格是提示词，后面才是生成的

    for (size_t i = 0; i < genLen; i++) {
        auto [out, code] = model.Run({ embdInput.Get(generated.back()) });
        if (code != 0) {
            std::cout << "Run error: " << code << " (" << model.errorBlgig << ")\n";
            break;  // 生成失败就把已经生成的部分打出来
        }

        const size_t next = embdInput.Nearest(out, candidates);
        if (next >= embdInput.tokenSize()) {  // 还原不出来（候选为空 / 维度对不上）
            break;
        }
        generated.push_back(next);
    }

    std::cout << "提示词: " << prompt << "\n";
    std::cout << "生成  : " << tokenizer.Decode(generated)
              << "（其中新生成 " << (generated.size() - head) << " 个 token）\n";

    // ---------------------------------------------------------------------
    // 7. 存 / 取往返：存下来 → 读进另一个模型 → 同一输入的输出必须逐位相同
    //
    //    路径默认放在 exe 旁边（exeDir 取不到就退化成当前目录），从哪个工作目录启动都一样；
    //    给了命令行参数就用参数 —— 容器里跑的时候，容器层会随 --rm 一起丢掉，只有写到
    //    挂载进来的目录，训练好的模型才留得住。
    // ---------------------------------------------------------------------
    std::string modelPath;
    if (argc > 1) {
        modelPath = argv[1];
    } else {
        const std::string binDir = wuki::exeDir();
        modelPath = (binDir.empty() ? std::string(".") : binDir) + "/drnn.model.json";
    }

    Arch::DRNNSL saveLoad;
    if (!saveLoad.Save(model, modelPath)) {
        std::cout << "Save 失败: " << saveLoad.errorCode << " (" << saveLoad.errorBlgig << ")\n";
        return 5;
    }
    std::cout << "Save ok: " << modelPath << "\n";

    Arch::DRNN loaded;  // 新模型，除了下面这次 Load 之外没碰过训练
    if (!saveLoad.Load(loaded, modelPath)) {
        std::cout << "Load 失败: " << saveLoad.errorCode << " (" << saveLoad.errorBlgig << ")\n";
        return 6;
    }
    std::cout << "Load ok\n";

    // 同一份权重、同一套前向 → 应当逐位相同（json 里的 float 是「能精确还原」的写法）
    const std::vector<float> probe = embdInput.Get(generated.front());
    auto [result, error] = model.Run({ probe });
    auto [result2, error2] = loaded.Run({ probe });

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
