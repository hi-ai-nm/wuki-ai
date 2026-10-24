#include "includeOne.hpp"  // llm.hpp（LLM 类）+ encap.hpp（流水线各阶段函数）
#include "fskio.hpp"       // comm：FlaskIO（把「提示词 → 文本」挂到 HTTP 上，供 Flask 端调用）
#include <cstring>
#include <string>
#include <vector>

constexpr std::string WUKI_VERSION = "V.F1B.DN";

/*
 * wuki-ai 主程序：一条最小的「自造 LLM」流水线
 *
 *   BBPE 分词  →  Skip-gram 词嵌入（学共现）  →  预训练 allRNN（正序读、倒序挂目标）
 *              →  双层 RNN 联合训练（allRNN 直挂真目标）预测下一个词
 *              →  生成的向量用最近邻还原成 token  →  解码回文本
 *
 * 各阶段的实现和原理说明都在 llm.hpp / encap.hpp / llm 子目录各头的注释里，这里只排顺序：
 *   Init（词表 + 词嵌入 + DRNN）→ Study（Skip-gram + 滑窗样本 + 预训练 + 联合训练）→ Run（生成）。
 * 配置全在这三个函数的带默认值的参数上（默认值就是原先写死在这份 main 里的那套）。
 *
 * 用法：无参数 / -R / --retrain 都是重训一遍；-S / --serve 起服务（和 Flask 通信）；
 *       -V / --version 版本；-H / --help 帮助。
 *       -C / --corpus <文件> 换成真语料（每行一句，句尾自带 </end>）；
 *       -N / --lines <行数>  只读前 n 行（真语料有 110 万行，先截一段）。
 *
 * 返回值（每一条都是不同的退出码，脚本里一眼能看出停在哪一步）：
 *   0 成功 | 1 词嵌入初始化 / 训练失败（含语料文件读不出来） | 2 DRNN 初始化失败
 *   3 语料 / 提示词编码失败 | 4 训练失败（含 allRNN 预训练） | 5 权重文件写不出去
 *   8 服务初始化失败 / 端口没抢到
 */
int main(int argc, char *argv[]) {
    // cout 默认是块缓冲：被重定向 / 在容器里（docker logs）时，训练进度会一直攒着不吐出来，
    // 看着像卡死。unitbuf = 每次 << 都 flush —— 这套流水线只在每轮 / 每 N 轮打印一次，不心疼
    std::cout.setf(std::ios::unitbuf);

    // ---------------- 参数 ----------------
    // 原来是一个 argc 一个分支地比字符串，加了 -C / -N 之后组合有十几种，改成扫一遍参数表
    std::string mode;               // "retrain" / "serve" / "version" / "help"（空 = 参数错误）
    std::string corpusPath;         // 空 = 用内置的玩具语料（7 句 "hello world"）
    size_t maxLines = 0;            // 0 = 全读
    size_t port = 8689;             // -S 的端口（docker-compose 映射出去的就是它）
    size_t EPOCHS = 3;         // 联合训练轮数（每轮 ≈ 样本数 × 0.19 ms）
    size_t EMBD_EPOCHS = 3;    // Skip-gram 轮数
    size_t PRE_EPOCHS = 2;     // allRNN 预训练轮数
    bool serveRetrain = false;  // -SR = -S -R

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-R" || a == "--retrain") {
            mode = "retrain";
        } else if (a == "-S" || a == "--serve") {
            mode = "serve";
        } else if (a == "-SR" || a == "--serveRetrain") {
            mode = "serve";
            serveRetrain = true;
        } else if (a == "-V" || a == "--version") {
            mode = "version";
        } else if (a == "-H" || a == "--help") {
            mode = "help";
        } else if ((a == "-C" || a == "--corpus") && i + 1 < argc) {
            corpusPath = argv[++i];
        } else if ((a == "-N" || a == "--lines") && i + 1 < argc) {
            try {
                maxLines = std::stoul(argv[++i]);
            } catch (...) {
                std::cout << "行数不合法: " << argv[i] << "（要一个正整数）\n";
                return 1;
            }
        } else if ((a == "-Esg" || a == "--epochsSkipgram") && i + 1 < argc) {
            try {
                EMBD_EPOCHS = std::stoul(argv[++i]);
            } catch (...) {
                std::cout << "轮数不合法: " << argv[i] << "（要一个正整数）\n";
                return 1;
            }
        } else if ((a == "-EPre" || a == "--epochsPre") && i + 1 < argc) {
            try {
                PRE_EPOCHS = std::stoul(argv[++i]);
            } catch (...) {
                std::cout << "轮数不合法: " << argv[i] << "（要一个正整数）\n";
                return 1;
            }
        } else if ((a == "-EAll" || a == "--epochsAll") && i + 1 < argc) {
            try {
                EPOCHS = std::stoul(argv[++i]);
            } catch (...) {
                std::cout << "轮数不合法: " << argv[i] << "（要一个正整数）\n";
                return 1;
            }
        } else {  // 孤立的数字 = -S 的端口（保持原来 "-S 9000" 的用法）
            try {
                port = std::stoul(a);
            } catch (...) {
                std::cout << "参数错误: " << a << "，请使用 -H 或 --help 查看帮助。" << std::endl;
                return 1;
            }
            if (port == 0 || port > 65535) {
                std::cout << "端口不合法: " << a << "（要 1..65535）" << std::endl;
                return 1;
            }
        }
    }

    if (mode == "version") {
        std::cout << "Wuki-ai " << WUKI_VERSION << std::endl;
        return 0;
    }

    if (mode == "help") {
        std::cout << // 版本信息
            "Wuki-ai " << WUKI_VERSION << R"(
更新日期：2026-10-24
* 此为 DevNode （开发节点）版，非正式版，切勿用于生产环境。
* 本项目为个人兴趣项目，不保证长期维护。)"
        << std::endl;

        std::cout << // 命令用法
            "重新训练：" << argv[0] << " [-R|--retrain] [-C 语料文件] [-N 行数]\n"
            "启动服务：" << argv[0] << " [-S|--serve] [端口] [-C 语料文件] [-N 行数]\n"
            "显示版本：" << argv[0] << " [-V|--version]\n"
            "指定语料：" << argv[0] << " [-C 语料文件]  # 想指定训多少行可以用 -N xx\n"
            "\n"
            "  -C 不给就用内置玩具语料（7 句 hello world，秒级训练）；\n"
            "     给了就读文件，每行一句、句尾自带 </end>（和 src/main/llm/corpus.txt 同格式），\n"
            "     结束符换成 </end>、参数自动切到真语料那一套（更大的词表 / 更少的轮数）。\n"
            "  -N 只读前 n 行。真语料 110 万行，训练开销是 epochs × 样本数，先截 5000 行试最稳。\n"
        << std::endl;

        return 0;
    }

    // 没给模式就重训一遍（上面注释和 help 里的 [-R|--retrain] 都是这个意思：
    // 方括号 = 可省）。-C / -N 只是修饰符，自己不构成模式 ——
    // 以前这里把空 mode 当参数错误，于是 `./wuki-ai -C corpus.txt` 和
    // dockerfile 里那句 CMD ["./wuki-ai", "-C", "corpus.txt"] 都直接退 1。
    if (mode.empty()) {
        mode = "retrain";
    }

    // ---------------- 语料 ----------------
    // 玩具语料（没给 -C 时用）：内容就是 Template::Init / Study 里那套默认值。
    // 走的是同一个调用出口，所以下面三个阶段的参数只有一份逻辑，不用"有语料 / 没语料"两条分支
    static const std::vector<std::string> TOY_CORPUS(7, std::string("hello world"));
    std::vector<std::string> realCorpus;
    const bool useFile = !corpusPath.empty();

    if (useFile) {
        if (const size_t code = encap::LoadCorpus(corpusPath, realCorpus, maxLines)) {
            std::cout << "语料读不出来: " << corpusPath
                      << "（" << (code == 1 ? "文件打不开" : "一行都没读到") << "）\n";
            return 1;
        }
        std::cout << "语料: " << corpusPath << " 读了 " << realCorpus.size() << " 行"
                  << (maxLines == 0 ? "（全读）" : "（截断）") << "\n";
    }
    const std::vector<std::string>& corpus = useFile ? realCorpus : TOY_CORPUS;

    // 两套参数。玩具那套直接走 Template::Init / Study 的全默认值（和以前逐字一致）；
    // 真语料这套按 5000 行样本量的：词表更大、轮数更少 —— Skip-gram 3500 轮在 30 万 token 上不现实。
    const size_t REAL_NUM_MERGES = 1500;  // BBPE 合并轮数（词表 ≈ 256 + 2 + 1500）
    const size_t REAL_EMBD_SIZE = 32;     // 词向量维度，整套 DRNN 尺寸的锚
    const float REAL_ALL_SCALE = 0.1f;    // 预训练过就把 allRNN 调小，别让联合训练全速改写它
    const size_t REAL_WINDOW = 8;         // 滑窗宽度（上下文几个 token）
    // 结束符的串：真语料的每一行自己就以 </end> 结尾，所以不再额外插（insertEnd = false）
    const std::string endTOKEN = useFile ? "</end>" : "@__SYSTEM<END>__";
    const bool insertEnd = !useFile;

    auto initModel = [&](LLM& model, encap::Template& tpl) -> int {
        if (!useFile) {
            return tpl.Init(model);  // 玩具：全默认（8 维 / {40,16,32} / h 30,60 那套）
        }
        // 结构参数（outRnnNeur / allRnnNeur / 各 hSize）全传 0 / {} = 按 embdSize 推导，
        // 换 embdSize 时不用手改那六个写死的数（见 encap.hpp 的 Init 注释）
        return tpl.Init(model, corpus, REAL_NUM_MERGES, REAL_EMBD_SIZE, REAL_ALL_SCALE,
                        { "</end>", "<UnitEnd>" }, -1.0f, 1.0f, {}, {}, 0, 0, 0, 0, "scth");
    };
    auto studyModel = [&](LLM& model, encap::Template& tpl) -> int {
        if (!useFile) {
            return tpl.Study(model);  // 玩具：全默认（Skip-gram 3500 轮 / 预训练 3000 / 联合 17000）
        }
        return tpl.Study(model, corpus,
                         3, EMBD_EPOCHS, 5, 0.05f,        // Skip-gram：window 3 / 轮数 / 负样本 / lr
                         8, PRE_EPOCHS, 0.001f, 0.00025f, // allRNN 预训练：窗口 / 轮数 / wlr / blr
                         REAL_WINDOW, EPOCHS, 0.002f, 0.0005f,
                         endTOKEN, insertEnd);
    };

    if (mode == "retrain") {
        // 三步走：Init 立词表 / 词嵌入 / DRNN → Study 训 Skip-gram + 预训练 + 联合训练 → Run 生成
        LLM model;
        encap::Template templateInstance; // 需实例化
        if (const int code = initModel(model, templateInstance)) {
            return code;  // 1 词嵌入初始化失败 / 2 DRNN 初始化失败
        }
        if (const int code = studyModel(model, templateInstance)) {
            return code;  // 1 词嵌入训练失败 / 3 编码失败或凑不出样本 / 4 训练失败
        }
        if (useFile) {
            if (const int code = templateInstance.Run(model, "一 只", corpus, 24, 0.05f, nullptr, endTOKEN)) {
                return code;  // 3 扫不出候选 / 提示词编码失败
            }
        } else if (const int code = templateInstance.Run(model)) {  // 玩具：默认提示词 hello
            return code;
        }
        return 0;
    }

    // mode == "serve"
    {
        // 1) 备模型：词表 / 词嵌入 / DRNN 结构（顺序不能反，原因见 encap.hpp 的 Init）
        LLM model;
        encap::Template templateInstance; // 需实例化
        if (const int code = initModel(model, templateInstance)) {
            return code;  // 1 词嵌入初始化失败 / 2 DRNN 初始化失败
        }

        // 2) 权重：文件里只有 DRNN，词表和词嵌入靠上面那次 Init 用同一份语料重建
        //    （见 DRNNSL.hpp 的格式说明），所以有文件就直接读（快），
        //    没有就训一遍再存下来 —— 下次启动就只读不训了
        const std::string modelPath = encap::ModelPath();
        if (serveRetrain == false) {
            if (model.Load(modelPath)) {
                std::cout << "已加载权重: " << modelPath << "\n";
            } else {
                std::cout << "读不到可用权重（" << model.errBlgig << "），先训一遍: " << modelPath << "\n";
                if (const int code = studyModel(model, templateInstance)) {
                    return code;  // 1 词嵌入训练失败 / 3 编码失败或凑不出样本 / 4 训练失败
                }
                if (!model.Save(modelPath)) {
                    std::cout << "Save 失败: " << model.errCode << " (" << model.errBlgig << ")\n";
                    return 5;
                }
                std::cout << "权重已存: " << modelPath << "\n";
            }
        } else {
            std::cout << "强制重新训练权重: " << std::endl;
            
            if (const int code = studyModel(model, templateInstance)) {
                return code;  // 1 词嵌入训练失败 / 3 编码失败或凑不出样本 / 4 训练失败
            }
            if (!model.Save(modelPath)) {
                std::cout << "Save 失败: " << model.errCode << " (" << model.errBlgig << ")\n";
                return 5;
            }
        }

        // 3) 候选 token：只跟 (语料, 词表) 有关，和提示词无关 —— 在这里算一次就够了。
        //    之前是每个请求都在 ReplyText 里现扫一遍整份语料，55 万行要一两分钟，
        //    一次 POST 得等 2~4 分钟。扫一次的成本挪到启动阶段（和建词表一起）。
        std::vector<size_t> candidates;
        size_t eos = 0;
        if (const size_t code = encap::Candidates(model, corpus, candidates, &eos, endTOKEN)) {
            std::cout << "扫不出候选 token: " << code
                      << "（3 = 结束符不在词表里 / 某句编码不出来）\n";
            return 3;
        }
        std::cout << "候选 token: " << candidates.size() << " 个（算一次，后面每次请求复用）\n";

        // 4) 起服务：Flask 端 POST 提示词，回生成的文本（协议见 src/comm/fskio.hpp）
        wuki::comm::FlaskIO io;
        if (const int code = io.Init([&model, &candidates, eos](const std::string& prompt) {
                return encap::ReplyText(model, prompt, candidates, eos, 24, 0.05f);
            }, "/post", port)) {
            std::cout << "服务初始化失败: " << code
                      << "（1 已经初始化过 / 2 回调为空 / 3 路径为空 / 4 端口不合法）\n";
            return 8;
        }
        std::cout << "Flask 端往 http://127.0.0.1:" << port << "/post POST 提示词即可\n";
        if (!io.Start()) {
            return 8;  // 监听失败（Start 里已经打印了原因）
        }
        return 0;
    }
}
