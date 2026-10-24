#pragma once
#ifndef NN_CPP
#define NN_CPP

#include "nn.hpp"
#include <algorithm>  // std::copy
#include <cmath>
#include <fstream>  // std::ifstream / std::ofstream
#include <utility>  // std::move
#include "../external/json.hpp"
#include "../optimize/simd.hpp"  // Simd::Runtime：前向点积 / 反向 axpy

/*
 * 归属 - run 编号 - 0x01：未初始化网络就运行了
 * 归属 - run 编号 - 0x02：输入大小和 neur[0] 对不上
 *
 * 归属 - study 编号 - 0x01：未初始化网络就训练了
 * 归属 - study 编号 - 0x02：缓存和当前结构对不上（set_neur 之后没重新 Run）
 * 归属 - study 编号 - 0x03：层数过少
 * 归属 - study 编号 - 0x04：goal 的宽度和输出层对不上
 *
 * 归属 - save 编号 - 0x01：未初始化网络（或 w / b 和 neur 对不上）就保存了
 * 归属 - save 编号 - 0x02：文件打不开 / 写不进去
 *
 * 归属 - load 编号 - 0x01：文件打不开 / 不是合法 JSON
 * 归属 - load 编号 - 0x02：不是本程序保存的 nn 模型（keys / rulesVersion 对不上）
 * 归属 - load 编号 - 0x03：content 缺字段，或 w / b 的尺寸和 neur 对不上
 */

namespace nn {
    // SIMD 内核的短别名（放在 namespace 里：includeMain.cpp 把 nn.cpp / rnn.cpp
    // 收进同一个编译单元，两个文件各自的全局别名会撞）
    namespace opt = Simd::Runtime;
    namespace study = Simd::StudyTime;

    // JSON 别名放在 namespace 里：includeMain.cpp 先 include nn.cpp 再 include rnn.cpp，
    // 写在全局作用域的话两个文件会重定义（rnn.cpp 里也有一个同名别名）。
    using json = nlohmann::json;
    // 激活函数名（原为 _ActFun_* 宏；宏不受 namespace 约束，改为 namespace 内常量）
    static constexpr const char* ActFun_SelfCreTanh = "scth";
    static constexpr const char* ActFun_ReLU = "relu";
    static constexpr const char* ActFun_SiLU = "silu";
    static constexpr const char* Alternative_ActFun_Sigmoid = "alt-sigmoid";

    /**
     * 激活函数在 z 点的导数 a'(z)（Study 里把误差换成 δ 用，和 rnn::ActDeriv 同一套公式）
     *
     * 乘上它之后，饱和区的更新自动趋 0 —— 那里本来就没有可学的自由度，
     * 原来不乘等于硬推，网络会越推越深、卡死在饱和面上。
     *
     * @param actFun 激活函数名（和 Run 里的分支同一套名字）
     * @param z      该神经元的加权和（insPar.z 里存着）
     * @param a      该神经元的激活值（insPar.a 里存着）
     * @return a'(z)；未知名字按 ReLU 处理（和 Run 的默认分支一致）
     */
    inline float ActDeriv(const std::string& actFun, const float z, const float a) {
        // 公式只在 Simd::StudyTime::deriv_scalar 里维护一份（Study 的批量 δ 也用它），
        // 这里保留成对外的小工具，避免两套导数漂移
        return study::deriv_scalar(study::ParseActKind(actFun), z, a);
    }

    struct Pars {  // 参数
        std::vector<float> w;
        std::vector<float> b;

        std::string actFun = ActFun_SiLU;  // 激活函数

        std::vector<size_t> neur;  // 每层神经元数量
        bool notNeurUnde = false;  // 是否已定义 neur

        size_t errorCode = 0x00;               // 错误码（默认无错误）
        std::string errorBlgig = "not error";  // 错误归属（errorBlonging）
    };

    struct InsPar {  // 瞬时参数
        std::vector<float> a;
        std::vector<float> z;
        std::vector<float> x;  // 最近一次 Run 的输入（Study 里更新第 1 层的 w 要用它）
    };

    class NeuNet {
      public:
        Pars par;
        InsPar insPar;

        // Run 函数
        /**
 * 执行神经网络的前向传播
 * @param input 输入向量
 * @return 神经网络的输出向量
 */
        std::vector<float> Run(const std::vector<float>& input) {
            // 检查神经网络是否已正确初始化
            // 处理未定义 neur 的情况
            if (par.notNeurUnde == false) {
                par.errorCode = 0x01;         // 设置错误代码
                par.errorBlgig = "run";       // 设置错误发生位置
                return std::vector<float>{};  // 返回空列表（需 C++11 ）
            }

            // 输入大小必须和输入层对得上，否则下面读 oldActVal 就会越界
            if (input.size() != par.neur[0]) {
                par.errorCode = 0x02;
                par.errorBlgig = "run";
                return std::vector<float>{};
            }

            // 清理瞬时参数
            insPar.a.clear();
            insPar.z.clear();
            insPar.x = input;  // 记下这一轮的输入（Study 里第 1 层要拿它当「前一层的输出」）

            // 初始化
            std::vector<float> oldActVal = input;
            size_t wIndex = 0;
            size_t bIndex = 0;
            size_t actBase = 0;  // 当前层在 insPar.z / insPar.a 里的起点
            const study::ActKind actKind = study::ParseActKind(par.actFun);

            // 运行
            for (size_t nol = 1; nol < par.neur.size(); nol++) {
                const size_t inCount = par.neur[nol - 1];  // 本层每个神经元的输入数
                const size_t outCount = par.neur[nol];

                // 先整层算 z（每个神经元一行连续权重，一个 SIMD 点积），
                // 再一次性批量激活：tanh / exp 是逐神经元的超越函数，
                // 一次处理 width 个才用得上向量数学；同时 actFun 只在入口解析一次。
                insPar.z.resize(actBase + outCount);
                insPar.a.resize(actBase + outCount);
                float* zPtr = insPar.z.data() + actBase;
                float* aPtr = insPar.a.data() + actBase;
                for (size_t cnn = 0; cnn < outCount; cnn++) {
                    const float sum = opt::dot(par.w.data() + wIndex, oldActVal.data(), inCount);
                    wIndex += inCount;
                    zPtr[cnn] = sum + par.b[bIndex];  // z = 加权求和 + 偏置
                    bIndex++;
                }
                study::activate(actKind, zPtr, aPtr, outCount);

                // 下一层的输入就是本层的激活值（和原来 push_back 的顺序完全一致）
                oldActVal.assign(aPtr, aPtr + outCount);
                actBase += outCount;
            }

            return oldActVal;
        }

        // Init
        size_t Init(std::vector<size_t>& neur) {
            // 初始化随机数引擎
            std::random_device rd;
            std::mt19937 gen(rd());

            par.w.resize(InteUtiFun::SumOfProducts(neur));  // 预留 w 空间
            // 预留 b 空间：只有 layer 1 ~ 最后一层有偏置（输入层没有），
            // 原来多分配了 neur[0] 个：不越界，但和 Run / Study 的下标用法对不上
            par.b.resize(InteUtiFun::SumUp(neur) - neur[0]);

            // 初始化 WBC （权重疲劳）

#ifdef _WBC_On
            size_t WBC = _WBC_Init;
#else
            size_t numberOfChenges = 20;  // 默认 20
            InteUtiFun::ThereAreLessThan(neur, numberOfChenges);
            size_t WBC = numberOfChenges;
#endif

            // 循环遍历 w
            size_t wIndex = 0;
            for (size_t nol = 1; nol < par.neur.size(); nol++) {
                for (size_t cnn = 0; cnn < par.neur[nol]; cnn++) {
                    size_t connectCount = 0;  // 当前神经元的连接数
                    for (size_t nonitpxl = 0; nonitpxl < par.neur[nol - 1]; nonitpxl++) {
                        float x;
#ifdef _WBC_On
                        if (connectCount >= WBC) {
                            x = 0.08f;  // 超过限制，使用权重 0.08
                        } else {
                            x = InteUtiFun::RandomRange(gen, (-0.5f), 0.5f);  // 未超过，使用随机权重
                        }
#else
                        x = InteUtiFun::RandomRange(gen, (-0.5f), 0.5f);
#endif
                        par.w[wIndex] = x;
                        wIndex++;
                        connectCount++;
                    }
                }
            }

            // 初始化偏置 b
            for (auto& b : par.b) {
                b = InteUtiFun::RandomRange(gen, (-0.3f), 0.3f);
            }
            return 0x00;
        }

        // 设置层数
        inline void set_neur(const std::vector<size_t>& neur, bool autoInit = true) {
            par.neur = neur;
            if (autoInit) {
                Init(par.neur);
            }
            par.notNeurUnde = true;
        }

        inline void set_neur(std::initializer_list<size_t> neur, bool autoInit = true) {
            par.neur = neur;
            if (autoInit) {
                Init(par.neur);
            }
            par.notNeurUnde = true;
        }

        // 设置激活函数
        inline void set_ActFun(std::string content) {
            if (content != ActFun_ReLU && content != ActFun_SiLU && content != Alternative_ActFun_Sigmoid) {
                par.errorCode = 0x01;
                par.errorBlgig = "SetActFun";
                return;
            }

            par.actFun = content;
        }

        // 对比学习（自创）
        // 核心公式：\text{目标激活值}_{\text{前一层}} = N \times \frac{w}{\sum |w|} \times \text{目标激活值}_{\text{当前层}}（Markdown）
        // 流程：① 输出层铺开目标 ② 逐层往回推目标 ③ 目标统一减实际激活值得到误差 ④ 用误差改 w 和 b
        // （和 rnn::Study 是同一套：中途推的是「目标」，最后才减激活值）
        float Study(const std::vector<float>& goal, float wlr = 0.01f, float blr = 0.001f) {
            // 处理未定义 neur 的情况
            if (par.notNeurUnde == false) {
                par.errorCode = 0x01;
                par.errorBlgig = "study";
                return 0.0f;
            }

            size_t numLayers = par.neur.size();
            if (numLayers < 2) {  // 处理层数过少的情况
                par.errorCode = 0x03;
                par.errorBlgig = "study";
                return 0.0f;
            }

            // 误差数组的长度：只存 layer 1 ~ 最后一层
            size_t totalNonInput = 0;
            for (size_t l = 1; l < numLayers; l++) {
                totalNonInput += par.neur[l];
            }

            // 缓存必须和当前结构对得上：set_neur 换了结构却没重新 Run 的话，
            // insPar 里还是旧尺寸的数据，按新结构读会越界
            if (insPar.a.size() != totalNonInput || insPar.z.size() != totalNonInput ||
                insPar.x.size() != par.neur[0]) {
                par.errorCode = 0x02;
                par.errorBlgig = "study";
                return 0.0f;
            }

            // goal 的宽度必须和输出层对得上，否则下面按 goal[i] 写 error 会越界
            if (goal.size() != par.neur[numLayers - 1]) {
                par.errorCode = 0x04;
                par.errorBlgig = "study";
                return 0.0f;
            }

            // 计算每层在 insPar.a / insPar.z 中的起始索引（不含输入层）
            // layer 1 从 0 开始，layer l 从 sum(neur[1..l-1]) 开始
            std::vector<size_t> aStart(numLayers, 0);
            for (size_t l = 2; l < numLayers; l++) {
                aStart[l] = aStart[l - 1] + par.neur[l - 1];
            }

            // 计算每层在 w 中的起始索引
            // wStart[l] = sum(neur[i] * neur[i+1]) for i = 0..l-2
            // （= 「第 l-1 层 → 第 l 层」那一段权重的起点，和 Run 里 w 的排布一致）
            std::vector<size_t> wStart(numLayers, 0);
            for (size_t l = 2; l < numLayers; l++) {
                wStart[l] = wStart[l - 1] + par.neur[l - 2] * par.neur[l - 1];
            }

            // 误差数组（只存 layer 1 ~ 最后一层）
            std::vector<float> error(totalNonInput, 0.0f);

            size_t lastLayer = numLayers - 1;
            size_t outputStart = aStart[lastLayer];
            const study::ActKind actKind = study::ParseActKind(par.actFun);  // 整个 Study 只解析一次

            // ==================== 第一步：铺开目标（输出层） ====================
            // 输出层的目标激活值就是 goal（误差 = goal - 实际激活值，留给第四步的 δ 统一算）
            std::copy(goal.begin(), goal.end(), error.begin() + outputStart);
            const float ero = study::mse(goal.data(), insPar.a.data() + outputStart, par.neur[lastLayer]);

            // ==================== 第二步：逐层往回推「目标的激活值」 ====================
            // 公式（和 rnn::Study 一致）：
            //   目标_{第 l-1 层, i} = Σ_cnn  N × w(cnn → i) / Σ_cnn|w(cnn → i)| × 目标_{第 l 层, cnn}
            //   N = 目标层（第 l-1 层）的神经元数
            //   两处向 rnn 对齐过的地方：
            //     1) N 用「目标层」的大小（原来用「当前层」的 curSize）
            //     2) Σ|w| 按「连出去的」归一（原来按上层神经元「连进来的」归一）
            //   而且推的是「目标」不是「误差」：减实际激活值统一放到第三步做
            //   （原来是一边推一边减，源头就成了误差，和 rnn 两回事）
            //
            // 只推到第 2 层为止：第 1 层的前一层是输入层，输入层没有权重，
            // error[] 里也没有它的位置（aStart[0] == 0 是第 1 层自己的格子，写进去就把第 1 层覆盖了）
            // 每列 1/Σ|w|：Σ 是按「前一层神经元」方向累加的（W[n][m] 对 n 求和）。
            // 权重是 [当前层][前一层] 行连续，所以一行一行扫、把 |w| 累进列向量即可 ——
            // 不需要把整层权重转置出来（转置是 O(cur×prev) 的额外读写）。
            std::vector<float> colSum;  // Σ_n |W[n][m]|
            std::vector<float> invSum;  // 1 / colSum[m]，0 = 分配不了
            for (size_t l = lastLayer; l >= 2; l--) {
                const size_t curStart = aStart[l];
                const size_t prevStart = aStart[l - 1];
                const size_t curSize = par.neur[l];
                const size_t prevSize = par.neur[l - 1];
                const float* wLayer = par.w.data() + wStart[l];

                colSum.assign(prevSize, 0.0f);
                for (size_t cnn = 0; cnn < curSize; cnn++) {
                    study::accumulate_abs(colSum.data(), wLayer + cnn * prevSize, prevSize);
                }
                invSum.assign(prevSize, 0.0f);
                for (size_t m = 0; m < prevSize; m++) {
                    if (colSum[m] != 0.0f) {  // 全 0 就分配不了（不然除出 inf / nan）
                        invSum[m] = 1.0f / colSum[m];
                    }
                }

                // 分「误差份额」，不带放大（给份额乘 N 的三种方案全部以爆炸 / 漂移 / 守卫
                // 触发收场，和 rnn.cpp 同一个结论；饱和激活 scth 才是正解）。
                // 目标和 rnn 一样用「先点积、最后乘一次 1/Σ|w|」：Σ份额 = 1，量级自然衰减。
                // 4×4 分块 GEMV：out 直接写进 error 的前一层区段（与 u = error+curStart 不重叠）
                study::backprop_targets(wLayer, error.data() + curStart, invSum.data(),
                                        error.data() + prevStart, curSize, prevSize);
                // invSum == 0 的列按「分配不了」处理，目标记 0（和原来的 continue 等价；
                // 显式清零是为了挡住 0·inf / 0·NaN 这种边界）
                for (size_t m = 0; m < prevSize; m++) {
                    if (invSum[m] == 0.0f) {
                        error[prevStart + m] = 0.0f;
                    }
                }
            }

            // ==================== 第三、四步：目标 - 激活 → δ → 更新 w / b ====================
            // 误差 = 目标激活值 - 实际激活值（都在激活值空间里算）。
            //
            // 原来是「先按 relu 反解出目标 z（targetZ = max(目标, 0)），再减实际 z」：
            // 那等于把激活函数写死成 relu —— actFun 换成 silu / alt-sigmoid 时反解就是错的
            // （silu 还没有解析反函数），而 rnn 那边一直是在激活值空间相减，这里对齐。
            //
            // 现在把「整表相减」和「逐神经元乘导数」合成一趟 study::delta：
            // δ = (目标 - 实际) · a'(z)，按层批量算（和 rnn::Train 的 2.5 步同一套公式）。
            std::vector<float> deltaBuf;  // 每层复用，不随神经元数反复分配
            for (size_t l = lastLayer; l >= 1; l--) {
                const size_t curStart = aStart[l];
                const size_t prevStart = aStart[l - 1];
                const size_t curSize = par.neur[l];
                const size_t prevSize = par.neur[l - 1];

                deltaBuf.resize(curSize);
                study::delta(actKind, insPar.z.data() + curStart, insPar.a.data() + curStart,
                             error.data() + curStart, deltaBuf.data(), curSize);

                // 更新 w：一个神经元一行连续权重，w += (wlr·δ)·前一层的激活值
                // 第 1 层的前一层是输入层：insPar.a 里没有输入层，
                // a[prevStart + nonitpl] 其实是第 1 层自己的激活值，
                // 这里要用 Run 记下来的真实输入
                const float* prevAct = (l == 1) ? insPar.x.data() : insPar.a.data() + prevStart;
                study::accumulate_weights(par.w.data() + wStart[l], deltaBuf.data(), prevAct, wlr, curSize, prevSize);

                // 更新 b
                study::accumulate_bias(par.b.data() + curStart, deltaBuf.data(), blr, curSize);
            }

            return ero;
        }

        // ==================== 模型存取 ====================
        /*
         * 文件格式（nn / rnn 各有一套 keys，用来区分是谁存的模型）：
         *   keys         : ["wuki", "model", "nn", "file", "wmn"]
         *   rulesVersion : 1
         *   content      : { w, b, neur, actFun }
         */

        /**
         * 保存模型到文件（JSON）
         * @param fileName 目标文件，已存在会被覆盖
         * @return 成功 true；失败 false，并写入 par.errorCode / par.errorBlgig
         */
        inline bool SaveModel(const std::string& fileName) {
            par.errorCode = 0x00;  // 每次调用重算「最近一次的错误」
            par.errorBlgig = "not error";

            // 没初始化（或者 w / b 还没按 neur 分配好）就保存：存出来是个空壳，
            // 而且这种文件自己都读不回来（load 会按尺寸拒收），按错误处理
            if (par.notNeurUnde == false || par.neur.size() < 2 ||
                par.w.size() < InteUtiFun::SumOfProducts(par.neur) ||
                par.b.size() < InteUtiFun::SumUp(par.neur) - par.neur[0]) {
                par.errorCode = 0x01;
                par.errorBlgig = "save";
                return false;
            }

            json outJson;
            outJson["keys"] = {"wuki", "model", "nn", "file", "wmn"};
            outJson["rulesVersion"] = 1;
            outJson["content"] = json::object({
                {"w", par.w},
                {"b", par.b},
                {"neur", par.neur},
                {"actFun", par.actFun},  // 激活函数不在 w / b 里，不存下来就复原不了
            });

            std::ofstream out(fileName, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {  // 路径不存在 / 没权限 / 指向目录
                par.errorCode = 0x02;
                par.errorBlgig = "save";
                return false;
            }

            out << outJson.dump(4) << '\n';
            out.flush();
            if (!out) {  // 磁盘满之类的落盘失败
                par.errorCode = 0x02;
                par.errorBlgig = "save";
                return false;
            }
            return true;
        }

        /**
         * 从文件读取模型
         *
         * 先认文件签名（keys / rulesVersion），再按 neur 校验 w / b 尺寸，
         * 全部通过之后才重建网络状态：
         *   - notNeurUnde = true（否则 Run / Study 会直接报「未初始化」）
         *   - 清空瞬时参数（缓存里的 a / z 是旧权重算出来的，留着会算错）
         *
         * @return 成功 true；失败 false，并写入 par.errorCode / par.errorBlgig
         */
        inline bool LoadModel(const std::string& fileName) {
            par.errorCode = 0x00;
            par.errorBlgig = "not error";

            std::ifstream in(fileName, std::ios::binary);
            if (!in.is_open()) {
                par.errorCode = 0x01;
                par.errorBlgig = "load";
                return false;
            }

            json inJson;
            try {
                in >> inJson;
            } catch (const json::exception&) {  // 不是合法 JSON
                par.errorCode = 0x01;
                par.errorBlgig = "load";
                return false;
            }

            // 认签名：这是本程序存的 nn 模型吗（rnn 的模型、其它 json 都会在这里被挡住）
            const json expectKeys = {"wuki", "model", "nn", "file", "wmn"};
            if (!inJson.is_object() || !inJson.contains("keys") || !inJson.contains("rulesVersion") ||
                !inJson.contains("content") || inJson["keys"] != expectKeys || inJson["rulesVersion"] != 1) {
                par.errorCode = 0x02;
                par.errorBlgig = "load";
                return false;
            }

            // 先把内容取到局部变量并校验，全过了再往 par 上写（失败不留半截状态）
            const json& content = inJson["content"];
            if (!content.is_object() || !content.contains("w") || !content.contains("b") ||
                !content.contains("neur")) {
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            std::vector<float> w;
            std::vector<float> b;
            std::vector<size_t> neur;
            std::string actFun = par.actFun;  // 旧文件没有 actFun：保持当前设置（默认 relu）

            try {
                w = content["w"].get<std::vector<float>>();
                b = content["b"].get<std::vector<float>>();
                neur = content["neur"].get<std::vector<size_t>>();
                if (content.contains("actFun")) {
                    actFun = content["actFun"].get<std::string>();
                }
            } catch (const json::exception&) {  // 类型对不上（w 里混了字符串之类）
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            if (neur.size() < 2 ||
                (actFun != ActFun_ReLU && actFun != ActFun_SiLU && actFun != Alternative_ActFun_Sigmoid)) {
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            // 尺寸校验：Run 只会读 w 的前 SumOfProducts(neur) 个、b 的前 (SumUp(neur) - neur[0]) 个，
            // 比这还小就一定会读越界 → 文件是坏的 / 版本对不上。
            // 这里用 >= 而不是 ==：老版本的 Init 给 b 多留了 neur[0] 个位置，
            // 按 == 判会把那时候存下来的文件判成坏的。
            const size_t needW = InteUtiFun::SumOfProducts(neur);
            const size_t needB = InteUtiFun::SumUp(neur) - neur[0];
            if (w.size() < needW || b.size() < needB) {
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            // 校验都过了，正式落地
            par.w = std::move(w);
            par.b = std::move(b);
            par.neur = std::move(neur);
            par.actFun = actFun;
            par.notNeurUnde = true;

            // 瞬时参数是旧权重算出来的，留着会让 Study 拿错的 a / z / x 去更新
            insPar.a.clear();
            insPar.z.clear();
            insPar.x.clear();
            return true;
        }
    };
}  // namespace nn

#endif  // NN_CPP
