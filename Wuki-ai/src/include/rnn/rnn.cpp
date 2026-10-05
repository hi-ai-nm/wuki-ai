#pragma once
#ifndef RNN_CPP
#define RNN_CPP

#include "rnn.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>  // std::ifstream / std::ofstream
#include <utility>  // std::move
#include "../external/json.hpp"   // 本项目依赖 nlohmann/json

/*
 * 归属 - run 编号 - 0x01：未初始化网络就运行了
 * 归属 - run 编号 - 0x02：输入大小和 neur[0] 对不上
 *
 * 归属 - train 编号 - 0x01：未初始化网络就训练了
 * 归属 - train 编号 - 0x02：层数过少 / 缓存尺寸和当前结构对不上
 * 归属 - train 编号 - 0x03：goals / inputs 的步数或宽度对不上
 * 归属 - train 编号 - 0x04：Run 失败（返回空）
 *
 * 返回值约定：Train 一次训练一整条序列，返回「每一步的 MSE 误差」（长度 = 步数）；
 * 出错时返回空 vector（不是负数哨兵 —— 一步的 MSE 可能就是 1.0 / 2.0 这类值，
 * 光看数字分不清）。具体原因看 errorCode() / errorInfo()。
 *
 * 归属 - save 编号 - 0x01：未初始化网络（或 w / b 和 neur 对不上）就保存了
 * 归属 - save 编号 - 0x02：文件打不开 / 写不进去
 *
 * 归属 - load 编号 - 0x01：文件打不开 / 不是合法 JSON
 * 归属 - load 编号 - 0x02：不是本程序保存的 rnn 模型（keys / rulesVersion 对不上）
 * 归属 - load 编号 - 0x03：content 缺字段，或 w / b / hSize 和 neur 对不上
 */

namespace rnn {
    // JSON 别名放在 namespace 里：原先写在全局作用域，会和 nn.cpp 的同名别名撞
    // （includeMain.cpp 把两个 .cpp 收在同一个编译单元里）。
    using json = nlohmann::json;
    // 激活函数名（原为 _ActFun_* 宏；宏不受 namespace 约束，改为 namespace 内常量）
    static constexpr const char* ActFun_SelfCreTanh = "scth";  // 把 tanh 的 [ -1, 1 ] 变成 [ -2, 2 ]
    static constexpr const char* ActFun_ReLU = "relu";
    static constexpr const char* ActFun_SiLU = "silu";
    static constexpr const char* Alternative_ActFun_Sigmoid = "alt-sigmoid";

    struct Pars {  // 参数
        std::vector<float> w;
        std::vector<float> b;
        size_t hSize = 0;  // 隐状态大小

        std::string actFun = ActFun_SiLU;  // 激活函数

        std::vector<size_t> neur;  // 每层神经元数量
        bool notNeurUnde = false;  // 是否已定义 neur

        size_t errorCode = 0x00;               // 错误码（默认无错误）
        std::string errorBlgig = "not error";  // 错误归属（errorBlonging）
    };

    struct InsPar {
        std::vector<std::vector<float>> a;  // 每个时间步的 a
        std::vector<std::vector<float>> z;  // 每个时间步的 z
        std::vector<std::vector<float>> x;

        std::vector<float> h;  // 隐状态

        std::vector<float> inputTarget; // 输入层的目标
    };

    class RNN {
      public:
        Pars par;
        InsPar insPar;

        // CleanContext 函数：清空「这一条序列」的上下文（隐状态 h + 瞬时参数 a / z / x）
        // （h 就是 RNN 的上下文，换一条序列就该从 0 开始；hTarSteps 不在这里清，
        //   它不属于某一条序列，只在换结构时清）
        inline void CleanContext() {
            insPar.a.clear();
            insPar.z.clear();
            insPar.x.clear();
            insPar.h.assign(par.hSize, 0.0f);
        }

        // Run 函数
        std::vector<float> Run(std::vector<float> input) {    // 值传，因为要修改
            par.errorCode = 0x00;  // 每次调用重算「最近一次的错误」
            par.errorBlgig = "not error";

            // 处理未定义 neur 的情况
            if (par.notNeurUnde == false) {
                par.errorCode = 0x01;
                par.errorBlgig = "run";
                return std::vector<float>{};  // 返回空列表（需 C++11 ）
            }

            // 输入大小必须和输入层对得上，否则下面读 oldActVal / 记 insPar.x 都会越界
            if (input.size() + insPar.h.size() != par.neur[0]) {
                par.errorCode = 0x02;
                par.errorBlgig = "run";
                return std::vector<float>{};
            }

            // 拼装 input 和 h
            input.reserve(input.size() + insPar.h.size());          // 减少扩容
            input.insert(input.end(), insPar.h.begin(), insPar.h.end());   // 把 b 拷贝拼到 a 后面

            insPar.a.push_back({});     // 添加一个 a 列表
            insPar.z.push_back({});     // 添加一个 z 列表
            std::vector<float>& a = insPar.a[insPar.a.size() - 1];
            std::vector<float>& z = insPar.z[insPar.z.size() - 1];

            // 初始化
            std::vector<float> oldActVal = input;
            std::vector<float> newActVal;
            size_t wIndex = 0;
            size_t bIndex = 0;

            // 运行
            for (size_t nol = 1; nol < par.neur.size(); nol++) {
                insPar.x.push_back(oldActVal);  // 记录输入
                newActVal.clear();
                for (size_t cnn = 0; cnn < par.neur[nol]; cnn++) {
                    float sum = 0;  // 加权求和
                    for (size_t nonitpl = 0; nonitpl < par.neur[nol - 1]; nonitpl++) {
                        sum += (par.w[wIndex] * oldActVal[nonitpl]);  // 加权求和
                        wIndex++;
                    }

                    // 计算 z（加权求和 + 偏置）
                    // 注意：不能叫 z，否则会遮蔽上面那个 insPar.z 的引用
                    float zVal = sum + par.b[bIndex];
                    bIndex++;

                    // 计算 a（激活函数）
                    float aVal;
                    if (par.actFun == ActFun_SelfCreTanh) {
                        aVal = std::tanh(zVal / 2.0f) * 2.0f;
                    } else  if (par.actFun == ActFun_ReLU) {
                        aVal = zVal > 0 ? zVal : 0;
                    } else if (par.actFun == ActFun_SiLU) {
                        aVal = zVal / (1.0f + expf(-zVal));
                    } else if (par.actFun == Alternative_ActFun_Sigmoid) {
                        aVal = 1.0f / (1.0f + expf(-zVal));
                    } else {
                        aVal = zVal > 0 ? zVal : 0;  // 默认 ReLU
                    }

                    // 存储瞬时参数
                    z.push_back(zVal);
                    a.push_back(aVal);

                    newActVal.push_back(aVal);
                }
                oldActVal = newActVal;
            }

            // 记录 h
            for (size_t i = 0; i < par.hSize; i++) {
                insPar.h[i] = oldActVal[i + oldActVal.size() - par.hSize];
            }

            return oldActVal;
        }

        // Init
        size_t Init(std::vector<size_t>& neur, size_t hSize = 3) {
            neur[0] += hSize;                // 输入层增加 hSize 个神经元
            neur[neur.size() - 1] += hSize;  // 输出层增加 hSize 个神经元

            // 初始化随机数引擎
            std::random_device rd;
            std::mt19937 gen(rd());

            par.w.resize(InteUtiFun::SumOfProducts(neur));  // 预留 w 空间
            par.b.resize(InteUtiFun::SumUp(neur) - neur[0]);          // 预留 b 空间

            // 初始化 WBC （权重疲劳）
#ifdef _WBC_On
            size_t WBC = _WBC_Init;
#endif

            // 循环遍历 w
            size_t wIndex = 0;
            for (size_t nol = 1; nol < par.neur.size(); nol++) {
                for (size_t cnn = 0; cnn < par.neur[nol]; cnn++) {
#ifdef _WBC_On
                    size_t connectCount = 0;  // 当前神经元的连接数（只在开启权重疲劳时统计）
#endif
                    for (size_t nonitpl = 0; nonitpl < par.neur[nol - 1]; nonitpl++) {
                        float x;
#ifdef _WBC_On
                        if (connectCount >= WBC) {
                            x = 0.08f;  // 超过限制，使用权重 0.08
                        } else {
                            x = InteUtiFun::RandomRange(gen, (-0.3f), 0.3f);  // 未超过，使用随机权重
                        }
                        connectCount++;
#else
                        x = InteUtiFun::RandomRange(gen, (-0.3f), 0.3f);  // 循环神经网络，范围少一点
#endif
                        par.w[wIndex] = x;
                        wIndex++;
                    }
                }
            }

            // 初始化偏置 b
            for (auto& b : par.b) {
                b = InteUtiFun::RandomRange(gen, -0.15f, 0.15f);  // 同理，范围少一点
            }

            par.hSize = hSize;
            insPar.h.assign(hSize, 0.0f);
            CleanContext();  // 换了结构 / 新权重，a / z / x 里可能还留着旧尺寸的缓存

            return 0x00;
        }

        // 设置层数
        inline void set_neur(const std::vector<size_t>& neur, size_t hSize, bool autoInit) {
            par.neur = neur;  // 必须先赋值：Init 是按 par.neur 来分配 w / b 的
            if (autoInit) {
                Init(par.neur, hSize);
            }
            par.notNeurUnde = true;
        }

        inline void set_neur(std::initializer_list<size_t> neur, size_t hSize, bool autoInit) {
            par.neur = neur;  // 同上：先赋值再 Init
            if (autoInit) {
                Init(par.neur, hSize);
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

        /**
         * 整段序列的对比学习（自创）：先把 t = 0..T-1 全部 Run 完，再倒着跨步推目标，最后统一更新
         *
         * 下标约定 —— 三张表各管一段，不能混用：
         *   aStart[l]  第 l 层在「展平激活值 / 目标 / 误差」里的起点（**含输入层**），整表尺寸 SumUp(neur)
         *   bStart[l]  第 l 层在 b 里的起点（**不含输入层**），bStart[l] == aStart[l] - neur[0]
         *   wStart[l]  「第 l-1 层 → 第 l 层」那段权重在 w 里的起点（和 Run 里 wIndex 的排布一致）
         *
         * 时间是怎么接上的 —— 这就是 RNN 和 nn 的根本区别，也是必须"展开"的原因：
         *   第 t 步喂进来的是 h(t-1)、产出的是 h(t)（输出层的 h 段）。
         *   第 t+1 步把目标一路反推到输入层，落在「输入层 h 段」上的值就是
         *   「喂进来的 h(t) 该是什么」，也就是第 t 步该产出的 h 的目标。
         *   我们从 t = T-1 倒着处理，所以第 t+1 步的结果能立刻喂给第 t 步（不用跨调用缓存）。
         *   最后一步产出的 h 没有下游消费者，它的目标按 0。
         *
         * 每次调用都会先 CleanContext()：一条序列从 h = 0 开始，顺手把上一条序列的缓存清掉。
         *
         * @param goals  每一步的目标，goals[t].size() 必须等于「输出层宽度 - hSize」
         * @param inputs 每一步的输入，inputs[t].size() 必须等于「输入层宽度 - hSize」
         * @return 每一步的 MSE 误差（长度 = 步数）；出错返回空 vector，原因见 errorCode() / errorInfo()
         */
        std::vector<float> Train(const std::vector<std::vector<float>>& goals,
                                 const std::vector<std::vector<float>>& inputs,
                                 const float wlr, const float blr) {
            par.errorCode = 0x00;  // 每次调用重算「最近一次的错误」
            par.errorBlgig = "not error";

            if (par.notNeurUnde == false) {
                par.errorCode = 0x01;
                par.errorBlgig = "train";
                return {};
            }

            const size_t numLayers = par.neur.size();
            if (numLayers < 2) {
                par.errorCode = 0x02;
                par.errorBlgig = "train";
                return {};
            }

            const size_t lastLayer = numLayers - 1;
            const size_t layerCount = numLayers - 1;                                      // Run 每步往 a / z / x 里 push 的条数
            const size_t outWidth = par.neur.back() - par.hSize;                          // 输出层里真正的输出宽度（不含 h）
            const size_t inWidth = par.neur[0] - par.hSize;                               // 输入层里真正的输入宽度（不含 h）
            const size_t inHBase = par.neur[0] - par.hSize;                               // 输入层里 h 段的起点（层内偏移）
            const size_t aStepSize = InteUtiFun::SumUp(par.neur) - par.neur[0];           // 每步 a / z 的格数
            const size_t steps = goals.size();

            if (steps == 0 || inputs.size() != steps) {
                par.errorCode = 0x03;
                par.errorBlgig = "train";
                return {};
            }
            for (size_t t = 0; t < steps; t++) {
                if (goals[t].size() != outWidth || inputs[t].size() != inWidth) {
                    par.errorCode = 0x03;
                    par.errorBlgig = "train";
                    return {};
                }
            }

            // 0. 三张起始表
            std::vector<size_t> aStart(numLayers, 0);  // 含输入层
            std::vector<size_t> bStart(numLayers, 0);  // 不含输入层
            std::vector<size_t> wStart(numLayers, 0);
            for (size_t l = 1; l < numLayers; l++) {
                aStart[l] = aStart[l - 1] + par.neur[l - 1];
                bStart[l] = aStart[l] - par.neur[0];  // b 里没有输入层那一段，整体前移 neur[0]
                if (l >= 2) {                         // wStart[1] == 0（「输入层 → 第 1 层」那段排在最前面）
                    wStart[l] = wStart[l - 1] + par.neur[l - 2] * par.neur[l - 1];
                }
            }

            // 1. 展开：先把整段跑完（每步的 a / z / x 都留在缓存里，跨步推目标要用）
            std::vector<std::vector<float>> outputs(steps);

            CleanContext();  // 一条序列从 h = 0 开始
            for (size_t t = 0; t < steps; t++) {
                outputs[t] = Run(inputs[t]);
                if (outputs[t].size() != par.neur.back()) {  // Run 失败（未初始化 / 输入对不上）时返回空
                    par.errorCode = 0x04;
                    par.errorBlgig = "train";
                    return {};
                }
            }
            // a 是「每步一条」的扁平表：a[步][bStart[l] + n] 就是第 l 层第 n 个神经元的激活值
            // x 是「每步、每层」各一条：x[步 * layerCount + (l - 1)] 是第 l 层的输入
            if (insPar.a.size() != steps || insPar.x.size() != steps * layerCount ||
                insPar.a[0].size() != aStepSize || insPar.x[0].size() != par.neur[0]) {
                par.errorCode = 0x02;  // 兜底：缓存和当前结构对不上
                par.errorBlgig = "train";
                return {};
            }

            // 2. 倒着走：推目标 → 记下「上一步该产出的 h」→ 算误差 → 累加修改量
            std::vector<float> loss(steps, 0.0f);              // 每步的 MSE 误差
            std::vector<float> hTar(par.hSize, 0.0f);          // 第 t+1 步推出来的「第 t 步该产出的 h」
            std::vector<float> dW(par.w.size(), 0.0f);         // 整段的权重修改量
            std::vector<float> dB(par.b.size(), 0.0f);         // 整段的偏置修改量

            for (size_t t = steps; t-- > 0;) {  // t = steps-1 ... 0
                const std::vector<float>& aFlat = insPar.a[t];               // 本步的激活值（第 1 层 ~ 最后一层首尾相接）
                const std::vector<float>& x0 = insPar.x[t * layerCount];     // 第 1 层的输入 = input ‖ h（也就是输入层）

                // 2.1 目标缓冲：和 aStart 同一套坐标（含输入层）
                std::vector<float> tg(InteUtiFun::SumUp(par.neur), 0.0f);

                // 输出层的 goal 段（层内偏移 0..outWidth-1）：目标就是 goals[t]
                for (size_t i = 0; i < outWidth; i++) {
                    tg[aStart[lastLayer] + i] = goals[t][i];
                    const float d = goals[t][i] - outputs[t][i];
                    loss[t] += d * d;  // MSE
                }

                // 输出层的 h 段（层内偏移 outWidth 起）：就是第 t+1 步反推出来的那个目标
                // （最后一步没有下游，hTar 还是初始的 0）
                for (size_t i = 0; i < par.hSize; i++) {
                    tg[aStart[lastLayer] + outWidth + i] = hTar[i];
                }

                // 2.2 逐层往回推目标：输出层 → …… → 第 1 层 → 输入层
                //     推到输入层是为了拿到「喂进来的 h 该是什么」（2.3 要用）
                for (size_t l = lastLayer; l >= 1; l--) {  // 因为是分配，所以从输出层开始往回
                for (size_t n = 0; n < par.neur[l]; n++) {
                    const size_t wBase = wStart[l] + n * par.neur[l - 1];

                    // 先计算 w 的和（绝对值后）
                    float wSum = 0.0f;
                    for (size_t m = 0; m < par.neur[l - 1]; m++) {
                        wSum += std::abs(par.w[wBase + m]);
                    }
                    if (wSum == 0.0f) continue;  // 全 0 分配不了（否则 0/0 = NaN）

                    // 计算目标（传递到上一层）
                    for (size_t m = 0; m < par.neur[l - 1]; m++) {
                        tg[aStart[l - 1] + m] += (par.w[wBase + m] / wSum) * tg[aStart[l] + n];
                    }
                }
            }

                // 2.3 输入层 h 段上的目标 = 「喂进来的 h(t) 该是什么」，也就是第 t-1 步该产出的 h
                //     倒着走的下一步正好是 t-1，所以立刻就能用上（这就是"展开"换来的东西）
                for (size_t i = 0; i < par.hSize; i++) {
                    hTar[i] = tg[aStart[0] + inHBase + i];
                }

                // 2.4 目标 → 误差：逐层减实际激活值
                //     全程走 aStart 坐标（含输入层），不会像 a.back() 那样越界
                std::vector<float> error = tg;
                for (size_t i = 0; i < par.neur[0]; i++) {  // 输入层：激活值就是 input ‖ h
                    error[aStart[0] + i] -= x0[i];
                }
                for (size_t l = 1; l < numLayers; l++) {
                    for (size_t n = 0; n < par.neur[l]; n++) {
                        error[aStart[l] + n] -= aFlat[bStart[l] + n];  // 第 l 层的激活值
                    }
                }

                // 2.5 累加 w / b 的修改量：误差 = 目标 - 实际，所以是朝目标方向（+=）
                for (size_t l = 1; l < numLayers; l++) {
                    for (size_t n = 0; n < par.neur[l]; n++) {
                        const float e = error[aStart[l] + n];

                        dB[bStart[l] + n] += blr * e;  // b 用 bStart（b 里没有输入层那一段）

                        const size_t wBase = wStart[l] + n * par.neur[l - 1];
                        for (size_t m = 0; m < par.neur[l - 1]; m++) {
                            // 第 l 层的前一层激活值：第 1 层的前一层是输入层（在 x 里，a 里没有）
                            const float prevAct = (l == 1) ? x0[m] : aFlat[bStart[l - 1] + m];
                            dW[wBase + m] += wlr * e * prevAct;
                        }
                    }
                }

                // 3. 更新inputTarget：给上一个网络链接用
                // 把输入层的目标（不含 h）存下来
                insPar.inputTarget.assign(tg.begin() + aStart[0], tg.begin() + aStart[0] + inWidth);
            }

            // 4. 整段的目标都推完、误差都算完之后，再统一更新一次
            //    （避免"边推边更新"让后面的步用上前面的修改 —— 那样结果会和步的顺序有关）
            for (size_t i = 0; i < par.w.size(); i++) {
                par.w[i] += dW[i];
            }
            for (size_t i = 0; i < par.b.size(); i++) {
                par.b[i] += dB[i];
            }

            return loss;  // 每一步的 MSE 误差
        }

        // ==================== 模型存取 ====================
        /*
         * 文件格式（nn / rnn 各有一套 keys，用来区分是谁存的模型）：
         *   keys         : ["wuki", "model", "rnn", "file", "wmrn"]
         *   rulesVersion : 1
         *   content      : { w, b, neur, hSize, actFun }
         *
         * 注意：par.neur 是 Init 加过 hSize 之后的尺寸（输入层 / 输出层都含 h），
         *       存下来的就是「含 h」的结构，所以加载时不再走 Init，直接恢复。
         *
         * 组装 / 校验 和 文件 IO 分成了两层，为什么：
         *   DRNN（双层 RNN）要把两个子网装进同一个文件，它拿到的只能是「内存里的 json」，
         *   不能再走文件。所以把 json 那一半抽出来，文件那一半只在外面包一层：
         *   toJson / fromJson      只管 json
         *   SaveModel / LoadModel  在 json 外面再包一层读写（行为、格式、错误码全都没变）
         */

        /**
         * 把当前模型组装成 json
         * @param out 成功时写入组装好的 json（失败时被清成 null，不留上一次的旧内容）
         * @return 成功 true；失败 false，并写入 par.errorCode / par.errorBlgig
         */
        inline bool toJson(json& out) {
            par.errorCode = 0x00;  // 每次调用重算「最近一次的错误」
            par.errorBlgig = "not error";
            out = json();          // 失败时外面拿到的是 null，不会误用上一次的结果

            // 没初始化（或者 w / b 还没按 neur 分配好）就保存：存出来是个空壳，
            // 而且这种文件自己都读不回来（load 会按尺寸拒收），按错误处理
            if (par.notNeurUnde == false || par.neur.size() < 2 ||
                par.w.size() < InteUtiFun::SumOfProducts(par.neur) ||
                par.b.size() < InteUtiFun::SumUp(par.neur) - par.neur[0]) {
                par.errorCode = 0x01;
                par.errorBlgig = "save";
                return false;
            }

            out["keys"] = {"wuki", "model", "rnn", "file", "wmrn"};
            out["rulesVersion"] = 1;
            out["content"] = json::object({
                {"w", par.w},
                {"b", par.b},
                {"neur", par.neur},
                {"hSize", par.hSize},
                {"actFun", par.actFun},  // 激活函数不在 w / b 里，不存下来就复原不了
            });
            return true;
        }

        /**
         * 保存模型到文件（JSON）
         * @param fileName 目标文件，已存在会被覆盖
         * @return 成功 true；失败 false，并写入 par.errorCode / par.errorBlgig
         */
        inline bool SaveModel(const std::string& fileName) {
            json outJson;
            if (!toJson(outJson)) {  // 错误码 / 归属已经由 toJson 写好，这里直接往上抛
                return false;
            }

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
         * 从 json 恢复模型（认签名 → 尺寸校验 → 才落地）
         *
         * 先认文件签名（keys / rulesVersion），再按 neur 校验 w / b 尺寸和 hSize，
         * 全部通过之后才重建网络状态：
         *   - notNeurUnde = true（否则 Run / Study 会直接报「未初始化」）
         *   - dW / db 按新尺寸清零、h 和 a / z / x 缓存全部清空
         *     （旧缓存是旧权重 / 旧结构算出来的，留着会越界或者算错）
         *
         * 文件那一层的错（打不开 / 不是合法 JSON）不在这里报，由 LoadModel 负责 ——
         * 这里只认内容：不是本程序存的 rnn 模型、尺寸对不上，都是 0x02 / 0x03。
         *
         * @return 成功 true；失败 false，并写入 par.errorCode / par.errorBlgig
         */
        inline bool fromJson(const json& inJson) {
            par.errorCode = 0x00;
            par.errorBlgig = "not error";

            // 这是本程序存的 rnn 模型（nn 的模型、其它 json 都会在这里被挡住）
            const json targetKeys = {"wuki", "model", "rnn", "file", "wmrn"};
            if (!inJson.is_object() || !inJson.contains("keys") || !inJson.contains("rulesVersion") ||
                !inJson.contains("content") || inJson["keys"] != targetKeys || inJson["rulesVersion"] != 1) {
                par.errorCode = 0x02;
                par.errorBlgig = "load";
                return false;
            }

            // 先把内容取到局部变量并校验，全过了再往 par 上写（失败不留半截状态）
            const json& content = inJson["content"];
            if (!content.is_object() || !content.contains("w") || !content.contains("b") ||
                !content.contains("neur") || !content.contains("hSize")) {
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            std::vector<float> w;
            std::vector<float> b;
            std::vector<size_t> neur;
            size_t hSize = 0;
            std::string actFun = par.actFun;  // 旧文件没有 actFun：保持当前设置（默认 relu）

            try {
                w = content["w"].get<std::vector<float>>();
                b = content["b"].get<std::vector<float>>();
                neur = content["neur"].get<std::vector<size_t>>();
                hSize = content["hSize"].get<size_t>();
                if (content.contains("actFun")) {
                    actFun = content["actFun"].get<std::string>();
                }
            } catch (const json::exception&) {  // 类型对不上（w 里混了字符串之类）
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            // hSize 必须和 neur 对得上：Init 是把 hSize 加进 neur[0] 和 neur[最后] 的
            // （hSize == 0 是合法的「无隐状态」配置，这里不拦）
            if (neur.size() < 2 || neur[0] < hSize || neur.back() < hSize ||
                (actFun != ActFun_ReLU && actFun != ActFun_SiLU && actFun != Alternative_ActFun_Sigmoid)) {
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            // 尺寸校验：Run / Study 只会读 w 的前 SumOfProducts(neur) 个、b 的前 (SumUp(neur) - neur[0]) 个，
            // 比这还小就一定会读越界 → 文件是坏的 / 版本对不上
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
            par.hSize = hSize;
            par.actFun = actFun;
            par.notNeurUnde = true;

            insPar.h.assign(par.hSize, 0.0f);
            CleanContext();
            return true;
        }

        /**
         * 从文件读取模型
         *
         * 只管文件这一层（打不开 / 不是合法 JSON → 0x01），认签名和尺寸校验都在 fromJson 里
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

            return fromJson(inJson);
        }

        inline std::vector<float> obtainInputTarget() {
            return insPar.inputTarget;
        }
    };
};  // namespace rnn

#endif  // RNN_CPP
