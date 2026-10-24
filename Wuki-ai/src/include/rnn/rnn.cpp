#pragma once
#ifndef RNN_CPP
#define RNN_CPP

#include "rnn.hpp"
#include <cmath>
#include <iostream>  // std::cerr（调试取证打印，定位后删）
#include <algorithm>
#include <fstream>  // std::ifstream / std::ofstream
#include <utility>  // std::move
#include "../external/json.hpp"   // 本项目依赖 nlohmann/json
#include "../optimize/simd.hpp"   // Simd::Runtime：前向点积 / 反向 axpy

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
    // SIMD 内核的短别名（放在 namespace 里：includeMain.cpp 把 nn.cpp / rnn.cpp
    // 收进同一个编译单元，两个文件各自的全局别名会撞）
    namespace opt = Simd::Runtime;
    namespace study = Simd::StudyTime;
    // 激活函数名（原为 _ActFun_* 宏；宏不受 namespace 约束，改为 namespace 内常量）
    static constexpr const char* ActFun_SelfCreTanh = "scth";  // 把 tanh 的 [ -1, 1 ] 变成 [ -2, 2 ]
    static constexpr const char* ActFun_ReLU = "relu";
    static constexpr const char* ActFun_SiLU = "silu";
    static constexpr const char* Alternative_ActFun_Sigmoid = "alt-sigmoid";

    /**
     * 激活函数在 z 点的导数 a'(z)（Train / Study 里把误差换成 δ 用）
     *
     * 手写而不是数值差分：这里每步都要算上万次，数值差分会把前向多跑两遍。
     * 参数 a 只是顺带传进来省一次重算（sigmoid 型能直接用 a 表达自己）。
     *
     * @param actFun 激活函数名（和 Run 里的分支保持同一套名字）
     * @param z      该神经元的加权和（insPar.z 里存着）
     * @param a      该神经元的激活值（insPar.a 里存着）
     * @return a'(z)；未知名字按 ReLU 处理（和 Run 的默认分支一致）
     */
    inline float ActDeriv(const std::string& actFun, const float z, const float a) {
        // 公式只维护在 Simd::StudyTime::deriv_scalar 一处（Train 的批量 δ 也用它）
        return study::deriv_scalar(study::ParseActKind(actFun), z, a);
    }

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

        /*
         * 每一步反推回输入层的目标（给「上一个网络」当训练目标用）
         *
         * 长度 = 步数；inputTargets[t] 的宽度 = neur[0] - hSize（输入层的真实宽度，不含 h 段）。
         *
         * 为什么要留下每一步的：双层架构（DRNN）里 outRNN 一次要吃掉 allRNN 一整块的产出
         * —— 那是好几片，一片一步喂进去 —— 拼出 allRNN 需要的整块目标就必须拿到「每一片」
         * 的目标。只留第 0 步那一份时，一块切两片以上就永远拼不满宽度（Train 只能报错退出）。
         */
        std::vector<std::vector<float>> inputTargets;
        std::vector<float> inputTarget;  // == inputTargets[0]（保留：单步 / 老用法的入口）
    };

    class RNN {
      public:
        Pars par;
        InsPar insPar;

        // CleanContext 函数：清空「这一条序列」的上下文（隐状态 h + 瞬时参数 a / z / x
        // + 反推出来的输入层目标）
        // （h 就是 RNN 的上下文，换一条序列就该从 0 开始；hTarSteps 不在这里清，
        //   它不属于某一条序列，只在换结构时清）
        inline void CleanContext() {
            insPar.a.clear();
            insPar.z.clear();
            insPar.x.clear();
            insPar.h.assign(par.hSize, 0.0f);

            // 输入层目标是「这条序列」反推出来的，换序列就作废（留着会让调用方
            // 拿到上一条序列的目标去训练上一个网络）
            insPar.inputTargets.clear();
            insPar.inputTarget.clear();
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
                std::cerr << "!! rnn::Run 宽度对不上：input=" << input.size() << " + h=" << insPar.h.size()
                          << " != neur[0]=" << par.neur[0] << "\n";
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
            // 一次分配到位：原来是空 vector 逐元素 push_back，doubling 扩容一路
            // realloc + 搬移；Run 是热循环，别在里面攒 reallocs。
            // resize 成「第 1 层 ~ 最后一层」的总格数，后面按 actBase 偏移填。
            const size_t aCap = InteUtiFun::SumUp(par.neur) - par.neur[0];
            a.resize(aCap);
            z.resize(aCap);

            // 初始化
            std::vector<float> oldActVal = input;
            size_t wIndex = 0;
            size_t bIndex = 0;
            size_t actBase = 0;  // 当前层在 a / z 里的起点
            const study::ActKind actKind = study::ParseActKind(par.actFun);

            // 运行
            for (size_t nol = 1; nol < par.neur.size(); nol++) {
                insPar.x.push_back(oldActVal);  // 记录输入
                const size_t inCount = par.neur[nol - 1];  // 本层每个神经元的输入数
                const size_t outCount = par.neur[nol];

                // 整层 z 先算（每行一个 SIMD 点积），再一次性批量激活
                float* zPtr = z.data() + actBase;
                float* aPtr = a.data() + actBase;
                for (size_t cnn = 0; cnn < outCount; cnn++) {
                    const float sum = opt::dot(par.w.data() + wIndex, oldActVal.data(), inCount);
                    wIndex += inCount;
                    zPtr[cnn] = sum + par.b[bIndex];  // z = 加权求和 + 偏置
                    bIndex++;
                }
                study::activate(actKind, zPtr, aPtr, outCount);

                oldActVal.assign(aPtr, aPtr + outCount);
                actBase += outCount;
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
            if (content != ActFun_ReLU && content != ActFun_SiLU && content != Alternative_ActFun_Sigmoid &&
                content != ActFun_SelfCreTanh) {
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
         *
         * 顺带产出 insPar.inputTargets：每一步反推回输入层的目标，见 obtainInputTargets()。
         * 失败路径上它是空的（不会留下半步的残值）。
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

            // 0.5 每个神经元「入边权重绝对值和」的倒数，下标和 b 同表（bStart[l] + n）。
            //
            // 为什么提前算：w 在整段反向里不变（更新最后统一加），而 2.2 的目标分配每一步
            // 都要拿它归一 —— 原来每个时间步都重算一遍 wSum（对全权重一趟扫描），8 步就是
            // 8 趟。这里整段 Train 算一次，反向循环只查表。
            //
            // 为什么用倒数乘法代替每连接除法：除法器吞吐低（且原来写在累加表达式里，
            // 编译器不敢改成乘法），这一处是全 Train 最大的单点开销；w*(1/s) 与 w/s
            // 最多差 1 ulp，对「按 Σ|w| 归一分份额」的学习规则没有可观察的影响。
            std::vector<float> invWSum(par.b.size(), 0.0f);  // 全 0 权重 = 0 → 2.2 里按「分配不了」跳过
            for (size_t l = 1; l < numLayers; l++) {
                for (size_t n = 0; n < par.neur[l]; n++) {
                    const size_t wBase = wStart[l] + n * par.neur[l - 1];
                    // Σ|w|：一个神经元一行连续权重，一个 SIMD 扫描
                    const float wSum = opt::sum_abs(par.w.data() + wBase, par.neur[l - 1]);
                    if (wSum != 0.0f) {
                        invWSum[bStart[l] + n] = 1.0f / wSum;
                    }
                }
            }

            // 1. 展开：先把整段跑完（每步的 a / z / x 都留在缓存里，跨步推目标要用）
            std::vector<std::vector<float>> outputs(steps);

            CleanContext();  // 一条序列从 h = 0 开始
            for (size_t t = 0; t < steps; t++) {
                outputs[t] = Run(inputs[t]);
                if (outputs[t].size() != par.neur.back()) {  // Run 失败（未初始化 / 输入对不上）时返回空
                    std::cerr << "!! rnn::Train：第 " << t << " 步 Run 失败，Run 自己的码 "
                              << par.errorCode << " (" << par.errorBlgig << ")\n";
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

            // 每步复用的缓冲（不在时间循环里反复分配）：
            //   tg        每步的目标，和 aStart 同一套坐标；所有区段都会被写满，不用清零
            //   u         2.2 的中间量 invWSum · tg
            //   deltaBuf  每层的 δ = (目标 - 激活) · a'(z)
            std::vector<float> tg(InteUtiFun::SumUp(par.neur));
            std::vector<float> u;
            std::vector<float> deltaBuf;

            // 每步的位置先开好：下面倒着走，走到哪一步就填哪一格（t = steps-1 ... 0）
            insPar.inputTargets.assign(steps, std::vector<float>());

            // 激活函数名字 → 枚举：整段 Train 只解析一次。
            // 逐神经元调 ActDeriv() 的话每次都要比 4 次字符串 —— 这是每步 × 每层 ×
            // 每神经元的热循环，量级是上亿次比较。
            const study::ActKind actKind = study::ParseActKind(par.actFun);

            for (size_t t = steps; t-- > 0;) {  // t = steps-1 ... 0
                const std::vector<float>& aFlat = insPar.a[t];               // 本步的激活值（第 1 层 ~ 最后一层首尾相接）
                const std::vector<float>& x0 = insPar.x[t * layerCount];     // 第 1 层的输入 = input ‖ h（也就是输入层）

                // 2.1 输出层的 goal 段（层内偏移 0..outWidth-1）：目标就是 goals[t]
                std::copy(goals[t].begin(), goals[t].end(), tg.begin() + aStart[lastLayer]);
                loss[t] = study::mse(goals[t].data(), outputs[t].data(), outWidth);  // MSE

                // 输出层的 h 段（层内偏移 outWidth 起）：就是第 t+1 步反推出来的那个目标
                // （最后一步没有下游，hTar 还是初始的 0）
                for (size_t i = 0; i < par.hSize; i++) {
                    tg[aStart[lastLayer] + outWidth + i] = hTar[i];
                }

                // 2.2 逐层往回推目标：输出层 → …… → 第 1 层 → 输入层
                //     推到输入层是为了拿到「喂进来的 h 该是什么」（2.3 要用）
                //
                //     分配的是「目标」，不是「误差」—— 和 nn::NeuNet::Study 的第二步完全一致。
                //     原来这里是 (tg - insPar.a[...]) * shareN，而 2.4 步又会统一减一次激活值，
                //     于是非输出层的目标被减了两遍激活值：真正落到 dW 上的是
                //       (推上来的误差) - a  =  target_推 - 2a
                //     那个 -a 项和目标毫无关系，等于给隐藏层塞了一个"把激活压向 0"的偏置。
                //     现在只推目标，激活值统一由 2.5 的 δ 减一次。
                //     份额按 Σ|w| 归一（Σ份额 = 1），链上量级自然衰减；不给份额乘 N：
                //     每跳都乘会沿层复利爆炸，试过（16^16 秒炸），稀释问题用饱和激活（scth）解决。
                //
                // 实现：先把归一后的目标乘进 u（u[n] = tg[n] · 1/Σ|w|），
                //       再做 4×4 分块 GEMV：tg[前一层][m] = Σ_n W[n][m] · u[n]。
                //       旧写法是每个 n 对 tg 做一次 axpy（前一层被读改写 prevSize 次）；
                //       现在一次写满前一层，4 个 row 累加器并行，且 u 为 0 的神经元自然贡献 0。
                for (size_t l = lastLayer; l >= 1; l--) {  // 因为是分配，所以从输出层开始往回
                    const size_t curSize = par.neur[l];
                    const size_t prevSize = par.neur[l - 1];
                    u.resize(curSize);
                    study::mul_into(u.data(), tg.data() + aStart[l], invWSum.data() + bStart[l], curSize);
                    study::backprop_targets(par.w.data() + wStart[l], u.data(), nullptr,
                                            tg.data() + aStart[l - 1], curSize, prevSize);
                }

                // 2.3 输入层 h 段上的目标 = 「喂进来的 h(t) 该是什么」，也就是第 t-1 步该产出的 h
                //     倒着走的下一步正好是 t-1，所以立刻就能用上（这就是"展开"换来的东西）
                for (size_t i = 0; i < par.hSize; i++) {
                    hTar[i] = tg[aStart[0] + inHBase + i];
                }

                // 2.35 先存输入层目标（给上一个网络链接用）：输入层（不含 h 段）的那一段
                //      tg 下一步会被 δ 读取，但不再原地改成误差；倒着走，走到 t 就填第 t 格
                insPar.inputTargets[t].assign(tg.begin() + aStart[0], tg.begin() + aStart[0] + inWidth);

                // 2.4 目标 → 误差：不再整表相减，和 2.5 的 δ 合成一趟。
                //      输入层那段 tg 在 2.3 / 2.35 之后没有任何下游读取（2.5 只用第 1 层
                //      及以后的目标），所以原来那趟输入层相减是死计算，一并省掉。
                //
                // 2.5 累加 w / b 的修改量：误差 = 目标 - 实际，所以是朝目标方向（+=）
                //
                // 误差先过一遍本层的激活导数：δ = (目标 - 实际) · a'(z)。
                //
                // 为什么必须乘：这一层是 z → a = f(z) 的饱和环节。原始写法直接拿「激活值空间
                // 的误差」去改 w，等于假设 a 和 z 是线性的 —— 一旦 z 落进 f 的饱和区（scth 在
                // |z| ≳ 4 就贴边），z 再怎么变 a 也几乎不动，可权重照样满幅更新，网络会越推
                // 越深、彻底卡死在饱和面上。乘上 a'(z) 之后饱和区的更新自动趋 0（想动也动不了，
                // 这本来就是对的事实），只有在线性区才真正学。这就是 delta rule 里的 f'(z)。
                for (size_t l = 1; l < numLayers; l++) {
                    const size_t curSize = par.neur[l];
                    const size_t prevSize = par.neur[l - 1];

                    // δ 整层批量算：目标减去实际激活，再乘整层激活导数
                    deltaBuf.resize(curSize);
                    study::delta(actKind, insPar.z[t].data() + bStart[l], aFlat.data() + bStart[l],
                                 tg.data() + aStart[l], deltaBuf.data(), curSize);

                    // b 用 bStart（b 里没有输入层那一段）
                    study::accumulate_bias(dB.data() + bStart[l], deltaBuf.data(), blr, curSize);

                    // 第 l 层的前一层激活值：第 1 层的前一层是输入层（在 x 里，a 里没有）
                    const float* prevAct = (l == 1) ? x0.data() : aFlat.data() + bStart[l - 1];
                    study::accumulate_weights(dW.data() + wStart[l], deltaBuf.data(), prevAct, wlr, curSize, prevSize);
                }
            }

            // 4. 整段的目标都推完、误差都算完之后，再统一更新一次
            //    （避免"边推边更新"让后面的步用上前面的修改 —— 那样结果会和步的顺序有关）
            // 限速：SiLU 正向无饱和 + 更新纯误差驱动、无收缩项 → 权重会指数发散
            // （误差大 → 更新大 → 激活大 → 误差更大；放大份额只是把引爆提前）。
            // 对每个元素的修改量硬 clip 到 ±5×wlr：只限步长、不约束方向。
            // 50×wlr 试过（±0.05，是权重初始幅度的 1/6）：样本间互相打架，loss 来回振荡不收敛；
            // 5×wlr（±0.005）让 50 个样本的拉扯能沉淀到折中解上。
            // 注意：这是限速不是刹车，不保证收敛 —— 真正的收缩（权重衰减之类）是算法决策。
            // clamp / NaN 守卫 / 落权合成一趟（逐元素的先后和原来分趟完全一致：先 clip，
            // clip 完查 finite，过了才 += —— 任一格非有限都整轮拒收，权重不动）。
            // std::clamp 拦得住 inf 拦不住 NaN（比较全假会原样返回），finite 检查就是防它的。
            const float clipVal = 5.0f * wlr;
            for (size_t i = 0; i < par.w.size(); i++) {
                const float d = std::clamp(dW[i], -clipVal, clipVal);
                if (!std::isfinite(d)) { par.errorCode = 0x05; par.errorBlgig = "train-dW-nonfinite"; return {}; }
                par.w[i] += d;
            }
            for (size_t i = 0; i < par.b.size(); i++) {
                const float d = std::clamp(dB[i], -clipVal, clipVal);
                if (!std::isfinite(d)) { par.errorCode = 0x05; par.errorBlgig = "train-dB-nonfinite"; return {}; }
                par.b[i] += d;
            }

            // 老入口 inputTarget 就是第 0 步那一份（obtainInputTarget 用它）
            insPar.inputTarget = insPar.inputTargets[0];

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
                (actFun != ActFun_ReLU && actFun != ActFun_SiLU && actFun != Alternative_ActFun_Sigmoid &&
                 actFun != ActFun_SelfCreTanh)) {
                par.errorCode = 0x03;
                par.errorBlgig = "load";
                return false;
            }

            // 已经按当前配置初始化过（Init 跑过）时，文件里的结构必须和当前的一致。
            // 以前这里照单全收：par.neur / par.hSize 被文件改掉，Run 的输出宽度跟着变成
            // 文件里的宽度，和外面那层（比如 LLM 的 embdSize）对不上 —— 表现是
            // 「权重加载成功、服务也起来了，但 Sample 永远返回越界哨兵，每次回复都是空串」。
            // 结构对不上就当加载失败，交给调用方走「重新训练 + 覆盖保存」。
            if (par.notNeurUnde &&
                (neur != par.neur || hSize != par.hSize || actFun != par.actFun)) {
                par.errorCode = 0x02;
                par.errorBlgig = "load: 文件结构和当前配置不一致（是另一个结构训出来的权重）";
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

        /**
         * 每一步反推回输入层的目标（obtainInputTarget 的整段版本）
         *
         * 长度 = 上一次 Train 的步数，每格宽度 = neur[0] - hSize；
         * 没 Train 过（或 Train 失败 / CleanContext 之后）返回空列表。
         *
         * 给「一个网络喂另一个网络」的架构用：下游网络这一整段的输入目标，
         * 就是上游网络该产出的那一整块（多片拼起来要用到每一片的目标）。
         */
        inline std::vector<std::vector<float>> obtainInputTargets() {
            return insPar.inputTargets;
        }
    };
};  // namespace rnn

#endif  // RNN_CPP
