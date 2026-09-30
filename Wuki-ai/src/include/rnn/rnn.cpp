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
 * 归属 - study 编号 - 0x01：未初始化网络就训练了
 * 归属 - study 编号 - 0x02：缓存和当前结构 / 目标步数对不上（set_neur 之后没重新 Run）
 * 归属 - study 编号 - 0x03：层数过少
 * 归属 - study 编号 - 0x04：goal 每步的宽度和输出层（去掉 h）对不上
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
    static constexpr const char* ActFun_ReLU = "relu";
    static constexpr const char* ActFun_SiLU = "silu";
    static constexpr const char* Alternative_ActFun_Sigmoid = "alt-sigmoid";

    struct Pars {  // 参数
        std::vector<float> w;
        std::vector<float> b;
        size_t hSize;  // 隐状态大小

        std::string actFun = ActFun_ReLU;  // 激活函数

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
    };

    class RNN {
      public:
        Pars par;
        InsPar insPar;
        std::vector<float> dW;  // 权重梯度累加器，和 par.w 同尺寸
        std::vector<float> db;  // 偏置梯度累加器，和 par.b 同尺寸

        // CleanContext 函数
        inline void CleanContext() {
            insPar.a.clear();
            insPar.z.clear();
            insPar.x.clear();
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
                    if (par.actFun == ActFun_ReLU) {
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
            for (size_t i = par.hSize; i < par.hSize; i++) { insPar.h[i] = oldActVal[i + oldActVal.size() - par.hSize]; }

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

            dW.resize(par.w.size(), 0.0f);
            db.resize(par.b.size(), 0.0f);

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
                            x = InteUtiFun::RandomRange(gen, 0.3f, 0.6f);  // 未超过，使用随机权重
                        }
                        connectCount++;
#else
                        x = InteUtiFun::RandomRange(gen, 0.3f, 0.6f);
#endif
                        par.w[wIndex] = x;
                        wIndex++;
                    }
                }
            }

            // 初始化偏置 b
            for (auto& b : par.b) {
                b = InteUtiFun::RandomRange(gen, 0.3f, 0.6f);
            }

            par.hSize = hSize;
            insPar.h.resize(hSize, 0.0f);

            return 0x00;
        }

        // 设置层数
        inline void set_neur(const std::vector<size_t>& neur, size_t hSize = 16, bool autoInit = true) {
            par.neur = neur;  // 必须先赋值：Init 是按 par.neur 来分配 w / b 的
            if (autoInit) {
                Init(par.neur, hSize);
            }
            par.notNeurUnde = true;
        }

        inline void set_neur(std::initializer_list<size_t> neur, size_t hSize = 16, bool autoInit = true) {
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

        // 对比学习（自创）
        // 核心公式：\text{目标激活值}_{\text{前一层}} = N \times \frac{w}{\sum |w|} \times \text{目标激活值}_{\text{当前层}}（Markdown）
        // 流程：① 输出层铺开目标 ② 逐层往回推目标 ③ 目标统一减实际激活值得到误差 ④ 用误差改 w 和 b
        // （和 nn::Study 是同一套：中途推的是「目标」，最后才减激活值）
        std::vector<float> Study(const std::vector<std::vector<float>>& goal, float wlr = 0.01f, float blr = 0.001f) {
            if (!par.notNeurUnde) {     // 没初始化就训练了
                par.errorCode = 0x01;  // 错误码 0x01
                par.errorBlgig = "study";
                return { (-1.0f) };  // 返回负数表示错误
            }
            if (par.neur.size() < 2) {  // 层数不对
                par.errorCode = 0x03;
                par.errorBlgig = "study";
                return { (-3.0f) };
            }

            // 下面所有下标都建立在「缓存和当前结构一致」上：
            // set_neur 换了结构却没重新 Run 的话，insPar 里还是旧尺寸的数据，
            // 按新结构去读（a[t][outStart + i]、x[t * 层数 + l]）就会越界，先逐层查一遍
            const size_t layerCount = par.neur.size() - 1;  // 每步 push 进 x 的层数
            size_t stepSize = 0;                            // 每步 a / z 的尺寸（不含输入层）
            for (size_t l = 1; l < par.neur.size(); l++) {
                stepSize += par.neur[l];
            }

            if (goal.size() != insPar.a.size() || insPar.z.size() != insPar.a.size() ||
                insPar.x.size() != insPar.a.size() * layerCount) {  // 步数 / 缓存条数不对
                par.errorCode = 0x02;
                par.errorBlgig = "study";
                return { (-2.0f) };
            }
            for (size_t t = 0; t < goal.size(); t++) {
                if (goal[t].size() + par.hSize != par.neur.back()) {  // 每步宽度不对
                    par.errorCode = 0x04;
                    par.errorBlgig = "study";
                    return { (-4.0f) };
                }
                if (insPar.a[t].size() != stepSize || insPar.z[t].size() != stepSize) {
                    par.errorCode = 0x02;
                    par.errorBlgig = "study";
                    return { (-2.0f) };
                }
                for (size_t l = 0; l < layerCount; l++) {
                    if (insPar.x[t * layerCount + l].size() != par.neur[l]) {
                        par.errorCode = 0x02;
                        par.errorBlgig = "study";
                        return { (-2.0f) };
                    }
                }
            }

            // =============== 准备展开 ===============
            std::vector<float> loss;                                             // 每步的损失
            std::vector<float> hGoalNext(par.hSize, 0.0f);                       // 输出层里 h 那部分的目标值
            std::vector<float> dW(par.w.size(), 0.0f), dB(par.b.size(), 0.0f);   // 修改值

            // 每层的起始下标（和 nn::Study 的 aStart / wStart 是同一套约定）：
            //   bStart[l] = sum(neur[1..l-1])，第 l 层在 a / z 里的起点
            //   wStart[l] = 「第 l-1 层 → 第 l 层」那段权重在 w 里的起点（wStart[1] == 0）
            // a / z 只存 layer 1 ~ 最后一层（不含输入层），所以累加时不能把 neur[0] 算进去。
            std::vector<size_t> bStart(par.neur.size(), 0), wStart(par.neur.size(), 0);
            for (size_t l = 2; l < par.neur.size(); l++) {
                bStart[l] = bStart[l - 1] + par.neur[l - 1];
                wStart[l] = wStart[l - 1] + par.neur[l - 2] * par.neur[l - 1];
            }

            // StudyOneStep
            auto StudyOneStep = [&wStart, &bStart, &dW, &dB, wlr, blr, &hGoalNext, layerCount, this](
                const std::vector<float>& goal, std::vector<float>& lossAll, size_t t
            ) {
                // 初始化：目标的格子只给 layer 1 ~ 最后一层（和 bStart 对应，输入层没有格子）
                std::vector<float> allTarget(InteUtiFun::SumUp(par.neur) - par.neur[0], 0.0f);

                lossAll.push_back(0.0f);
                auto& loss = lossAll.back();

                const size_t outBase = bStart[par.neur.size() - 1];   // 输出层在 a / z 里的起点
                const size_t outWidth = par.neur.back() - par.hSize;  // 输出层里真正输出的宽度（不含 h）

                {
                    {
                        // 计算输出层的误差
                        for (size_t idx = 0; idx < outWidth; idx++) {
                            allTarget[outBase + idx] = goal[idx] - insPar.a[t][outBase + idx];    // 计算误差
                            loss += allTarget[outBase + idx] * allTarget[outBase + idx];          // 计算总 loss
                        }

                        // 把 hGoalNext 用上（目标现在恒为 0，等于先把 h 压向 0 占位）
                        // hGoalNext 只有 hSize 个格子，下标要减掉 outWidth，不能直接用 idx
                        for (size_t idx = outWidth; idx < par.neur.back(); idx++) {
                            allTarget[outBase + idx] = hGoalNext[idx - outWidth] - insPar.a[t][outBase + idx];    // 计算误差
                            loss += allTarget[outBase + idx] * allTarget[outBase + idx];                           // 计算总 loss
                        }
                    }

                    {
                        // 计算隐藏层的误差（往回摊到前一层）
                        // 只推到第 2 层：第 1 层的前一层是输入层，allTarget 里没有输入层的格子
                        // （bStart[1] == bStart[0] == 0，再往下推会把第 1 层的格子覆盖掉）
                        for (size_t l = par.neur.size() - 1; l >= 2; l--) {
                            for (size_t neu = 0; neu < par.neur[l]; neu++) {
                                // 计算链接的 w 的和
                                float sum = 0.0f;
                                for (size_t n = 0; n < par.neur[l - 1]; n++) {
                                    sum += par.w[wStart[l] + neu * par.neur[l - 1] + n];
                                }
                                if (sum == 0.0f) continue;  // 全 0 摊不了（不然除出 inf / nan）

                                for (size_t n = 0; n < par.neur[l - 1]; n++) {
                                    float wij = par.w[wStart[l] + neu * par.neur[l - 1] + n];
                                    float intenScore = wij / sum;

                                    float err =
                                        intenScore * allTarget[bStart[l] + neu];  // 计算误差

                                    allTarget[bStart[l - 1] + n] += err;          // 更新误差
                                }
                            }
                        }
                    }

                    {
                        // 计算 w 和 b 的修改值
                        for (size_t l = 1; l < par.neur.size(); l++) {
                            // 前一层（l == 1 时就是输入层）的激活值：输入层不在 a 里，要去 x 拿
                            // （每步往 x 里记 layerCount 条，第 l 层的输入就是第 l-1 条）
                            const std::vector<float>& prevAct = (l == 1) ? insPar.x[t * layerCount] : insPar.a[t];
                            const size_t prevBase = (l == 1) ? 0 : bStart[l - 1];

                            for (size_t neu = 0; neu < par.neur[l]; neu++) {
                                dB[bStart[l] + neu] += blr * allTarget[bStart[l] + neu];

                                for (size_t n = 0; n < par.neur[l - 1]; n++) {
                                    dW[wStart[l] + neu * par.neur[l - 1] + n] +=
                                        wlr * allTarget[bStart[l] + neu] * prevAct[prevBase + n];
                                }
                            }
                        }
                    }
                }

            };

            // =============== 开始训练 ===============
            for (int t = goal.size() - 1; t >= 0; t--) {
                StudyOneStep(goal[t], loss, (size_t)t);
            }
            // =============== 更新参数 ===============
            for (size_t i = 0; i < par.w.size(); i++) { par.w[i] += dW[i]; }
            for (size_t i = 0; i < par.b.size(); i++) { par.b[i] += dB[i]; }

            return loss;
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
            outJson["keys"] = {"wuki", "model", "rnn", "file", "wmrn"};
            outJson["rulesVersion"] = 1;
            outJson["content"] = json::object({
                {"w", par.w},
                {"b", par.b},
                {"neur", par.neur},
                {"hSize", par.hSize},
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
         * 先认文件签名（keys / rulesVersion），再按 neur 校验 w / b 尺寸和 hSize，
         * 全部通过之后才重建网络状态：
         *   - notNeurUnde = true（否则 Run / Study 会直接报「未初始化」）
         *   - dW / db 按新尺寸清零、h 和 a / z / x 缓存全部清空
         *     （旧缓存是旧权重 / 旧结构算出来的，留着会越界或者算错）
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

            // 梯度累加器跟着新尺寸走；隐状态和 a / z / x 缓存是旧权重 / 旧结构算出来的，全部清掉
            dW.assign(par.w.size(), 0.0f);
            db.assign(par.b.size(), 0.0f);
            insPar.h.assign(par.hSize, 0.0f);
            CleanContext();
            return true;
        }
    };
};  // namespace rnn

#endif  // RNN_CPP
