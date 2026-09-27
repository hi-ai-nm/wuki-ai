#pragma once
#ifndef NN_CPP
#define NN_CPP

#include "nn.hpp"
#include <cmath>
#include <fstream>  // std::ifstream / std::ofstream
#include <utility>  // std::move
#include "../external/json.hpp"

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
    // JSON 别名放在 namespace 里：includeMain.cpp 先 include nn.cpp 再 include rnn.cpp，
    // 写在全局作用域的话两个文件会重定义（rnn.cpp 里也有一个同名别名）。
    using json = nlohmann::json;
    // 激活函数名（原为 _ActFun_* 宏；宏不受 namespace 约束，改为 namespace 内常量）
    static constexpr const char* ActFun_ReLU = "relu";
    static constexpr const char* ActFun_SiLU = "silu";
    static constexpr const char* Alternative_ActFun_Sigmoid = "alt-sigmoid";

    struct Pars {  // 参数
        std::vector<float> w;
        std::vector<float> b;

        std::string actFun = ActFun_ReLU;  // 激活函数

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
            std::vector<float> newActVal;
            size_t wIndex = 0;
            size_t bIndex = 0;

            // 运行
            for (size_t nol = 1; nol < par.neur.size(); nol++) {
                newActVal.clear();
                for (size_t cnn = 0; cnn < par.neur[nol]; cnn++) {
                    float sum = 0;  // 加权求和
                    for (size_t nonitpl = 0; nonitpl < par.neur[nol - 1]; nonitpl++) {
                        sum += par.w[wIndex] * oldActVal[nonitpl];
                        wIndex++;
                    }

                    // 计算 z（加权求和 + 偏置）
                    float z = sum + par.b[bIndex];
                    bIndex++;

                    // 计算 a（激活函数）
                    float a;
                    if (par.actFun == ActFun_ReLU) {
                        a = z > 0 ? z : 0;
                    } else if (par.actFun == ActFun_SiLU) {
                        a = z / (1.0f + expf(-z));
                    } else if (par.actFun == Alternative_ActFun_Sigmoid) {
                        a = 1.0f / (1.0f + expf(-z));
                    } else {
                        a = z > 0 ? z : 0;  // 默认 ReLU
                    }

                    // 存储瞬时参数
                    insPar.z.push_back(z);
                    insPar.a.push_back(a);

                    newActVal.push_back(a);
                }
                oldActVal = newActVal;
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
                            x = InteUtiFun::RandomRange(gen, 0.3f, 0.6f);  // 未超过，使用随机权重
                        }
#else
                        x = InteUtiFun::RandomRange(gen, 0.3f, 0.6f);
#endif
                        par.w[wIndex] = x;
                        wIndex++;
                        connectCount++;
                    }
                }
            }

            // 初始化偏置 b
            for (auto& b : par.b) {
                b = InteUtiFun::RandomRange(gen, 0.3f, 0.6f);
            }
            return 0x00;
        }

        // 设置层数
        void set_neur(const std::vector<size_t>& neur, bool autoInit = true) {
            par.neur = neur;
            if (autoInit) {
                Init(par.neur);
            }
            par.notNeurUnde = true;
        }

        void set_neur(std::initializer_list<size_t> neur, bool autoInit = true) {
            par.neur = neur;
            if (autoInit) {
                Init(par.neur);
            }
            par.notNeurUnde = true;
        }

        // 设置激活函数
        void set_ActFun(std::string content) {
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

            // ==================== 第一步：铺开目标（输出层） ====================
            float ero = 0;  // MSE 误差
            // 输出层的目标激活值就是 goal（误差 = goal - 实际激活值，留给第三步统一减）
            for (size_t i = 0; i < par.neur[lastLayer]; i++) {
                error[outputStart + i] = goal[i];
                ero += (goal[i] - insPar.a[outputStart + i]) * (goal[i] - insPar.a[outputStart + i]);
            }

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
            for (size_t l = lastLayer; l >= 2; l--) {
                const size_t curStart = aStart[l];
                const size_t prevStart = aStart[l - 1];
                const size_t curSize = par.neur[l];
                const size_t prevSize = par.neur[l - 1];

                for (size_t nonitpl = 0; nonitpl < prevSize; nonitpl++) {  // 目标层的每个神经元
                    // 计算 Σ|w|：这个神经元连出去的权重绝对值之和
                    float sumAbsW = 0.0f;
                    for (size_t cnn = 0; cnn < curSize; cnn++) {
                        sumAbsW += std::abs(par.w[wStart[l] + cnn * prevSize + nonitpl]);
                    }

                    if (sumAbsW == 0.0f) continue;  // 全 0 就分配不了（不然除出 inf / nan）

                    float target = 0.0f;
                    for (size_t cnn = 0; cnn < curSize; cnn++) {
                        const float wVal = par.w[wStart[l] + cnn * prevSize + nonitpl];
                        target += static_cast<float>(prevSize) * (wVal / sumAbsW) * error[curStart + cnn];
                    }
                    error[prevStart + nonitpl] = target;
                }
            }

            // ==================== 第三步：目标 → 误差 ====================
            // 误差 = 目标激活值 - 实际激活值（都在激活值空间里算）
            //
            // 原来是「先按 relu 反解出目标 z（targetZ = max(目标, 0)），再减实际 z」：
            // 那等于把激活函数写死成 relu —— actFun 换成 silu / alt-sigmoid 时反解就是错的
            // （silu 还没有解析反函数），而 rnn 那边一直是在激活值空间相减，这里对齐
            for (size_t i = 0; i < error.size(); i++) {
                error[i] -= insPar.a[i];
            }

            // ==================== 第四步：用误差更新 w 和 b（从输出层往第 1 层走） ====================
            for (size_t l = lastLayer; l >= 1; l--) {
                const size_t curStart = aStart[l];
                const size_t prevStart = aStart[l - 1];
                const size_t curSize = par.neur[l];
                const size_t prevSize = par.neur[l - 1];

                for (size_t cnn = 0; cnn < curSize; cnn++) {
                    size_t wIdx = wStart[l] + cnn * prevSize;

                    // 更新 w
                    for (size_t nonitpl = 0; nonitpl < prevSize; nonitpl++) {
                        // 第 1 层的前一层是输入层：insPar.a 里没有输入层，
                        // a[prevStart + nonitpl] 其实是第 1 层自己的激活值，
                        // 这里要用 Run 记下来的真实输入
                        const float prevAct = (l == 1) ? insPar.x[nonitpl] : insPar.a[prevStart + nonitpl];
                        par.w[wIdx] += wlr * error[curStart + cnn] * prevAct;
                        wIdx++;
                    }

                    // 更新 b
                    par.b[curStart + cnn] += blr * error[curStart + cnn];
                }
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
        bool SaveModel(const std::string& fileName) {
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
        bool LoadModel(const std::string& fileName) {
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
