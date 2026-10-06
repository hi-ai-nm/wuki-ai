// 双层 RNN 架构的 SaveLoad 类

#pragma once
#ifndef LLM_DRNNSL_HPP
#define LLM_DRNNSL_HPP

#include "architecture.hpp"      // Arch::DRNN（同一目录）
#include "../external/json.hpp"  // nlohmann/json（经 -I src/include 解析到 src/external/json.hpp）
#include <fstream>               // std::ifstream / std::ofstream
#include <string>                // std::string
#include <utility>               // std::move / std::swap
#include <vector>                // std::vector

/*
 * 归属 - save 编号 - 0x01：子网还没初始化（没 set_neur / w·b 没按 neur 分配）就保存了
 * 归属 - save 编号 - 0x02：文件打不开 / 写不进去
 * 归属 - save 编号 - 0x03：自己 dump 出来的字符串解析不动了（正常不该发生）
 *
 * 归属 - load 编号 - 0x01：文件打不开 / 不是合法 JSON
 * 归属 - load 编号 - 0x02：不是本程序保存的 drnn 模型（keys / rulesVersion 对不上）
 * 归属 - load 编号 - 0x03：content 缺字段 / 类型不对，或从子网推不出合法的结构参数
 * 归属 - load 编号 - 0x04：内层子网自己判为坏模型（rnn 的 keys / 尺寸对不上）
 * 归属 - load 编号 - 0x05：Init 拒绝了推出来的结构参数（兜底，正常不该出现）
 *
 * 文件格式（沿用 nn / rnn 的 keys + rulesVersion + content 惯例，一个模型就是一个文件）：
 *   keys         : ["wuki", "model", "drnn", "file", "wmdn"]
 *   rulesVersion : 1
 *   content      : {
 *       "allRNN": { "keys": ["wuki","model","rnn","file","wmrn"], "rulesVersion": 1,
 *                   "content": { "w": [...], "b": [...], "neur": [...], "hSize": n, "actFun": "silu" } },
 *       "outRNN": { ...同上，第二个子网... }
 *   }
 *
 * 为什么不另存结构参数（onceAllInputNum / onceAllOutputNum / sliceWidth / ...）：
 *   它们全都推得出来（Init 就是把 hSize 加到 neur 的首尾），存两份就意味着有「两份对不上」
 *   的可能。这里一份都不存，读的时候现推，所以文件里不可能出现「结构参数和子网打架」的状态。
 *
 * 存的是「结构 + 权重 + 激活函数」，不含上下文（h / a / z / x）：读回来是一条新序列的起点。
 *
 * 用法：
 *   Arch::DRNN model;
 *   model.Init(...); model.Train(...);
 *
 *   Arch::DRNNSL saveLoad;                       // 只管存 / 取，不碰训练和推理
 *   if (!saveLoad.Save(model, "Bin/drnn.model.json")) {
 *       std::cout << "Save 失败: " << saveLoad.errorCode << " (" << saveLoad.errorBlgig << ")\n";
 *   }
 *
 *   Arch::DRNN loaded;                           // 读进另一个模型，两个互不影响
 *   if (!saveLoad.Load(loaded, "Bin/drnn.model.json")) {
 *       std::cout << "Load 失败: " << saveLoad.errorCode << " (" << saveLoad.errorBlgig << ")\n";
 *   }
 */

namespace Arch {
    using json = nlohmann::json;  // 别名放进 namespace：和 nn.cpp / rnn.cpp 一个做法

    class DRNNSL {
      public:
        size_t errorCode = 0x00;               // 最近一次的错误码（0x00 表示没有错误）
        std::string errorBlgig = "not error";  // 错误归属（errorBlonging）

        /**
         * 把 model 存进一个文件：两个子网的 w / b / neur / hSize / actFun 全在里面
         *
         * @param model    要保存的模型（没 Init 过会报 0x01 / save-allRnn）
         * @param fileName 目标文件，已存在会被覆盖
         * @return 成功 true；失败 false，原因见 errorCode / errorBlgig
         */
        bool Save(Arch::DRNN& model, const std::string& fileName) {
            errorCode = 0x00;
            errorBlgig = "not error";

            // 先让两个子网各自序列化：没初始化 / w·b 和 neur 对不上时，rnn 那边会返回空串
            // 并写下自己的原因（0x01 / "save"），这里换成更细的归属，方便一眼看出是哪个子网
            const std::string allStr = model.allRNN.ModelToJson();
            if (allStr.empty()) {
                errorCode = 0x01;
                errorBlgig = "save-allRnn";
                return false;
            }
            const std::string outStr = model.outRNN.ModelToJson();
            if (outStr.empty()) {
                errorCode = 0x01;
                errorBlgig = "save-outRnn";
                return false;
            }

            // 两段子网 json 嵌进 content：不直接塞字符串（那会变成一大坨转义文本，
            // 既看不懂也没法 diff）
            json subAll;
            json subOut;
            try {
                subAll = json::parse(allStr);
                subOut = json::parse(outStr);
            } catch (const json::exception&) {  // 自己 dump 出来的字符串解析不动，正常不该发生
                errorCode = 0x03;
                errorBlgig = "save-json";
                return false;
            }

            json contentJson;
            contentJson["allRNN"] = std::move(subAll);
            contentJson["outRNN"] = std::move(subOut);

            json outRoot;
            outRoot["keys"] = {"wuki", "model", "drnn", "file", "wmdn"};
            outRoot["rulesVersion"] = 1;
            outRoot["content"] = std::move(contentJson);

            // 注意顺序：文件到这一步才打开 —— 上面任何一步失败，都不会把已有的好文件截断
            std::ofstream out(fileName, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {  // 路径不存在 / 没权限 / 指向目录
                errorCode = 0x02;
                errorBlgig = "save";
                return false;
            }

            out << outRoot.dump(4) << '\n';
            out.flush();
            if (!out) {  // 磁盘满之类的落盘失败
                errorCode = 0x02;
                errorBlgig = "save";
                return false;
            }
            return true;
        }

        /**
         * 从文件读回模型，直接换进 model
         *
         * 顺序是「先全部验完，最后一次性换」：内层模型读在临时子网上，结构参数反推出来之后再
         * 搭一个临时 DRNN，全部成功才 std::swap 进 model —— 失败的路径上 model 一点没动，
         * 不会出现「权重换了一半」这种状态。
         *
         * @param model    被替换的模型（它的旧权重会被释放）
         * @param fileName 之前 Save 出来的文件
         * @return 成功 true；失败 false，原因见 errorCode / errorBlgig
         */
        bool Load(Arch::DRNN& model, const std::string& fileName) {
            errorCode = 0x00;
            errorBlgig = "not error";

            std::ifstream in(fileName, std::ios::binary);
            if (!in.is_open()) {  // 路径不存在 / 没权限 / 指向目录
                errorCode = 0x01;
                errorBlgig = "load";
                return false;
            }

            json inJson;
            try {
                in >> inJson;
            } catch (const json::exception&) {  // 不是合法 JSON（空文件 / 截断 / 别的东西）
                errorCode = 0x01;
                errorBlgig = "load";
                return false;
            }

            // 认签名：这是本程序存的 drnn 模型吗
            // （nn 的 wmn、rnn 的 wmrn、随便一个 json 数组都会在这里被挡住）
            const json targetKeys = {"wuki", "model", "drnn", "file", "wmdn"};
            if (!inJson.is_object() || !inJson.contains("keys") || !inJson.contains("rulesVersion") ||
                !inJson.contains("content") || inJson["keys"] != targetKeys || inJson["rulesVersion"] != 1) {
                errorCode = 0x02;
                errorBlgig = "load";
                return false;
            }

            const json& content = inJson["content"];
            if (!content.is_object() || !content.contains("allRNN") || !content.contains("outRNN") ||
                !content["allRNN"].is_object() || !content["outRNN"].is_object()) {
                errorCode = 0x03;
                errorBlgig = "load";
                return false;
            }

            // 1. 先把内层模型读进两个临时子网：这一步过了，内层的 keys / 尺寸就都是好的
            wuki::RNN tmpAll;
            wuki::RNN tmpOut;
            if (!tmpAll.ModelFromJson(content["allRNN"].dump(4))) {
                errorCode = 0x04;
                errorBlgig = "load-allRnn";
                return false;
            }
            if (!tmpOut.ModelFromJson(content["outRNN"].dump(4))) {
                errorCode = 0x04;
                errorBlgig = "load-outRnn";
                return false;
            }

            // 2. 用子网的结构把 Init 参数反推回来
            //    存下来的 neur 是 Init 加过 hSize 的，但 hSize **只加在首尾**：
            //    rnn::RNN::Init 里只有 neur[0] += hSize、neur[back] += hSize，
            //    中间层一个都没动 —— 所以只有首尾两段要减回去，中间层原样照抄
            const std::vector<size_t> aNeur = tmpAll.neurons();
            const std::vector<size_t> bNeur = tmpOut.neurons();
            const size_t allH = tmpAll.hSize();
            const size_t outH = tmpOut.hSize();

            // allRNN 至少要「输入层 + 一层隐层 + 输出层」；outRNN 至少要「输入层 + 输出层」
            if (aNeur.size() < 3 || bNeur.size() < 2 ||
                aNeur.front() <= allH || aNeur.back() <= allH ||
                bNeur.front() <= outH || bNeur.back() <= outH) {
                errorCode = 0x03;
                errorBlgig = "load-structure";
                return false;
            }

            const size_t onceAllInputNum = aNeur.front() - allH;
            const size_t onceAllOutputNum = aNeur.back() - allH;
            // 中间那几层本来就没被加过 hSize，直接照抄（减了会下溢成一个天文数字）
            const std::vector<size_t> allRnnNeur(aNeur.begin() + 1, aNeur.end() - 1);

            std::vector<size_t> outRnnNeur = bNeur;  // Init 要的是「不含 h」的完整 neur
            outRnnNeur.front() -= outH;              // 只有输入层这一段挂着 h
            outRnnNeur.back() -= outH;               // 只有输出层这一段挂着 h

            // 这几条 Init 也会查，这里提前查：结构是从文件里读来的（不一定是自己存的），
            // 早点报出来归属更准，也免得白搭一个临时模型
            if (onceAllInputNum == 0 || onceAllOutputNum == 0 ||
                outRnnNeur.front() == 0 || outRnnNeur.back() == 0 ||
                onceAllOutputNum % outRnnNeur.front() != 0) {
                errorCode = 0x03;
                errorBlgig = "load-structure";
                return false;
            }

            // 3. 全部验完了，才动目标：先搭临时的，最后一次性换进去
            Arch::DRNN tmp;
            if (tmp.Init(outRnnNeur, allRnnNeur, onceAllInputNum, onceAllOutputNum, outH, allH) != 0x00) {
                errorCode = 0x05;  // 上面都查过了，这里还能被拒就是结构本身不合法（兜底）
                errorBlgig = "load-init";
                return false;
            }

            // Init 刚随机初始化出来的两个子网，立刻被加载好的换掉：
            // 结构参数仍然只由 Init 一处决定（sliceWidth / sliceNum / outWidth 都是它推的），
            // 权重用移动赋值搬进去，不用再解析一遍 json
            tmp.allRNN = std::move(tmpAll);
            tmp.outRNN = std::move(tmpOut);

            std::swap(model, tmp);  // 只有走到这里，model 才会被换掉（旧的权重随 tmp 一起释放）
            return true;
        }
    };
}  // namespace Arch

#endif  // LLM_DRNNSL_HPP
