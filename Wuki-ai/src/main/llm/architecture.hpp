#pragma once
#ifndef LLM_ARCHITECTURE_HPP
#define LLM_ARCHITECTURE_HPP

#include "framework.hpp"
#include <string>   // std::string（errorBlgig）
#include <utility>  // std::pair

namespace Arch {
    class DRNN {   // 自创架构 DoubleRNN 先由一个 RNN 预测后面所有，再由另一个 RNN 预测真正的输出
        struct ArchDatas {
            std::vector<size_t> allRnnNeur;  // allRNN 的每层神经元数量（不含输入层和输出层）
            std::vector<size_t> outRnnNeur;  // outRNN 的完整每层神经元数量（含输入层和输出层，不含 h）
            size_t allRnnHSize = 32;  // allRNN 的隐状态大小（allRNN 输出里尾部的 h 段长度）
            size_t outRnnHSize = 16;  // outRNN 的隐状态大小
            size_t onceAllInputNum = 32;   // allRNN 一次性喂进来的输入数量（不含 h）
            size_t onceAllOutputNum = 32;  // allRNN 一次性产出的输出数量（不含 h）
            // 下面三个由 outRnnNeur 推出来，免得 Run / Train 每次自己算
            size_t sliceWidth = 0;  // 每片宽度 == outRnnNeur[0] == outRNN 的输入层宽度（不含 h）
            size_t sliceNum = 1;    // allRNN 一次产出要切成几片 == onceAllOutputNum / sliceWidth
            size_t outWidth = 0;    // DRNN 最终输出宽度 == outRnnNeur.back()（不含 h）
        };

        wuki::RNN allRNN;  // 预测后续所有的 RNN
        wuki::RNN outRNN;  // 预测真正输出的 RNN
        ArchDatas data;    // 结构参数

        // 存 / 取（DRNNSL，定义在 DRNNSL.hpp）需要直接读写上面两个子网和 data：
        // 库外只能通过 wuki::RNN 的 ModelToJson / ModelFromJson / neurons / hSize 看子网，
        // 拿不到成员本身，所以只能开友元，而不是把这些内部成员暴露成 public
        friend class DRNNSL;

        /*
         * 把「按步给的输入」压平成一条浮点序列，再按 onceAllInputNum 切成块；最后一块不足的用 0 补满
         *
         * 为什么先压平：rnn::Run 要求输入宽度精确等于输入层宽度（par.neur[0] - hSize），
         * 而调用方给的是「每步若干个数」，步宽和 onceAllInputNum 不一定能整除。
         *
         * 切点是按浮点个数算的，可能落在某一「步」中间 —— DRNN 只关心整条序列，不关心步边界。
         *
         * 修掉的 bug（原来 Run / Train 各抄了一份）：
         *   - 拿「第几个输入向量」（idx % onceAllInputNum）当宽度算，步宽不是 1 就算错；
         *   - 尾巴从 plrun 重新拼一遍，把已经喂过的那一步又算了一次（重复计数）；
         *   - 只要跑过一块，`if (!run)` 就为假，尾巴整段被丢掉，连补 0 都没有。
         */
        inline std::vector<std::vector<float>> chunkInput(const std::vector<std::vector<float>>& input) const {
            std::vector<float> flat;
            for (const auto& step : input) {
                flat.insert(flat.end(), step.begin(), step.end());
            }

            std::vector<std::vector<float>> chunks;
            for (size_t b = 0; b < flat.size(); b += data.onceAllInputNum) {
                size_t end = b + data.onceAllInputNum;
                if (end > flat.size()) end = flat.size();  // 不用 std::min，免得为它加 <algorithm>
                std::vector<float> chunk(flat.begin() + b, flat.begin() + end);
                chunk.resize(data.onceAllInputNum, 0.0f);  // 尾巴补 0，凑满一块
                chunks.push_back(std::move(chunk));
            }
            return chunks;  // 空输入 → 空列表（调用方按错误处理）
        }

        /*
         * 逐块跑 allRNN，块之间 h 连续（同一次 DRNN 调用算一条序列），结果写进 allOut
         *
         * @return 错误码：0x00 成功；0x03 某一块的 allRNN.Run 失败了（宽度对不上）
         */
        inline size_t runAllRnn(const std::vector<std::vector<float>>& chunks,
                                std::vector<std::vector<float>>& allOut) {
            allOut.clear();
            allRNN.CleanContext();  // 一条序列从 h = 0 开始（块之间不清，h 要一路传下去）

            for (const auto& chunk : chunks) {
                std::vector<float> one = allRNN.Run(chunk);
                // allRNN.Run 正常返回的宽度 = 输出层真实宽度 + 它自己的 h 段
                if (one.size() != data.onceAllOutputNum + data.allRnnHSize) {
                    errorCode = 0x03;
                    errorBlgig = "run-allRnn";
                    return errorCode;
                }
                allOut.push_back(std::move(one));
            }
            return 0x00;
        }

        /*
         * 丢掉 allRNN 自己尾部的 allRnnHSize 个 h（它不属于 DRNN 的语义输出），
         * 再按 sliceWidth 切成 sliceNum 片，每片正好是 outRNN 的输入层宽度
         */
        inline std::vector<std::vector<float>> sliceAllOut(const std::vector<float>& oneAllOut) const {
            std::vector<std::vector<float>> pieces;
            pieces.reserve(data.sliceNum);
            for (size_t s = 0; s < data.sliceNum; s++) {
                const size_t b = s * data.sliceWidth;
                pieces.emplace_back(oneAllOut.begin() + b, oneAllOut.begin() + b + data.sliceWidth);
            }
            return pieces;
        }

      public:
        size_t errorCode = 0x00;               // 最近一次的错误码（0x00 表示没有错误）
        std::string errorBlgig = "not error";  // 错误归属（哪个阶段 / 哪个子网出的错）

        inline size_t Init(   // 返回错误码
            const std::vector<size_t>& outRnnNeur, /* outRNN 的完整 neur：含输入层和输出层，不含 h */
            const std::vector<size_t>& allRnnNeur = { 48 , }, /* 不含输入层和输出层 */
            const size_t onceAllInputNum = 32,
            const size_t onceAllOutputNum = 32,
            const size_t outRnnHSize = 16, const size_t allRnnHSize = 32
        ) {
            errorCode = 0x00;
            errorBlgig = "not error";

            // 参数校验：outRNN 的输入层宽度必须能整切 onceAllOutputNum，否则 Run / Train
            // 一定会喂进宽度对不上的数据（这正是之前 Result 为空的根因），宁可在 Init 就报出来
            if (outRnnNeur.size() < 2 || allRnnNeur.size() < 1 ||
                onceAllInputNum == 0 || onceAllOutputNum == 0 ||
                outRnnNeur.front() == 0 || outRnnNeur.back() == 0 ||
                onceAllOutputNum % outRnnNeur.front() != 0) {
                errorCode = 0x01;
                errorBlgig = "init";
                return errorCode;
            }

            std::vector<size_t> allNeur = { onceAllInputNum };
            allNeur.insert(allNeur.end(), allRnnNeur.begin(), allRnnNeur.end());
            allNeur.push_back(onceAllOutputNum);

            allRNN.set_neur(allNeur, allRnnHSize);
            // outRNN 的 neur 由调用方给全（含输入层和输出层）：首项就是 allRNN 一次产出的宽度，
            // 末项是 DRNN 最终输出宽度；hSize 由 set_neur → Init 自己加到首尾，这里不要自己加 h！
            outRNN.set_neur(outRnnNeur, outRnnHSize);

            // 无需初始化，set_neur 会自动初始化 w 和 b

            // 逐字段赋值：{ ... } 聚合初始化一加字段就会静默错位
            data.allRnnNeur = allRnnNeur;
            data.outRnnNeur = outRnnNeur;
            data.allRnnHSize = allRnnHSize;
            data.outRnnHSize = outRnnHSize;
            data.onceAllInputNum = onceAllInputNum;
            data.onceAllOutputNum = onceAllOutputNum;
            data.sliceWidth = outRnnNeur.front();
            data.sliceNum = onceAllOutputNum / outRnnNeur.front();
            data.outWidth = outRnnNeur.back();

            return errorCode;  // 0x00
        }

        /*
         * 前向：allRNN 分块预测出「后面所有的输出」→ 每块的产出丢掉 h 段后切成 sliceNum 片
         * → 依次喂给 outRNN（片与片之间 h 连续）→ 最后一片的真实输出段就是结果
         *
         * 一次调用算一条序列（两个子网各自先 CleanContext）
         *
         * @return { 结果（宽度 = outRNN 输出层真实宽度）, 错误码 }，错误说明见 errorCode / errorBlgig
         */
        inline std::pair<std::vector<float>, size_t> Run(std::vector<std::vector<float>> input) {
            errorCode = 0x00;
            errorBlgig = "not error";

            std::vector<std::vector<float>> chunks = chunkInput(input);
            if (chunks.empty()) {
                errorCode = 0x02;
                errorBlgig = "run-emptyInput";
                return { {}, errorCode };  // 别再去 allOut.back()（原来是 UB）
            }

            // 先用 allRNN 预测后续所有的输出
            std::vector<std::vector<float>> allOut;
            if (size_t c = runAllRnn(chunks, allOut)) {
                return { {}, c };
            }

            // 再用 outRNN 预测真正的输出
            outRNN.CleanContext();  // 一条序列从 h = 0 开始
            std::vector<float> out;
            for (const auto& one : allOut) {
                for (auto& piece : sliceAllOut(one)) {
                    out = outRNN.Run(piece);
                    if (out.size() != data.outWidth + data.outRnnHSize) {  // 宽度对不上就不能当结果
                        errorCode = 0x03;
                        errorBlgig = "run-outRnn";
                        return { {}, errorCode };
                    }
                }
            }
            out.resize(data.outWidth);  // 丢掉 outRNN 自己的 h 段，只留真正的输出

            return { out , errorCode };  // 0x00 表示没有错误
        }

        /*
         * 训练：先用 allRNN 预测（和 Run 完全同一套前向）→ 再用 outRNN 学真正的输出
         * → 最后用 outRNN 回传的输入层目标反过来训 allRNN
         *
         * 目标约定：只有最后一片有外部目标（targets），前面的片拿「该片实际输出」当目标，
         * 表示这一步不施加外部监督 —— 注意这不是严格的 0 误差步：RNN 输出层的 h 段目标
         * （hTar，由后一步反推上来）仍然会造成改动。
         *
         * @param targets 最后一片的目标，宽度必须等于 DRNN 的输出层真实宽度
         * @return { outRNN 逐步的 MSE 误差（长度 = 片数）, 错误码 }；
         *         出错时第一个元素可能是空 vector，原因见 errorCode / errorBlgig
         */
        inline std::pair<std::vector<float>, size_t> Train(
            std::vector<std::vector<float>> input, std::vector<float> targets,
            const float wlr = 0.01f, const float blr = 0.01f
        ) {
            errorCode = 0x00;
            errorBlgig = "not error";

            // 目标宽度必须和 DRNN 的输出层真实宽度一致（rnn::Train 也会查，这里查能给出更准的归属）
            if (targets.size() != data.outWidth) {
                errorCode = 0x01;
                errorBlgig = "train-targetWidth";
                return { {}, errorCode };
            }

            std::vector<std::vector<float>> chunks = chunkInput(input);
            if (chunks.empty()) {
                errorCode = 0x02;
                errorBlgig = "train-emptyInput";
                return { {}, errorCode };
            }

            // 先跑 allRNN（训练看到的输入必须和推理看到的完全一样）
            std::vector<std::vector<float>> allOut;
            if (size_t c = runAllRnn(chunks, allOut)) {
                return { {}, c };
            }

            // outRNN 的输入：每一块的产出都丢掉 h 段、切成 sliceNum 片，再按顺序连起来
            std::vector<std::vector<float>> pieces;
            for (const auto& one : allOut) {
                for (auto& piece : sliceAllOut(one)) {
                    pieces.push_back(std::move(piece));
                }
            }

            // 每一步的目标：最后一片用外部目标，前面的片用该片实际输出
            std::vector<std::vector<float>> goals(pieces.size());
            if (pieces.size() == 1) {
                goals[0] = targets;  // 只有一片，用不着为了取实际输出多跑一次
            } else {
                outRNN.CleanContext();
                for (size_t t = 0; t < pieces.size(); t++) {
                    std::vector<float> actual = outRNN.Run(pieces[t]);
                    // rnn::Train 内部会自己 CleanContext 再 Run 一遍，权重没变 → 实际输出和这里一致
                    if (actual.size() != data.outWidth + data.outRnnHSize) {
                        errorCode = 0x03;
                        errorBlgig = "train-outRnnActual";
                        return { {}, errorCode };
                    }
                    goals[t].assign(actual.begin(), actual.begin() + data.outWidth);  // 实际输出段
                }
                goals.back() = targets;  // 最后一片才是真正的监督
            }

            // 把整段目标和输入交给 outRNN 学（goals 在前、inputs 在后）
            std::vector<float> loss = outRNN.Train(goals, pieces, wlr, blr);
            if (loss.empty()) {
                errorCode = 0x03;
                errorBlgig = "train-outRnn";
                return { {}, errorCode };
            }

            // outRNN 回传的输入层目标（第 0 步那一份）：本该就是 allRNN 该产出的东西
            std::vector<float> allRnnTarget = outRNN.obtainInputTarget();
            if (allRnnTarget.size() != data.onceAllOutputNum) {
                // 切片数 > 1 时它只有「第一片」那么宽，拼不出 allRNN 需要的整段（宽 onceAllOutputNum）
                // → 不补 0 假装成功，直接报出来（完整支持要给 wuki::RNN 加 obtainInputTargets()）
                errorCode = 0x04;
                errorBlgig = "train-allRnnTarget";
                return { loss, errorCode };
            }

            // allRNN 的目标：第 0 块用 outRNN 回传的真目标，其余块用该块实际产出（丢掉 h 段）
            std::vector<std::vector<float>> allRnnGoals(allOut.size());
            for (size_t t = 0; t < allOut.size(); t++) {
                allRnnGoals[t].assign(allOut[t].begin(), allOut[t].begin() + data.onceAllOutputNum);
            }
            allRnnGoals[0] = allRnnTarget;

            // 用 allRNN 训练（同样是 goals 在前、inputs 在后；每块宽 onceAllInputNum、目标宽 onceAllOutputNum）
            if (allRNN.Train(allRnnGoals, chunks, wlr, blr).empty()) {
                errorCode = 0x03;
                errorBlgig = "train-allRnn";
                return { loss, errorCode };
            }

            return { loss , errorCode };  // 0x00：loss 是 outRNN 逐步的 MSE 误差
        }
    };
};

#endif
