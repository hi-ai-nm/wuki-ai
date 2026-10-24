// Word embedding tools
// 两个职责：
//   1. 查表：token id → 词向量；反向反查 Nearest / Sample（带温度采样）还原 token id
//   2. 训练：Train（Skip-gram）用语料的共现关系学这张表 —— word2vec 的思想，
//      中心词预测上下文，正样本拉近、负采样推远；逐对更新词向量，不走网络
#ifndef WET_HPP
#define WET_HPP

#include "framework.hpp"
#include "optimize/simd.hpp"  // Simd::Runtime：Skip-gram 点积 / 最近邻距离 / 归一化
#include <cstddef>  // std::size_t
#include <cmath>    // std::exp（sigmoid）
#include <vector>
#include <random>

class WorEmbd {
  public:
    /**
     * 初始化词向量表：tokenSize 个 token，每个 outputSize 维，均匀分布在 [min, max]
     *
     * 重复调用会重新生成一整张新表（旧表作废）—— 词表变了（BBPE 又学了新的 merge）
     * 就得重新来，不然新 token 取不到向量、旧向量也和词表对不上号。
     *
     * @return 0 成功；1 参数非法（tokenSize 或 outputSize 为 0）
     */
    inline size_t Init(const size_t tokenSize, const size_t outputSize, const float min = -0.7f, const float max = 0.7f) {
        if (tokenSize == 0 || outputSize == 0) {
            return 1;  // 空词表 / 0 维向量：没有可用的东西
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        target.assign(tokenSize, std::vector<float>(outputSize));
        for (auto& vec : target) {
            for (auto& value : vec) {
                value = dist(gen);
            }
        }
        inited = true;
        return 0;
    }

    /// 是否已经初始化过（没初始化时 Convert / Get 都取不到东西）
    inline bool initialized() const {
        return inited;
    }

    /// 词表大小（token 个数）
    inline size_t tokenSize() const {
        return target.size();
    }

    /// 词向量维度（没初始化时是 0）
    inline size_t embdSize() const {
        return target.empty() ? 0 : target.front().size();
    }

    /**
     * 单个 token id → 词向量
     *
     * @return 词向量；没初始化 / id 越界返回空 vector（调用方按「取不到」处理）
     */
    inline std::vector<float> Get(const size_t id) const {
        if (!inited || id >= target.size()) {
            return {};
        }
        return target[id];
    }

    /**
     * 整张词向量表（存权重用）：只读引用，调用方别改它
     *
     * 为什么需要「整张表」而不是一行行 Get：这张表是训练出来的、不是算出来的，
     * 存权重时必须原样带出去 —— 只存 DRNN 的话，下次启动 Init 会用 std::random_device
     * 重新随机生成一整张表，RNN 拿到的就是噪声（线上表现：每次重启回答都不一样、而且都是乱码）。
     *
     * @return 表本身（没 Init 过是空表）
     */
    inline const std::vector<std::vector<float>>& Rows() const {
        return target;
    }

    /**
     * 整张词向量表（读权重用）：把存下来的表放回来
     *
     * 只在「表还是刚 Init 出来的随机值、没训过」的时候用（Load 之后直接接推理）。
     * 尺寸必须和当前词表严丝合缝：行数 != tokenSize 说明语料 / merge 轮数变了，
     * 列数 != embdSize 说明结构变了 —— 两种情况都对不上号，宁可整体失败让上层重训，
     * 也不能让半张表（前几行对、后面全是随机）混进来，那种错查起来最费劲。
     *
     * @param rows 存的表
     * @return 0 成功；1 还没 Init（不知道目标尺寸）；2 尺寸对不上 / 空表
     */
    inline size_t SetRows(const std::vector<std::vector<float>>& rows) {
        if (!inited) {
            return 1;
        }
        if (rows.empty() || rows.size() != target.size() || rows.front().size() != embdSize()) {
            return 2;
        }
        for (const auto& row : rows) {  // 逐行再核一遍列数：别只信第一行
            if (row.size() != embdSize()) {
                return 2;
            }
        }
        target = rows;
        return 0;
    }

    /**
     * 把每一行缩放到同一个范数（默认 1），且不改方向
     *
     * 为什么需要：Skip-gram 的更新是「拉近 / 推远」，只有方向有约束、长度没有 ——
     * 正样本每步都在加，跑几千轮之后一行的范数能长到 4~5。而 DRNN 的监督目标是
     * 「下一个 token 的词向量」，输出层却用饱和激活（scth 每个分量上限 ±2）。
     * 目标范数 4.6 对上输出能力 ±2，等于让它去够一个物理上够不到的点：
     * 实测不可消除的 MSE 地板 0.27~2.67，训练 loss 永远压不到 0，看着像"训不动"。
     *
     * 归一化之后所有 token 都在同一个球面上：比较变成纯夹角比较（最近邻 / 余弦），
     * 输出层的可达半径也回到了 ±1（每个分量最大 1），scth 的 ±2 有了余量。
     *
     * @param norm 目标范数（必须 > 0）；每一行缩放成 |v| = norm
     * @return 0 成功；1 还没 Init（没表可归一）；2 norm 不合法
     */
    inline size_t Normalize(const float norm = 1.0f) {
        if (!inited) {
            return 1;
        }
        if (norm <= 0.0f) {
            return 2;
        }
        for (auto& vec : target) {
            // 模长平方一次 SIMD 扫描；缩放也只走一遍向量
            const float len = std::sqrt(Simd::Runtime::sum_sq(vec.data(), vec.size()));
            if (len == 0.0f) {
                continue;  // 全 0 行没有方向可言，跳过（也免得除 0）
            }
            Simd::Runtime::scale_inplace(vec.data(), norm / len, vec.size());
        }
        return 0;
    }

    /**
     * 一批 token id → 一批词向量
     *
     * 任何一个 id 越界都整体返回空：给半截向量的话，调用方拿到的是「长度对得上、
     * 内容错位」的序列，比直接失败更难查。
     *
     * @return 和 input 等长的一批词向量；没初始化 / 有 id 越界返回空
     */
    inline std::vector<std::vector<float>> Convert(const std::vector<size_t>& input) const {
        if (!inited) {
            return {};  // 如果没有初始化，直接返回空
        }

        std::vector<std::vector<float>> output;
        output.reserve(input.size());
        for (const size_t id : input) {
            if (id >= target.size()) {
                return {};  // 超出词表：整体失败
            }
            output.push_back(target[id]);
        }

        return output;
    }

    /**
     * 反向查找：把一个「算出来的词向量」还原成最接近的 token id（欧氏距离最近）
     *
     * 生成时要用：DRNN 吐出来的是词向量，要接着往下喂就得先知道它是哪个 token。
     *
     * @return 最近的 token id；没初始化 / 向量维度对不上返回 tokenSize()（越界值，
     *         和任何合法 id 都分得开）
     */
    inline size_t Nearest(const std::vector<float>& vec) const {
        std::vector<size_t> all(target.size());
        for (size_t i = 0; i < all.size(); ++i) {
            all[i] = i;
        }
        return Nearest(vec, all);
    }

    /**
     * 只在一批候选 id 里找最近的
     *
     * 为什么要有这个：词表里有大量语料中根本没出现过的 token（256 个字节 token、
     * 特殊 token），它们的向量没经过训练、基本还是随机初始化时的样子，全词表找
     * 最近邻很容易挑到它们（生成出来的就是乱码）。生成时把候选限成
     * 「语料里出现过的 token」，模型就只能在这些 token 里挑 —— 这正是它能学到的东西。
     *
     * @return 候选里最近的 token id（候选按出现顺序返回第一个最近的）；
     *         没初始化 / 维度对不上 / 候选为空时返回 tokenSize()（越界哨兵）
     */
    inline size_t Nearest(const std::vector<float>& vec, const std::vector<size_t>& candidates) const {
        if (!inited || vec.empty() || vec.size() != embdSize() || candidates.empty()) {
            return target.size();
        }

        size_t bestId = target.size();
        float bestDist = 0.0f;
        for (const size_t id : candidates) {
            if (id >= target.size()) {
                continue;  // 候选里有越界 id：跳过（这里不整体失败，生成时少一个候选没什么）
            }

            const float dist = Simd::Runtime::distance_sq(target[id].data(), vec.data(), vec.size());
            if (bestId == target.size() || dist < bestDist) {
                bestDist = dist;
                bestId = id;
            }
        }
        return bestId;
    }

    /**
     * 带温度的反向查找：距离转 softmax 概率采样，而不是总取最近的那个（Nearest 是 T→0 的特例）
     *
     * 权重 w_i = exp(−(d_i − d_min) / T)：离最优候选的平方距离差越大，被采到的概率
     * 越低；减掉 d_min 是数值稳定（最优候选权重恒为 1，exp 不溢出）。
     * T → 0 退化成 Nearest（贪心）；T 越大分布越平，越容易采到次近的 token。
     *
     * 注意 T 的量纲是「平方距离」，合理取值取决于词向量的实际尺度
     * （Init 的 [min, max] × 维度），没有通用值，按生成效果调。
     *
     * @param temperature 温度；<= 0 时直接退化为 Nearest（贪心）
     * @param gen         采样用的随机源（由调用方持有，方便复现/控制）
     * @return 采到的 token id；入口检查和 Nearest 相同，失败返回 tokenSize()（越界哨兵）
     */
    inline size_t Sample(const std::vector<float>& vec, const std::vector<size_t>& candidates,
                         const float temperature, std::mt19937& gen) const {
        if (!inited || vec.empty() || vec.size() != embdSize() || candidates.empty()) {
            return target.size();
        }
        if (temperature <= 0.0f) {
            return Nearest(vec, candidates);  // 零温 = 贪心
        }

        std::vector<std::pair<size_t, float>> scored;
        scored.reserve(candidates.size());
        float minDist = 0.0f;
        for (const size_t id : candidates) {
            if (id >= target.size()) {
                continue;  // 和 Nearest 一致：越界候选跳过
            }
            const float dist = Simd::Runtime::distance_sq(target[id].data(), vec.data(), vec.size());
            if (scored.empty() || dist < minDist) {
                minDist = dist;
            }
            scored.emplace_back(id, dist);
        }
        if (scored.empty()) {
            return target.size();  // 候选全是越界 id
        }

        std::vector<float> weights(scored.size());
        float sum = 0.0f;
        for (size_t i = 0; i < scored.size(); ++i) {
            weights[i] = std::exp(-(scored[i].second - minDist) / temperature);
            sum += weights[i];
        }

        std::uniform_real_distribution<float> pick(0.0f, sum);
        float acc = 0.0f;
        const float r = pick(gen);
        for (size_t i = 0; i < scored.size(); ++i) {
            acc += weights[i];
            if (r <= acc) {
                return scored[i].first;
            }
        }
        return scored.back().first;  // 浮点累加误差兜底
    }

    // ------------------------------------------------------------------
    // Skip-gram 训练（word2vec 的思想：中心词预测上下文）
    //
    // 和原版 word2vec 的两处简化，都是为小而做的：
    //   - 一张表同时当输入向量和输出向量（W 和 W' 合并）：语料小，效果一样，
    //     而且 Get / Nearest 的语义不用动
    //   - 负样本从整张词表均匀采（原版用 unigram^0.75）：语料太小，分不出差别
    //
    // 不走网络、不碰 BP（项目规矩）：全部数学就是逐对 delta rule —
    //   正样本 (label=1)：g = +lr·(1 − σ(va·vb))，va += g·vb，vb += g·va（拉近）
    //   负样本 (label=0)：g = −lr·σ(va·vb)，同上（推远）
    // σ 是 sigmoid；两个向量同时更新、各自都用对方的旧值。
    // ------------------------------------------------------------------

    /// sigmoid，截断在 ±20 防 exp 溢出
    inline static float Sigmoid(const float x) {
        if (x > 20.0f) return 1.0f;
        if (x < -20.0f) return 0.0f;
        return 1.0f / (1.0f + std::exp(-x));
    }

    /*
     * 一对 (a, b) 的更新；label = 1 是正样本（拉近），0 是负样本（推远）
     * 调用方保证 a != b（负采样里会跳过中心自己，自己推自己会一行向量两头更新打架）
     *
     * 两个分量的更新互相依赖（va += g·vb 要用旧 vb，vb += g·va 要用旧 va），
     * 原来为此整份拷贝一份 va 再循环；其实按分量各留一个标量旧值就够了 ——
     * 每对省一次堆分配 + 一份向量拷贝。Skip-gram 阶段是几十亿次调用，省下的时间可观。
     */
    inline void SgdPair(const size_t a, const size_t b, const float lr, const float label) {
        std::vector<float>& va = target[a];
        std::vector<float>& vb = target[b];
        // 点积和「两边各自用对方旧值」的耦合更新都是纯逐元素运算，整对交给 SIMD
        const float g = lr * (label - Sigmoid(Simd::Runtime::dot(va.data(), vb.data(), va.size())));
        Simd::Runtime::paired_update(va.data(), vb.data(), g, va.size());
    }

    /*
     * 一对（中心，上下文）+ 若干负样本的一步更新
     * id 越界的直接跳过（负采样采到表外不可能，防御而已）
     */
    inline void SkipGramOne(const size_t center, const size_t context, const std::vector<size_t>& negatives, const float lr) {
        if (!inited || center >= target.size() || context >= target.size()) {
            return;
        }
        SgdPair(center, context, lr, 1.0f);  // 正样本：拉近
        for (const size_t n : negatives) {
            if (n >= target.size() || n == center) {
                continue;
            }
            SgdPair(center, n, lr, 0.0f);  // 负样本：推远
        }
    }

    /*
     * 整段语料的 Skip-gram 训练：每个位置当中心，向左右各看 window 个当上下文
     *
     * @param sentences 已编码的 id 序列（一条句子一个序列；Encode 编不出来的句子
     *                  调用方别塞进来，白跑）
     * @return 0 成功；1 还没 Init（没表可训）
     */
    inline size_t Train(const std::vector<std::vector<size_t>>& sentences,
                        const size_t window = 2, const size_t epochs = 1,
                        const size_t negNum = 5, const float lr = 0.05f) {
        if (!inited) {
            return 1;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> negDist(0, target.size() - 1);

        for (size_t ep = 0; ep < epochs; ++ep) {
            for (const auto& sent : sentences) {
                for (size_t i = 0; i < sent.size(); ++i) {
                    const size_t lo = i >= window ? i - window : 0;
                    const size_t hi = i + window + 1 < sent.size() ? i + window + 1 : sent.size();  // 不用 std::min
                    for (size_t j = lo; j < hi; ++j) {
                        if (j == i) {
                            continue;  // 中心自己不算自己的上下文
                        }
                        std::vector<size_t> negs;
                        negs.reserve(negNum);
                        for (size_t k = 0; k < negNum; ++k) {
                            const size_t n = negDist(gen);
                            if (n != sent[j]) {  // 别把正样本当负样本拉
                                negs.push_back(n);
                            }
                        }
                        SkipGramOne(sent[i], sent[j], negs, lr);
                    }
                }
            }
        }
        return 0;
    }

  private:
    // id 是连续的（0 ~ tokenSize-1）→ 直接用 vector 按下标取，
    // 原来用 map<size_t, vector<float>>：operator[] 在缺 key 时会静默插入一个空向量，
    // Convert 就会「悄悄给出一个空词向量」而不是报错；size() 也只是「有几个 key」，
    // 和「词表多大」并不等价
    std::vector<std::vector<float>> target;
    bool inited = false;
};

#endif
