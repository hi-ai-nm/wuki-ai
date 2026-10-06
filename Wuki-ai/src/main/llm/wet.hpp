// Word embedding tools
// 把一个 token id 转换成一个词向量（不训练，因为直连 RNN，何必呢）
#ifndef WET_HPP
#define WET_HPP

#include "framework.hpp"
#include <cstddef>  // std::size_t
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
     * 为什么要有这个：词表里 256 个字节 token 也各有一张随机向量，全词表找最近邻很容易
     * 挑到一个语料里从没出现过的控制字节（生成出来的就是乱码）。生成时把候选限成
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

            float dist = 0.0f;
            for (size_t j = 0; j < vec.size(); ++j) {
                const float diff = target[id][j] - vec[j];
                dist += diff * diff;
            }
            if (bestId == target.size() || dist < bestDist) {
                bestDist = dist;
                bestId = id;
            }
        }
        return bestId;
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
