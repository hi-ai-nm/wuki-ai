#ifndef LLM_BBPE_HPP
#define LLM_BBPE_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <map>
#include <unordered_map>
#include <cstdint>  // uint64_t（pair 统计的整数 key）
#include <limits>
#include <cstddef>  // std::size_t / std::ptrdiff_t
#include <utility>  // std::pair

namespace BBpe {

    class BBPE {
        struct Token {
            std::string ch = "";
            size_t rank = 0;
        };
        struct Model {
            std::vector<Token> tokens;
        };
        struct Corpus {
            /* sentences 示例：
            * [
            *   [ "今天", "天气", "很好", "。" ],
            *   [ "我", "是", "中国人", "。" ]
            * ]
            * （实际是 UTF-8 编码的字符串）
            */
            std::vector<std::vector<std::vector<size_t>>> sentences;
        };
    public:
        Model model;

        // 把句子列表转换为语料库
        std::vector<size_t> ToBytes(const std::string& s) {
            std::vector<size_t> v;
            v.reserve(s.size());
            for (unsigned char c : s) {
                v.push_back((size_t)c);
            }
            return v;
        }
        inline Corpus ConvertCorpus(const std::vector<std::string>& sentences) {
            Corpus output;
            output.sentences.reserve(sentences.size());

            for (const auto& sentence : sentences) {
                std::vector<std::vector<size_t>> words;
                std::istringstream iss(sentence);
                std::string token;

                while (iss >> token) {
                    words.push_back(ToBytes(token));
                }

                output.sentences.push_back(words);
            }

            return output;
        }

        static std::string BytesToString(const std::vector<size_t>& bytes) {
            std::string s;
            s.reserve(bytes.size());
            for (size_t b : bytes) {
                s.push_back(static_cast<char>(static_cast<unsigned char>(b)));
            }
            return s;
        }

        // 初始化模型
        // 默认参数是 END 字面量的第四份拷贝（llm.hpp 的 InitPar::specialTokensTOKEN 有完整说明）：
        // LLM 总是显式传，所以这份默认值只是潜伏的第二真相源 —— 谁直接 new 一个 BBPE 才会踩到
        inline void Init(std::vector<std::string> specialTokens = { "@__SYSTEM<END>__" , "@__CMD<PRINT>__" }) {
            model.tokens.clear();

            // 普通字节
            for (size_t i = 0; i < 256; i++) {
                model.tokens.push_back(Token{
                    std::string(1, static_cast<char>(i)),
                    i
                });
            }

            // 特殊 token
            for (size_t idx = 0; idx < specialTokens.size(); idx++) {
                model.tokens.push_back(Token{
                    specialTokens[idx],
                    256 + idx
                });
            }

            RebuildMatchIndex();  // 词表刚立起来（或整个换过），Encode 的匹配索引要重建
            return;
        }

        // 真·BBPE 训练：选最常见的相邻 token pair 进行合并，最多合并 numMerges 轮。
        // 词表初始化是 Init 的职责，这里不再自动调 —— 想续训 / 固定词表大小都由调用方掌控。
        // 不 Init 直接 Train：词表是空的，无事可做，直接返回。
        //
        // 符号全程用 token id（size_t）表示：字节 0..255 的天生 id 就是它的值，merge 出来的
        // 新 token 注册进词表后拿它的 rank。原来的实现拿字符串当符号，每轮统计 pair 都要建
        // 一个 pair<string,string> 当 key 的 map —— 几十亿次字符串比较 + 构建，是全流程
        // 里最贵的一段（5000 行 × 1500 轮实测约 10 s 的绝大部分）。换成 8 字节整数 key 的
        // 哈希表后同样的活便宜一个量级；并列频次的取舍仍按「字符串对字典序最小」，和原来
        // map 序 + max_element 的行为完全一致（同语料训出的词表逐 token 相同）。
        inline void Train(const Corpus& corpus, size_t numMerges) {
            if (model.tokens.empty()) {
                return;
            }

            // 一个「词」= 一条符号 id 序列 + 词频（相同字节串的词只存一份，次数累加）
            struct Word {
                std::vector<size_t> syms;  // 当前符号序列（一开始是字节 id）
                size_t count = 0;          // 这个词在语料里出现几次
            };

            std::vector<Word> words;
            std::map<std::string, size_t> wordIndex;  // 词 → 在 words 里的下标

            for (const auto& sentence : corpus.sentences) {
                for (const auto& raw : sentence) {
                    const std::string text = BytesToString(raw);
                    if (text.empty()) {
                        continue;
                    }

                    const auto it = wordIndex.find(text);
                    if (it != wordIndex.end()) {  // 见过的词：只加词频
                        ++words[it->second].count;
                        continue;
                    }

                    Word word;
                    word.count = 1;
                    word.syms.reserve(text.size());
                    for (unsigned char c : text) {
                        word.syms.push_back(c);  // 字节值 = token id（0..255）
                    }
                    wordIndex[text] = words.size();
                    words.push_back(std::move(word));
                }
            }

            // 已经有过的 token 字符串 → id：合并出来的 token 可能是「同一个字符串、不同
            // 拼法」（a+bc 和 ab+c 都会得到 abc），重复塞进词表只会让同一个词裂成两个 id
            std::map<std::string, size_t> tokenId;
            for (size_t i = 0; i < model.tokens.size(); ++i) {
                tokenId[model.tokens[i].ch] = i;
            }

            for (size_t merge = 0; merge < numMerges; ++merge) {
                // 1. 数 pair（按词频加权，同一个词出现 100 次它内部的 pair 就算 100 次）
                //    key = (a << 32) | b；顺便跟踪最优 pair —— 频次只增不减，每个 pair 的
                //    最后一次累加就是它最终的值，当场和当前最优比即可，不用数完再扫一遍
                std::unordered_map<uint64_t, size_t> pairFreq;
                size_t bestA = 0, bestB = 0, bestFreq = 0;
                bool hasBest = false;
                for (const Word& word : words) {
                    for (size_t i = 0; i + 1 < word.syms.size(); ++i) {
                        const size_t a = word.syms[i];
                        const size_t b = word.syms[i + 1];
                        const uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
                        const size_t f = (pairFreq[key] += word.count);

                        // 频次更高就换；并列取字符串对字典序更小的（和原来
                        // std::map 的字典序 + max_element 保第一个等价）
                        if (!hasBest || f > bestFreq ||
                            (f == bestFreq &&
                             std::make_pair(model.tokens[a].ch, model.tokens[b].ch) <
                                 std::make_pair(model.tokens[bestA].ch, model.tokens[bestB].ch))) {
                            bestA = a;
                            bestB = b;
                            bestFreq = f;
                            hasBest = true;
                        }
                    }
                }

                if (pairFreq.empty()) {  // 每个词都已经是单个符号了：没有可合并的了
                    break;
                }

                // 2. 加进词表（同一个字符串见过就复用旧 id）
                const std::string merged = model.tokens[bestA].ch + model.tokens[bestB].ch;
                size_t newId;
                const auto idIt = tokenId.find(merged);
                if (idIt == tokenId.end()) {
                    newId = model.tokens.size();
                    tokenId[merged] = newId;
                    model.tokens.push_back(Token{ merged, newId });
                } else {
                    newId = idIt->second;
                }

                // 3. 在每条符号序列上原地合并（双指针读写）—— 这样下一轮的 pair 统计
                //    才看得见这次合并，而且每轮至少消掉一个符号，总符号数有限 → 一定会停。
                //    和非重叠从左到右合并的结果一致：a a a → aa a，被并出来的符号不回头
                //    再和前面配（和原实现相同）
                for (Word& word : words) {
                    std::vector<size_t>& s = word.syms;
                    size_t out = 0;
                    for (size_t r = 0; r < s.size();) {
                        if (r + 1 < s.size() && s[r] == bestA && s[r + 1] == bestB) {
                            s[out++] = newId;
                            r += 2;
                        } else {
                            s[out++] = s[r++];
                        }
                    }
                    s.resize(out);
                }
            }

            RebuildMatchIndex();  // 这轮 merge 往词表里加了新 token，Encode 的匹配索引要跟着重建
        }

        /*
         * 按模型 token 表做编码：优先匹配最长 token
         *
         * 匹配走 matchIndex（按首字节分桶、桶内长度降序、id 升序）：第一个 compare
         * 命中的就是「最长且 id 最小」的那个 —— 和原来全表顺序扫描的结果语义一致，
         * 代价从每吐一个 token 扫一遍整个词表（O(词表)）降到只看同首字节的候选。
         * 找不到再退单字节 fallback；连单字节 token 都没有才算编码失败。
         * 下标只挪不 erase：原来是 erase(0, bestLen)，每步都把剩余整串搬一遍。
         */
        inline std::vector<size_t> Encode(const std::string& text) const {
            std::vector<size_t> ids;      // 存储编码后的token ID
            const size_t none = std::numeric_limits<size_t>::max();

            if (model.tokens.empty() || matchIndex.empty()) {  // 没 Init 过就编码：词表是空的，一个字都编不出来
                return {};
            }

            size_t pos = 0;  // 已处理到的下标（取代原来的 remaining.erase）
            while (pos < text.size()) {
                const unsigned char first = static_cast<unsigned char>(text[pos]);

                size_t bestId = none;  // 最佳匹配token的ID
                size_t bestLen = 0;    // 最佳匹配token的长度
                for (const auto& [id, len] : matchIndex[first]) {
                    if (text.compare(pos, len, model.tokens[id].ch) == 0) {
                        bestId = id;
                        bestLen = len;
                        break;  // 长度降序：第一个命中即最长，同长度取 id 最小者
                    }
                }

                // 如果没有找到匹配的token，使用单字节fallback机制
                if (bestId == none) {
                    bestId = byteTokenId[first];  // 词表里这个字节对应的单字节 token（按 id 顺序的第一个）
                    if (bestId == none) {
                        return {};  // 词表里连这个字节都没有：编码失败，直接返回空（调用方按「编不出来」处理）
                    }
                    bestLen = 1;
                }

                ids.push_back(bestId);  // 添加找到的最佳token ID
                pos += bestLen;         // 跳过已匹配的部分
            }

            return ids; // 返回编码结果
        }

        // 把 token id 还原回原始字符串
        inline std::string Decode(const std::vector<size_t>& ids) const {
            std::string text;  // 用于存储解码后的文本
    // 遍历ID向量
            for (size_t id : ids) {
        // 检查ID是否在有效范围内
                if (id < model.tokens.size()) {
            // 将对应ID的字符添加到文本中
                    text += model.tokens[id].ch;
                }
            }
            return text;  // 返回解码后的文本
        }

      private:
        /*
         * Encode 的匹配索引，按首字节分 256 个桶：
         *   matchIndex[首字节] = 以该字节开头的所有 token，桶内按（长度降序、id 升序）排好
         *   byteTokenId[字节]  = 这个字节对应的单字节 token id（fallback 用；没有 = max）
         *
         * 为什么要有它：Encode 原来每吐一个 token 都拿全文和整个词表逐个 compare，
         * 词表 1700+ 时编一遍语料是几十亿次字符串比较（全量 55 万行语料单遍十几分钟的
         * 量级，而训练管线要扫三遍）。分桶后每次只和「首字节相同」的候选比，且长度降序
         * → 循环里第一个命中就是答案，和原来全表扫描的结果完全一致
         * （最长优先；同长度取 id 最小者 —— 原扫描用严格大于保第一个）。
         *
         * 生命周期跟着词表走：Init / Train 改动了 tokens 就重建。这两处都发生在启动
         * 阶段的单线程里，重建完之后 Encode 是纯读 —— -S 起服务后 HTTP 多线程并发
         * 调它也不用加锁。
         */
        inline void RebuildMatchIndex() {
            matchIndex.assign(256, {});
            byteTokenId.assign(256, std::numeric_limits<size_t>::max());

            for (size_t i = 0; i < model.tokens.size(); ++i) {
                const std::string& ch = model.tokens[i].ch;
                if (ch.empty()) {
                    continue;  // 和原扫描一致：空 token 不参与匹配
                }
                const unsigned char first = static_cast<unsigned char>(ch[0]);
                matchIndex[first].emplace_back(i, ch.size());
                if (ch.size() == 1 && byteTokenId[first] == std::numeric_limits<size_t>::max()) {
                    byteTokenId[first] = i;  // 按 id 顺序取第一个（和原 fallback 的逐项找第一个一致）
                }
            }

            for (auto& bucket : matchIndex) {
                std::sort(bucket.begin(), bucket.end(),
                          [](const auto& lhs, const auto& rhs) {
                              if (lhs.second != rhs.second) {
                                  return lhs.second > rhs.second;  // 长度降序：先比长的
                              }
                              return lhs.first < rhs.first;        // 同长度 id 升序
                          });
            }
        }

        std::vector<std::vector<std::pair<size_t, size_t>>> matchIndex;  // [首字节] → [(token id, 长度)]
        std::vector<size_t> byteTokenId;  // 单字节 token 的 id（词表里没有这个字节 = max）
    };
}

#endif // LLM_BBPE_HPP