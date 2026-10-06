#ifndef LLM_BBPE_HPP
#define LLM_BBPE_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <map>
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
        inline void Init() {
            model.tokens.clear();

            // 普通字节
            for (size_t i = 0; i < 256; ++i) {
                model.tokens.push_back(Token{
                    std::string(1, static_cast<char>(i)),
                    i
                });
            }

            // 特殊起始 token
            model.tokens.push_back(Token{ "<__system.START__>", 256 });
            model.tokens.push_back(Token{ "<__system.END__>", 257 });
        }

        // 真·BPE 训练：不断选择最常见的相邻 token pair 进行合并
        inline void Train(const Corpus& corpus) {
            Init();

            /*
             * 一个「词」= 一条符号序列 + 词频
             *
             * 关键：符号序列必须一路带着走。原来的写法把 wordFreq 的 key（拼出来的
             * 字符串）每轮重新拆成「字节」再数 pair —— 合并过的 "ll" 下一轮又被拆回
             * l + l，于是每一轮都在选同一个 pair、又拼出同一个字符串，keys 和频次
             * 一个字都没变 → while(true) 永远退不出来（Train 直接死循环）。
             */
            struct Word {
                std::vector<std::string> symbols;  // 当前的符号序列（一开始是字节）
                size_t count = 0;                  // 这个词在语料里出现几次
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
                    word.symbols.reserve(text.size());
                    for (unsigned char c : text) {
                        word.symbols.push_back(std::string(1, static_cast<char>(c)));
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

            while (true) {
                // 1. 数 pair：按词频加权（同一个词出现 100 次，它内部的 pair 就算 100 次）
                std::map<std::pair<std::string, std::string>, size_t> pairFreq;
                for (const Word& word : words) {
                    for (size_t i = 0; i + 1 < word.symbols.size(); ++i) {
                        pairFreq[{word.symbols[i], word.symbols[i + 1]}] += word.count;
                    }
                }

                if (pairFreq.empty()) {  // 每个词都已经是单个符号了：没有可合并的了
                    break;
                }

                // 2. 选最高频的 pair（频次相同就取字典序最小的那个，结果可复现）
                const auto best = std::max_element(pairFreq.begin(), pairFreq.end(),
                    [](const auto& lhs, const auto& rhs) {
                        if (lhs.second != rhs.second) {
                            return lhs.second < rhs.second;
                        }
                        return lhs.first > rhs.first;
                    });

                const auto pair = best->first;
                const std::string merged = pair.first + pair.second;

                // 3. 加进词表（同一个字符串见过就复用旧 id）
                if (tokenId.find(merged) == tokenId.end()) {
                    const size_t rank = model.tokens.size();
                    tokenId[merged] = rank;
                    model.tokens.push_back(Token{ merged, rank });
                }

                // 4. 在每条符号序列上原地合并 —— 这样下一轮的 pair 统计才看得见这次合并，
                //    而且每轮至少消掉一个符号，总符号数是有限的 → 一定会停
                for (Word& word : words) {
                    for (size_t i = 0; i + 1 < word.symbols.size();) {
                        if (word.symbols[i] == pair.first && word.symbols[i + 1] == pair.second) {
                            word.symbols[i] = merged;
                            word.symbols.erase(word.symbols.begin() + static_cast<std::ptrdiff_t>(i) + 1);
                            // 这里不 ++i：合并出来的 merged 可能还能和后面接着配（a a a → aa a）
                        } else {
                            ++i;
                        }
                    }
                }
            }
        }

        // 按模型 token 表做编码：优先匹配最长 token
        inline std::vector<size_t> Encode(const std::string& text) const {
            std::vector<size_t> ids;      // 存储编码后的token ID
            const size_t none = std::numeric_limits<size_t>::max();

            if (model.tokens.empty()) {  // 没 Init 过就编码：词表是空的，一个字都编不出来
                return {};
            }

            std::string remaining = text; // 剩余待处理的文本

    // 循环处理直到所有文本都被编码
            while (!remaining.empty()) {
                size_t bestId = none;  // 最佳匹配token的ID
                size_t bestLen = 0;    // 最佳匹配token的长度

        // 遍历所有token，寻找最长匹配
                for (size_t i = 0; i < model.tokens.size(); ++i) {
                    const auto& token = model.tokens[i].ch;
                    if (token.empty()) {
                        continue; // 跳过空token
                    }

            // 检查当前token是否匹配文本开头，且比之前找到的更长
                    if (remaining.compare(0, token.size(), token) == 0 && token.size() > bestLen) {
                        bestLen = token.size();
                        bestId = i;
                    }
                }

        // 如果没有找到匹配的token，使用单字节fallback机制
                if (bestId == none) {
                    // 找不到匹配 token，退化成单字节 fallback
                    const unsigned char c = static_cast<unsigned char>(remaining[0]);
                    std::string single(1, static_cast<char>(c));
                    size_t byteId = none;
                    for (size_t i = 0; i < model.tokens.size(); ++i) {
                        if (model.tokens[i].ch == single) {
                            byteId = i;
                            break;
                        }
                    }

                    // 词表里连这个字节都没有：编码失败，直接返回空（调用方按「编不出来」处理）。
                    // 原来这里不 push 也不删字符就 continue —— remaining 一个字都没少，
                    // 于是原地死循环（空词表 / 被人裁过的词表都会中招）
                    if (byteId == none) {
                        return {};
                    }

                    ids.push_back(byteId);       // 添加单字节token的ID
                    remaining.erase(0, 1);       // 移除已处理的首个字符
                    continue;                    // 继续处理剩余文本
                }

        // 添加找到的最佳token ID，并移除已匹配的部分
                ids.push_back(bestId);
                remaining.erase(0, bestLen);
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
    };
}

#endif // LLM_BBPE_HPP