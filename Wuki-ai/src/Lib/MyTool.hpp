#pragma once
#ifndef MYTOOL_HPP
#define MYTOOL_HPP

#include <vector>
#include <string>
#include "MyConfig.hpp"

std::vector<size_t> vs_to_vs(const std::vector<std::string>& vs_input) {
    std::vector<size_t> vs;
    vs.reserve(vs_input.size());

    for (const auto& s : vs_input) {
        try {
            std::size_t pos = 0;
            int value = (size_t)std::stoi(s, &pos);
            if (pos == s.size()) {
                vs.push_back(value);
            }
        } catch (...) {
            // 忽略非法元素
        }
    }

    return vs;
}

#endif  // MYTOOL_HPP
