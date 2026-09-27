#pragma once
#ifndef MYCONFIG_HPP
#define MYCONFIG_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <fstream>

namespace MyConfig {
    class MC {
        struct Data {
            std::map<std::string, std::string> varData;
            std::map<std::string, std::vector<std::string>> listData;
        };

      public:
        Data data;
        void parse(std::string fileName) {
            std::ifstream file(fileName);
            if (!file.is_open()) {
                std::cerr << "config.conf 打开失败" << std::endl;
                return;
            }

            std::string line;
            while (std::getline(file, line)) {
                std::istringstream lineStream(line);
                if (line.empty() || line[0] == '#') {  // 注释和空行
                    continue;
                }

                std::string mark;
                lineStream >> mark;
                if (mark == "var") {
                    std::string key, value;
                    lineStream >> key >> value;
                    data.varData[key] = value;
                } else if (mark == "list") {
                    std::string key;
                    // value不用，直接存到 data
                    lineStream >> key;
                    std::string valueOne;
                    while (lineStream >> valueOne) {
                        data.listData[key].push_back(valueOne);
                    }
                } else {
                    std::cerr << "config.conf 格式错误" << std::endl;
                    return;
                }
            }
        }
    };
};  // namespace MyConfig

#endif  // MYCONFIG_HPP
