#ifndef FSKIO_CPP
#define FSKIO_CPP

/*
 * WukiComm 动态库的唯一编译单元
 *
 * 把 ras.hpp（传输层：Server）和 fskio.hpp（协议层：FlaskIO）收拢成一个编译单元
 * → Bin/libWukiComm.{dll,dylib,so}，对外只暴露这两个头里的 wuki::comm 接口。
 *
 * httplib.h / json.hpp 只在这里出现：用库的代码（比如 src/main/main.cpp）
 * 不用跟着编译这一万三千行的头，也不会看见 httplib 的类型。
 */

#include "ras.hpp"
#include "fskio.hpp"

#include "../external/httplib.h"  // 本模块依赖 cpp-httplib（HTTP 服务）
#include "../external/json.hpp"   // nlohmann/json（请求 / 响应里的 JSON）

#include <atomic>    // std::atomic（请求计数）
#include <chrono>    // 耗时日志
#include <cstddef>   // std::size_t
#include <exception> // std::exception_ptr
#include <iostream>  // std::cout
#include <mutex>     // std::mutex（生成串行化）
#include <string>
#include <utility>  // std::move

namespace wuki {
    namespace comm {

        namespace {

            /*
             * 正文第一个非空白字符是 '{' 就当 JSON 看待（不看 Content-Type）：
             * Flask 端图省事发 text/plain 也能用，纯文本提示词也不容易和 JSON 撞。
             */
            bool LooksLikeJson(const std::string& body) {
                for (const char c : body) {
                    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                        continue;
                    }
                    return c == '{';
                }
                return false;  // 空正文：按纯文本走，下面再按「提示词为空」拒掉
            }

            /// 出错时的响应：纯文本请求给一行说明，JSON 请求给 {"text": "", "error": "..."}
            Reply ErrorReply(const int status, const std::string& message, const bool json) {
                if (!json) {
                    return Reply{ status, message };
                }
                nlohmann::json obj;
                obj["text"] = std::string{};
                obj["error"] = message;
                return Reply{ status, obj.dump(), "application/json; charset=UTF-8" };
            }

            /// 从 begin 到现在的毫秒数（日志用）
            long long MsSince(const std::chrono::steady_clock::time_point begin) {
                const auto cost = std::chrono::steady_clock::now() - begin;
                return static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(cost).count());
            }

        }  // namespace

        // ====================================================================
        //  Server：收发字节
        // ====================================================================
        struct Server::Impl {
            httplib::Server http;
            Handler handler;
            std::string route = "/post";
            std::size_t port = 9000;
            bool init = false;
        };

        Server::Server() : impl_(new Impl()) {}

        Server::~Server() {
            delete impl_;
            impl_ = nullptr;
        }

        int Server::Init(Handler handler, const std::string& route, std::size_t port) {
            if (impl_->init) {  // 路由只注册一次：重复注册会让 httplib 里同一路径排两条
                return 1;
            }
            if (!handler) {
                return 2;
            }
            if (route.empty()) {
                return 3;
            }
            if (port == 0 || port > 65535) {
                return 4;
            }

            Impl* im = impl_;
            im->handler = std::move(handler);
            im->route = route;
            im->port = port;

            im->http.Post(route, [im](const httplib::Request& req, httplib::Response& res) {
                // 回调抛出的异常不在这里接：交给下面 set_exception_handler 统一转 500
                const Reply reply = im->handler(req.body);
                res.status = reply.status;
                res.set_content(reply.body, reply.type);
            });

            // 兜底：回调抛异常 / httplib 内部出错时回 500，别让连接吊在那儿
            im->http.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                              std::exception_ptr ep) {
                std::string what = "未知异常";
                try {
                    if (ep) {
                        std::rethrow_exception(ep);
                    }
                } catch (const std::exception& e) {
                    what = e.what();
                } catch (...) {
                    // 保持 "未知异常"
                }
                res.status = 500;
                res.set_content("内部错误: " + what, "text/plain; charset=UTF-8");
            });

            im->init = true;
            return 0;
        }

        bool Server::Start() {
            if (!impl_->init) {  // 没注册路由就监听：请求进来没处去，直接拒绝
                std::cout << "[comm] 还没 Init 就 Start，先注册路由" << std::endl;
                return false;
            }
            std::cout << "[comm] 监听 http://0.0.0.0:" << impl_->port << impl_->route
                      << "（Ctrl+C 结束）" << std::endl;

            const bool ok = impl_->http.listen("0.0.0.0", static_cast<int>(impl_->port));
            if (!ok) {
                std::cout << "[comm] 监听失败：端口 " << impl_->port << " 可能已被占用" << std::endl;
            } else {
                std::cout << "[comm] 服务已停止" << std::endl;
            }
            return ok;
        }

        void Server::Stop() {
            impl_->http.stop();
        }

        bool Server::IsInit() const {
            return impl_->init;
        }

        std::size_t Server::port() const {
            return impl_->port;
        }

        // ====================================================================
        //  FlaskIO：把「提示词 → 文本」挂到 HTTP 上
        // ====================================================================
        struct FlaskIO::Impl {
            Server server;
            std::function<std::string(const std::string&)> reply;
            std::mutex lock;                     // 生成串行化：模型不是线程安全的
            std::atomic<std::size_t> served{ 0 };  // 已处理请求数
            bool init = false;
        };

        FlaskIO::FlaskIO() : impl_(new Impl()) {}

        FlaskIO::~FlaskIO() {
            delete impl_;
            impl_ = nullptr;
        }

        int FlaskIO::Init(std::function<std::string(const std::string&)> reply,
                          const std::string& route, std::size_t port) {
            if (impl_->init) {
                return 1;
            }
            if (!reply) {
                return 2;
            }

            Impl* im = impl_;
            im->reply = std::move(reply);

            // 协议层：正文 → 提示词 → 生成 → 响应
            const int code = im->server.Init(
                [im](const std::string& body) {
                    const auto begin = std::chrono::steady_clock::now();
                    const std::size_t index = ++im->served;  // 第几个请求（从 1 数）
                    const bool json = LooksLikeJson(body);

                    // 1) 取提示词：JSON 里读 prompt 字段，纯文本就整份正文
                    std::string prompt;
                    if (json) {
                        try {
                            const nlohmann::json obj = nlohmann::json::parse(body);
                            if (!obj.is_object() || !obj.contains("prompt") || !obj["prompt"].is_string()) {
                                std::cout << "[comm] #" << index << " 400 JSON 里没有字符串字段 prompt" << std::endl;
                                return ErrorReply(400, "JSON 里要有一个字符串字段 prompt", true);
                            }
                            prompt = obj["prompt"].get<std::string>();
                        } catch (const std::exception& e) {
                            std::cout << "[comm] #" << index << " 400 JSON 解析失败: " << e.what() << std::endl;
                            return ErrorReply(400, std::string("JSON 解析失败: ") + e.what(), true);
                        }
                    } else {
                        prompt = body;
                    }
                    if (prompt.empty()) {
                        std::cout << "[comm] #" << index << " 400 提示词为空" << std::endl;
                        return ErrorReply(400, "提示词为空（POST 正文要带上提示词）", json);
                    }

                    // 2) 生成：拿着锁跑，同一时刻只有一个请求在动模型
                    std::string text;
                    try {
                        const std::lock_guard<std::mutex> guard(im->lock);
                        text = im->reply(prompt);
                    } catch (const std::exception& e) {
                        std::cout << "[comm] #" << index << " 500 生成失败: " << e.what()
                                  << "（" << MsSince(begin) << "ms）" << std::endl;
                        return ErrorReply(500, std::string("生成失败: ") + e.what(), json);
                    } catch (...) {
                        std::cout << "[comm] #" << index << " 500 生成失败: 未知异常"
                                  << "（" << MsSince(begin) << "ms）" << std::endl;
                        return ErrorReply(500, "生成失败: 未知异常", json);
                    }

                    // 3) 响应：进来的格式决定回去的格式
                    std::cout << "[comm] #" << index << " 200 提示词 " << prompt.size()
                              << " 字节 → 生成 " << text.size() << " 字节（"
                              << MsSince(begin) << "ms）" << std::endl;
                    if (!json) {
                        return Reply{ 200, text };
                    }
                    nlohmann::json obj;
                    obj["text"] = text;
                    return Reply{ 200, obj.dump(), "application/json; charset=UTF-8" };
                },
                route, port);

            if (code != 0) {  // 3 路径为空 / 4 端口不合法，原样透传
                return code;
            }
            impl_->init = true;
            return 0;
        }

        bool FlaskIO::Start() {
            return impl_->server.Start();
        }

        void FlaskIO::Stop() {
            impl_->server.Stop();
        }

        bool FlaskIO::IsInit() const {
            return impl_->init;
        }

        std::size_t FlaskIO::port() const {
            return impl_->server.port();
        }

        std::size_t FlaskIO::served() const {
            return impl_->served.load();
        }

    }  // namespace comm
}  // namespace wuki

#endif  // FSKIO_CPP
