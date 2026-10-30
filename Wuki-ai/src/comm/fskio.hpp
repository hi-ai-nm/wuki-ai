#pragma once
#ifndef FSKIO_HPP
#define FSKIO_HPP

/*
 * WukiComm 的协议层：FSK IO = Flask IO（Python 端 `import flask as fsk`）
 *
 *   实现：src/comm/fskio.cpp（和 ras.hpp 同一个编译单元）→ Bin/libWukiComm.*
 *   内容：把「提示词 → 文本」这一件事挂到 HTTP 上，让 Flask 端能直接调
 *
 * 和 ras.hpp 的分工：Server 只管收发字节；FlaskIO 负责这套约定的编解码、日志、
 * 异常转 500，以及最要紧的一条 —— 把请求串行化（模型不是线程安全的，
 * 而 httplib 每条连接一个线程，并发进来会同时改动同一份权重）。
 *
 * 协议（POST {route}）：
 *   请求正文是纯文本      → 整份正文就是提示词；响应 200 text/plain，正文是新生成的文本
 *   请求正文是 JSON 对象  → 取其中的 "prompt" 字段（字符串）；响应 200 application/json {"text": "..."}
 *   （看正文第一个非空白字符是不是 '{' 来区分，不看 Content-Type ——
 *     Flask 那边图省事发 text/plain 也能用）
 *
 *   出错：提示词为空 / JSON 不合规 → 400；生成时抛异常 → 500。
 *   正文都是错误说明；JSON 请求的回包是 {"text": "", "error": "..."}。
 *
 * 响应正文只含「新生成的部分」，不含提示词本身。
 *
 * 典型用法：
 *     wuki::comm::FlaskIO io;
 *     io.Init([&model](const std::string& prompt) { return encap::ReplyText(model, prompt); });
 *     io.Start();                                   // 阻塞
 */

#include <cstddef>     // std::size_t
#include <functional>  // std::function
#include <string>      // std::string

#include "WukiExport.hpp"  // WUKI_COMM_API
#include "ras.hpp"         // 底下那个 Server

namespace wuki {
    namespace comm {

        /*
         * Flask 侧的服务入口（内部持有一个 Server，自己不加额外的传输逻辑）
         *
         * 生命周期和 Server 一致：Init 注册 → Start 阻塞监听。
         */
        class WUKI_COMM_API FlaskIO {
          public:
            FlaskIO();
            ~FlaskIO();

            FlaskIO(const FlaskIO&) = delete;
            FlaskIO& operator=(const FlaskIO&) = delete;

            /*
             * 注册「提示词 → 文本」的生成函数
             *
             * reply 会在服务线程里被调用，并且被本类的一把锁串行化：
             * 一次只跑一个请求（排队进来的后一个等前一个生成完）。
             * 只要 reply 自己不去开线程改模型，就不用担心重入。
             *
             * @param reply 提示词 → 生成文本；返回空串 = 「这次没生成出东西」，照样回 200
             * @param route POST 的路径，默认 "/post"
             * @param port  端口，默认 8689（docker-compose 里映射的就是它）
             * @return 0 成功；1 已经初始化过；2 reply 为空；3 路径为空；4 端口不在 1..65535
             */
            int Init(std::function<std::string(const std::string&)> reply,
                     const std::string& route = "/post", std::size_t port = 9000);

            /// 阻塞监听，直到 Stop；返回 false 表示端口没抢到
            bool Start();

            /// 请求停止
            void Stop();

            /// 已经 Init 过
            bool IsInit() const;

            /// 当前配置的端口
            std::size_t port() const;

            /// 已经处理过的请求数（打印 / 排查用；服务线程里自增，可以边跑边读）
            std::size_t served() const;

          private:
            struct Impl;
            Impl* impl_;
        };

    }  // namespace comm
}  // namespace wuki

#endif  // FSKIO_HPP
