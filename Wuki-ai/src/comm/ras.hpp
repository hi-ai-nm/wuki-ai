#pragma once
#ifndef RAS_HPP
#define RAS_HPP

/*
 * WukiComm 的传输层：RAS = Receive And Send
 *
 *   实现：src/comm/fskio.cpp  →  Bin/libWukiComm.{dll,dylib,so}
 *   内容：一个把「请求体 → 响应体」交给回调的 HTTP 服务（底下是 cpp-httplib）
 *
 * 这一层只管收发，不知道对面是 Flask、提示词是什么、模型怎么生成 —— 那些在
 * fskio.hpp（协议层，FlaskIO）和 src/main（模型）里。分出这一层是为了让
 * httplib.h 只出现在库里：用它的代码只 #include 本头文件，不用跟着编译 httplib。
 *
 * 典型用法（更省事的写法见 fskio.hpp 的 FlaskIO）：
 *     wuki::comm::Server server;
 *     if (server.Init([](const std::string& req) {
 *             return wuki::comm::Reply{ 200, "收到 " + std::to_string(req.size()) + " 字节" };
 *         }, "/post", 9000) != 0) {
 *         return 8;                                 // 已经初始化过 / 回调为空
 *     }
 *     server.Start();                               // 阻塞，直到 Stop 或监听失败
 */

#include <cstddef>    // std::size_t
#include <functional>  // std::function
#include <string>      // std::string

#include "WukiExport.hpp"  // WUKI_COMM_API

namespace wuki {
    namespace comm {

        /*
         * 一次响应：状态码 + 正文 + Content-Type
         *
         * 不挂 WUKI_COMM_API：它没有需要导出的函数（全在头里构造），
         * 跨库传的只是它里面的 std::string 成员，和 Tools.hpp 里那些接口一个道理。
         */
        struct Reply {
            int status = 200;                                  // HTTP 状态码
            std::string body;                                  // 响应正文（UTF-8）
            std::string type = "text/plain; charset=UTF-8";    // Content-Type
        };

        /// 回调：请求体（UTF-8）→ 响应。抛出的异常由本层兜住，转成 500
        using Handler = std::function<Reply(const std::string&)>;

        /*
         * HTTP 服务（RAII：构造即持有内部实现，析构时释放）
         *
         * 生命周期是「先 Init 注册路由，再 Start 阻塞监听」两步：
         *   服务不跑在后台线程里 —— Start 会一直卡在那里，正好是 exe 想要的样子
         *   （Ctrl+C / Stop 结束）。要放到别的线程里就自己起线程调 Start。
         *
         * 线程模型：httplib 每条连接一个线程，回调会被并发调用。
         * 本层不替调用方加锁 —— 回调共享的东西（比如模型）不是线程安全的，
         * 要串行化的话由上层负责（FlaskIO 就自带一把锁）。
         */
        class WUKI_COMM_API Server {
          public:
            Server();
            ~Server();

            Server(const Server&) = delete;
            Server& operator=(const Server&) = delete;

            /*
             * 注册路由（只注册一次，重复调用返回非 0）
             *
             * @param handler 请求体 → 响应；为空视为参数错误
             * @param route   POST 的路径，默认 "/post"（Flask 端就是往这个地址 POST）
             * @param port    端口，默认 8689（和 docker-compose 里映射出去的那个一致）
             * @return 0 成功；1 已经初始化过；2 回调为空；3 路径为空；4 端口不在 1..65535
             */
            int Init(Handler handler, const std::string& route = "/post", std::size_t port = 9000);

            /*
             * 阻塞监听 0.0.0.0:port，直到 Stop 或被外部打断；返回 false 表示端口没抢到
             *
             * 注意「端口被占」不一定报错：httplib 给监听 socket 设了 SO_REUSEPORT，
             * macOS / Linux 都允许同一端口被重复绑定（请求会被分流到两个进程里）。
             * 所以起服务前自己确认这个端口没被别人（另一个 wuki-ai-b3a）听着。
             */
            bool Start();

            /// 请求停止（可以从别的线程 / 信号处理里调），Start 随即返回
            void Stop();

            /// 已经 Init 过（路由就绪）
            bool IsInit() const;

            /// 当前配置的端口（Init 之后才有意义）
            std::size_t port() const;

          private:
            struct Impl;  // 内部实现（httplib，只存在于动态库里）
            Impl* impl_;
        };

    }  // namespace comm
}  // namespace wuki

#endif  // RAS_HPP
