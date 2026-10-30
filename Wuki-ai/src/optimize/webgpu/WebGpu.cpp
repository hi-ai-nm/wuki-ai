/*
 * WebGPU 计算后端实现（wgpu-native 的 webgpu.h C API）
 *
 * 许可证：本文件是本项目自己的代码（MIT，见仓库根 LICENSE）。它调用 wgpu-native
 * （Apache-2.0 / MIT 双许可，本项目按 Apache-2.0 使用）和 webgpu.h（BSD-3-Clause），
 * 二者的许可证原文与合规说明见 src/external/wgpu-native-LICENSE.*、
 * src/external/webgpu-headers-LICENSE.txt 和 README 的「第三方许可证」一节。
 */

#include "optimize/webgpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <thread>
#include <vector>

#if defined(WUKI_HAS_WEBGPU) && WUKI_HAS_WEBGPU

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include "WebGpuShaders.hpp"

namespace {

    using Clock = std::chrono::steady_clock;

    /// 一次调用的固定参数：4 个 32 位槽，整数 / 浮点按需解释
    struct Params {
        std::uint32_t i0 = 0;
        std::uint32_t i1 = 0;
        float f0 = 0.0f;
        std::uint32_t i2 = 0;
    };
    static_assert(sizeof(Params) == 16, "uniform 参数块必须 16 字节对齐");

    /// 一个内核 = 管线 + 它的 bind group layout（auto layout，按 entry point 生成）
    struct Kernel {
        WGPUComputePipeline pipeline = nullptr;
        WGPUBindGroupLayout layout = nullptr;
    };

    /// 设备 + 管线 + 常驻 uniform buffer
    struct Context {
        bool tried = false;   ///< 初始化试过了（无论成败都不再试）
        bool ready = false;   ///< 设备可用
        std::string name;     ///< 适配器名，例如 "Apple M5"

        WGPUInstance instance = nullptr;
        WGPUAdapter adapter = nullptr;
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WGPUShaderModule module = nullptr;
        WGPUBuffer uniform = nullptr;

        Kernel dot;
        Kernel dotReduce;
        Kernel axpy;
        Kernel gemv;
        Kernel gemt;
        Kernel rank1;
        Kernel distance;
        Kernel paired;

        // 缓冲池：同一尺寸反复申请 / 归还，省掉每次 wgpuDeviceCreateBuffer 的开销。
        // 所有内核调用都是「提交 → 等完成 → 读回」的同步流程，复用是安全的。
        std::unordered_map<std::uint64_t, std::vector<WGPUBuffer>> storagePool;
        std::unordered_map<std::uint64_t, std::vector<WGPUBuffer>> readbackPool;
    };

    Context g_ctx;
    std::mutex g_mutex;

    // 运行期错误（设备丢失 / 校验失败）：回调可能在任何线程触发，所以：
    //   - 「坏了」只用一个 atomic<bool>；
    //   - 错误文本单独一把锁，绝不在持锁调用 WebGPU API 时去拿它（不会和 g_mutex 交叉）。
    std::atomic<bool> g_broken{false};
    std::mutex g_errorMutex;
    std::string g_errorText;

    void setError(std::string text) {
        g_broken.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(g_errorMutex);
        g_errorText = std::move(text);
    }

    std::string getError() {
        std::lock_guard<std::mutex> lock(g_errorMutex);
        return g_errorText;
    }

    /// 设备可用性的快速缓存：-1 = 还没探测过，0 = 不可用，1 = 可用。
    /// ShouldUse 在训练循环里会被每层每步调用一次，不能每次都去抢 g_mutex。
    std::atomic<int> g_ready{-1};

    std::atomic<Wuki::Gpu::Mode> g_mode{Wuki::Gpu::Mode::Auto};
    std::atomic<std::size_t> g_minElements{0};  // 0 = 还没定，用 kDefaultMinElements

    std::atomic<std::uint64_t> g_ops{0};
    std::atomic<std::uint64_t> g_bytes{0};

    /// 阶段耗时（纳秒），只在 WUKI_GPU_TRACE=1 时累加
    struct PhaseCost {
        std::uint64_t calls = 0;
        std::uint64_t write = 0;
        std::uint64_t bind = 0;
        std::uint64_t submit = 0;
        std::uint64_t wait = 0;
    };
    PhaseCost g_phase;
    bool g_trace = false;

    void initTrace();

    std::uint64_t nowNs() {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
                .count());
    }

    /**
     * Auto 模式的默认阈值（元素个数）
     *
     * 来源：wuki-gpu-bench 在本机（Apple M5 / Metal，wgpu-native v29）上的实测：
     *   - 一次「上传 → dispatch → 等完成 → 读回」的固定开销约 0.43 ms；
     *   - 拿真正的 SIMD 内核当基线，gemv / gemt 要到 16M 元素（64 MB）才打平，
     *     4M 时 GPU 只有 CPU 的 1/3；rank1 / axpy / dot 这种「搬进搬出占大头」的
     *     内核在任何尺寸都赢不了。
     * 所以 Auto 阈值定在 16M，宁可不用也别默认拖慢 —— 本项目默认模型只有几十到
     * 几万个元素的矩阵，永远走 SIMD。换大模型 / 换显卡后用 bench 重新标定，
     * 或者用 WUKI_GPU_MIN 覆盖。
     */
    constexpr std::size_t kDefaultMinElements = 1u << 24;

    std::string viewToString(WGPUStringView v) {
        if (v.data == nullptr) {
            return {};
        }
        const std::size_t n = (v.length == WGPU_STRLEN) ? std::strlen(v.data) : v.length;
        return std::string(v.data, n);
    }

    std::string backendName(WGPUBackendType t) {
        switch (t) {
            case WGPUBackendType_Null: return "null";
            case WGPUBackendType_WebGPU: return "WebGPU";
            case WGPUBackendType_D3D11: return "D3D11";
            case WGPUBackendType_D3D12: return "D3D12";
            case WGPUBackendType_Metal: return "Metal";
            case WGPUBackendType_Vulkan: return "Vulkan";
            case WGPUBackendType_OpenGL: return "OpenGL";
            case WGPUBackendType_OpenGLES: return "OpenGL ES";
            default: return "unknown";
        }
    }

    /// 等一个「自发性回调」完成：wgpu-native 没实现 wgpuInstanceWaitAny，只能 poll
    template <typename State>
    bool waitCallback(State& state, const int timeoutMs = 10000) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        while (!state.done.load(std::memory_order_acquire)) {
            wgpuInstanceProcessEvents(g_ctx.instance);
            if (Clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    struct AdapterState {
        std::atomic<bool> done{false};
        WGPUAdapter adapter = nullptr;
        std::string error;
    };

    struct DeviceState {
        std::atomic<bool> done{false};
        WGPUDevice device = nullptr;
        std::string error;
    };

    void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message,
                   void* userdata1, void*) {
        auto* state = static_cast<AdapterState*>(userdata1);
        if (status == WGPURequestAdapterStatus_Success && adapter != nullptr) {
            state->adapter = adapter;
        } else {
            state->error = viewToString(message);
            if (state->error.empty()) {
                state->error = "请求适配器失败";
            }
        }
        state->done.store(true, std::memory_order_release);
    }

    void onDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message,
                  void* userdata1, void*) {
        auto* state = static_cast<DeviceState*>(userdata1);
        if (status == WGPURequestDeviceStatus_Success && device != nullptr) {
            state->device = device;
        } else {
            state->error = viewToString(message);
            if (state->error.empty()) {
                state->error = "请求设备失败";
            }
        }
        state->done.store(true, std::memory_order_release);
    }

    /// 设备报错（校验失败、驱动炸了）：只置标志，不在这里加锁 / 打印
    void onDeviceError(WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
        if (type == WGPUErrorType_NoError) {
            return;
        }
        setError(viewToString(message));
    }

    void onDeviceLost(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message,
                      void*, void*) {
        if (reason == WGPUDeviceLostReason_Destroyed) {
            return;
        }
        setError(viewToString(message));
    }

    Kernel makeKernel(const char* entryPoint) {
        WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        desc.compute.module = g_ctx.module;
        desc.compute.entryPoint = {entryPoint, WGPU_STRLEN};
        Kernel k;
        k.pipeline = wgpuDeviceCreateComputePipeline(g_ctx.device, &desc);
        if (k.pipeline != nullptr) {
            k.layout = wgpuComputePipelineGetBindGroupLayout(k.pipeline, 0);
        }
        return k;
    }

    /// 初始化（调用方必须已经持有 g_mutex）
    void initLocked() {
        if (g_ctx.tried) {
            return;
        }
        g_ctx.tried = true;
        initTrace();

        g_ctx.instance = wgpuCreateInstance(nullptr);
        if (g_ctx.instance == nullptr) {
            setError("创建 WebGPU instance 失败");
            return;
        }

        WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
        options.powerPreference = WGPUPowerPreference_HighPerformance;

        AdapterState adapterState;
        WGPURequestAdapterCallbackInfo adapterCb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
        adapterCb.mode = WGPUCallbackMode_AllowSpontaneous;
        adapterCb.callback = onAdapter;
        adapterCb.userdata1 = &adapterState;
        wgpuInstanceRequestAdapter(g_ctx.instance, &options, adapterCb);
        if (!waitCallback(adapterState) || adapterState.adapter == nullptr) {
            setError(adapterState.error.empty() ? "没有可用的 WebGPU 适配器" : adapterState.error);
            return;
        }
        g_ctx.adapter = adapterState.adapter;

        WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
        if (wgpuAdapterGetInfo(g_ctx.adapter, &info) == WGPUStatus_Success) {
            const std::string deviceName = viewToString(info.device);
            const std::string desc = viewToString(info.description);
            g_ctx.name = deviceName.empty() ? desc : deviceName;
            if (g_ctx.name.empty()) {
                g_ctx.name = "WebGPU 适配器";
            }
            g_ctx.name += " (";
            g_ctx.name += backendName(info.backendType);
            g_ctx.name += ")";
            wgpuAdapterInfoFreeMembers(info);
        }

        WGPUDeviceDescriptor deviceDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
        deviceDesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        deviceDesc.deviceLostCallbackInfo.callback = onDeviceLost;
        deviceDesc.uncapturedErrorCallbackInfo.callback = onDeviceError;

        DeviceState deviceState;
        WGPURequestDeviceCallbackInfo deviceCb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
        deviceCb.mode = WGPUCallbackMode_AllowSpontaneous;
        deviceCb.callback = onDevice;
        deviceCb.userdata1 = &deviceState;
        wgpuAdapterRequestDevice(g_ctx.adapter, &deviceDesc, deviceCb);
        if (!waitCallback(deviceState) || deviceState.device == nullptr) {
            setError(deviceState.error.empty() ? "请求 WebGPU 设备失败" : deviceState.error);
            return;
        }
        g_ctx.device = deviceState.device;
        g_ctx.queue = wgpuDeviceGetQueue(g_ctx.device);

        WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgsl.code = {kWukiWgsl, WGPU_STRLEN};
        moduleDesc.nextInChain = &wgsl.chain;
        g_ctx.module = wgpuDeviceCreateShaderModule(g_ctx.device, &moduleDesc);
        if (g_ctx.module == nullptr) {
            setError("编译 WGSL 失败");
            return;
        }

        g_ctx.dot = makeKernel("dot_partial");
        g_ctx.dotReduce = makeKernel("dot_reduce");
        g_ctx.axpy = makeKernel("axpy_main");
        g_ctx.gemv = makeKernel("gemv_main");
        g_ctx.gemt = makeKernel("gemt_main");
        g_ctx.rank1 = makeKernel("rank1_main");
        g_ctx.distance = makeKernel("distance_main");
        g_ctx.paired = makeKernel("paired_main");

        WGPUBufferDescriptor uniformDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
        uniformDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        uniformDesc.size = sizeof(Params);
        g_ctx.uniform = wgpuDeviceCreateBuffer(g_ctx.device, &uniformDesc);

        if (g_ctx.dot.pipeline == nullptr || g_ctx.dotReduce.pipeline == nullptr ||
            g_ctx.axpy.pipeline == nullptr ||
            g_ctx.gemv.pipeline == nullptr || g_ctx.gemt.pipeline == nullptr ||
            g_ctx.rank1.pipeline == nullptr || g_ctx.distance.pipeline == nullptr ||
            g_ctx.paired.pipeline == nullptr || g_ctx.uniform == nullptr) {
            setError("创建 WebGPU 计算管线失败");
            return;
        }

        g_ctx.ready = true;
    }

    bool ensureReady() {
        if (g_broken.load(std::memory_order_acquire)) {
            return false;
        }
        if (!g_ctx.tried) {
            initLocked();
        }
        return g_ctx.ready;
    }

    WGPUBuffer createBuffer(std::uint64_t bytes, WGPUBufferUsage usage) {
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.usage = usage;
        desc.size = bytes;
        return wgpuDeviceCreateBuffer(g_ctx.device, &desc);
    }

    /// 从池里拿一个「至少 bytes 大」的 buffer；尺寸按 16 字节对齐，减少碎片
    WGPUBuffer acquireBuffer(std::unordered_map<std::uint64_t, std::vector<WGPUBuffer>>& pool,
                             std::uint64_t bytes, WGPUBufferUsage usage) {
        const std::uint64_t aligned = (bytes + 15u) & ~std::uint64_t{15u};
        auto it = pool.find(aligned);
        if (it != pool.end() && !it->second.empty()) {
            WGPUBuffer buffer = it->second.back();
            it->second.pop_back();
            return buffer;
        }
        return createBuffer(aligned, usage);
    }

    void recycleBuffer(std::unordered_map<std::uint64_t, std::vector<WGPUBuffer>>& pool,
                       WGPUBuffer buffer, std::uint64_t bytes) {
        if (buffer == nullptr) {
            return;
        }
        pool[(bytes + 15u) & ~std::uint64_t{15u}].push_back(buffer);
    }

    /// 上传一段主机内存到 storage buffer（可读回）
    WGPUBuffer uploadBuffer(const void* data, std::uint64_t bytes) {
        WGPUBuffer buffer = acquireBuffer(g_ctx.storagePool, bytes,
                                          WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                              WGPUBufferUsage_CopySrc);
        if (buffer != nullptr && data != nullptr && bytes != 0) {
            wgpuQueueWriteBuffer(g_ctx.queue, buffer, 0, data, static_cast<std::size_t>(bytes));
        }
        return buffer;
    }

    /// 用完还回池里（不是销毁）
    void releaseBuffer(WGPUBuffer buffer, std::uint64_t bytes) {
        recycleBuffer(g_ctx.storagePool, buffer, bytes);
    }

    struct MapState {
        std::atomic<bool> done{false};
        bool ok = false;
        std::string error;
    };

    void onMap(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*) {
        auto* state = static_cast<MapState*>(userdata1);
        state->ok = (status == WGPUMapAsyncStatus_Success);
        if (!state->ok) {
            state->error = viewToString(message);
        }
        state->done.store(true, std::memory_order_release);
    }

    /// 把 readback buffer 的内容拷回主机；失败返回 false
    bool readbackInto(WGPUBuffer buffer, void* host, std::uint64_t bytes) {
        MapState state;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowSpontaneous;
        cb.callback = onMap;
        cb.userdata1 = &state;
        wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, static_cast<std::size_t>(bytes), cb);

        // 等队列干完 + 把 map 回调放出来。三个坑（都在本机实测过）：
        //   - 每轮 sleep 1ms：固定开销 2.6ms；
        //   - 自旋里调 wgpuDevicePoll(wait=true)：它内部按毫秒粒度睡，仍是 1.3ms；
        //   - 纯 processEvents 自旋 + 偶尔 poll(false)：0.4~0.5ms，是 wgpu-native 在这台
        //     M5 上能到的下限。每 64 轮补一次非阻塞 poll，照顾需要主动推进的 Vulkan 后端。
        const auto deadline = Clock::now() + std::chrono::milliseconds(20000);
        std::uint64_t spin = 0;
        while (!state.done.load(std::memory_order_acquire)) {
            wgpuInstanceProcessEvents(g_ctx.instance);
            if ((spin & 0x3f) == 0x3f) {
                wgpuDevicePoll(g_ctx.device, WGPU_FALSE, nullptr);
            }
            if ((spin & 0x3ff) == 0x3ff) {
                std::this_thread::yield();
            }
            if ((spin & 0xffff) == 0xffff && Clock::now() >= deadline) {
                return false;
            }
            ++spin;
        }
        if (!state.ok) {
            return false;
        }

        const void* mapped = wgpuBufferGetConstMappedRange(buffer, 0, static_cast<std::size_t>(bytes));
        if (mapped != nullptr) {
            std::memcpy(host, mapped, static_cast<std::size_t>(bytes));
        }
        wgpuBufferUnmap(buffer);
        return mapped != nullptr;
    }

    struct Binding {
        WGPUBuffer buffer = nullptr;
        std::uint64_t size = 0;
    };

    /// 需要读回主机的输出
    struct Output {
        WGPUBuffer buffer = nullptr;
        std::uint64_t size = 0;
        void* host = nullptr;
        WGPUBuffer readback = nullptr;
    };

    /**
     * 一次完整的内核调用：写参数 → 建 bind group → dispatch → 拷到 readback →
     * 提交 → 映射读回。成功返回 true。
     *
     * 约定：bindings[0..] 依次是 binding 1、2、3...（binding 0 永远是 uniform 参数块）。
     * 所有 buffer 由调用方创建 / 释放，本函数只管用。
     */
    bool runKernel(const Kernel& kernel, const Params& params, const std::vector<Binding>& bindings,
                   std::uint32_t groupsX, std::uint32_t groupsY, std::vector<Output>& outputs) {
        if (kernel.pipeline == nullptr || groupsX == 0) {
            return false;
        }

        const std::uint64_t t0 = g_trace ? nowNs() : 0;
        wgpuQueueWriteBuffer(g_ctx.queue, g_ctx.uniform, 0, &params, sizeof(params));

        std::vector<WGPUBindGroupEntry> entries(1 + bindings.size());
        entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[0].binding = 0;
        entries[0].buffer = g_ctx.uniform;
        entries[0].offset = 0;
        entries[0].size = sizeof(Params);
        for (std::size_t i = 0; i < bindings.size(); ++i) {
            entries[i + 1] = WGPU_BIND_GROUP_ENTRY_INIT;
            entries[i + 1].binding = static_cast<std::uint32_t>(i + 1);
            entries[i + 1].buffer = bindings[i].buffer;
            entries[i + 1].offset = 0;
            entries[i + 1].size = bindings[i].size;
        }

        WGPUBindGroupDescriptor bindGroupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bindGroupDesc.layout = kernel.layout;
        bindGroupDesc.entryCount = entries.size();
        bindGroupDesc.entries = entries.data();
        const std::uint64_t t1 = g_trace ? nowNs() : 0;
        WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(g_ctx.device, &bindGroupDesc);
        if (bindGroup == nullptr) {
            return false;
        }

        for (auto& out : outputs) {
            out.readback = acquireBuffer(g_ctx.readbackPool, out.size,
                                         WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
            if (out.readback == nullptr) {
                wgpuBindGroupRelease(bindGroup);
                return false;
            }
        }

        WGPUCommandEncoderDescriptor encoderDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_ctx.device, &encoderDesc);

        WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, kernel.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, groupsX, groupsY, 1);
        wgpuComputePassEncoderEnd(pass);

        for (auto& out : outputs) {
            wgpuCommandEncoderCopyBufferToBuffer(encoder, out.buffer, 0, out.readback, 0, out.size);
        }

        WGPUCommandBufferDescriptor cmdDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        const std::uint64_t t2 = g_trace ? nowNs() : 0;
        WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(g_ctx.queue, 1, &command);
        const std::uint64_t t3 = g_trace ? nowNs() : 0;

        bool ok = !g_broken.load(std::memory_order_acquire);
        for (auto& out : outputs) {
            if (ok && out.host != nullptr) {
                ok = readbackInto(out.readback, out.host, out.size);
            }
            recycleBuffer(g_ctx.readbackPool, out.readback, out.size);
            out.readback = nullptr;
        }

        const std::uint64_t t4 = g_trace ? nowNs() : 0;
        if (g_trace) {
            g_phase.calls += 1;
            g_phase.write += t1 - t0;
            g_phase.bind += t2 - t1;
            g_phase.submit += t3 - t2;
            g_phase.wait += t4 - t3;
        }

        wgpuCommandBufferRelease(command);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bindGroup);
        return ok && !g_broken.load(std::memory_order_acquire);
    }

    void addStats(std::uint64_t bytes) {
        g_ops.fetch_add(1, std::memory_order_relaxed);
        g_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    /// 统一的「先锁、再确认、再跑」外壳；内核失败一律回退（返回 false）
    template <typename Fn>
    bool guarded(std::uint64_t bytes, Fn&& fn) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!ensureReady()) {
            return false;
        }
        const bool ok = fn();
        if (!ok) {
            // 出错一次就不再折腾 GPU：后面全部走 SIMD，避免训练循环被反复超时拖死
            if (g_broken.load(std::memory_order_acquire)) {
                g_ctx.ready = false;
                g_ready.store(0, std::memory_order_release);
            }
            return false;
        }
        addStats(bytes);
        return true;
    }

    void initTrace() {
        static const bool once = [] {
            g_trace = std::getenv("WUKI_GPU_TRACE") != nullptr;
            return true;
        }();
        (void)once;
    }

    std::size_t effectiveMinElements() {
        const std::size_t v = g_minElements.load(std::memory_order_relaxed);
        return v == 0 ? kDefaultMinElements : v;
    }

}  // namespace

namespace Wuki::Gpu {

    bool Available() {
        const int cached = g_ready.load(std::memory_order_acquire);
        if (cached >= 0) {
            return cached == 1;
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const int again = g_ready.load(std::memory_order_relaxed);
        if (again >= 0) {
            return again == 1;
        }
        const bool ok = ensureReady();
        g_ready.store(ok ? 1 : 0, std::memory_order_release);
        return ok;
    }

    const char* Error() {
        // 返回的指针在下一次调用前有效（thread_local 兜住临时 string，别返回悬垂指针）
        static thread_local std::string text;
        text = getError();
        return text.c_str();
    }

    std::string Info() {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (ensureReady()) {
            return g_ctx.name;
        }
        const std::string reason = getError();
        return reason.empty() ? "WebGPU 不可用（未编译进 WebGPU 支持）" : reason;
    }

    Mode GetMode() {
        return g_mode.load(std::memory_order_relaxed);
    }

    void SetMode(Mode mode) {
        g_mode.store(mode, std::memory_order_relaxed);
    }

    std::size_t MinElements() {
        return effectiveMinElements();
    }

    void SetMinElements(std::size_t elements) {
        g_minElements.store(elements, std::memory_order_relaxed);
    }

    bool ShouldUse(std::size_t elements) {
        if (GetMode() == Mode::Off) {
            return false;
        }
        if (GetMode() == Mode::Auto && elements < effectiveMinElements()) {
            return false;
        }
        return Available();
    }

    void ConfigureFromEnv() {
        if (const char* mode = std::getenv("WUKI_GPU")) {
            const std::string text(mode);
            if (text == "off" || text == "0" || text == "false") {
                SetMode(Mode::Off);
            } else if (text == "force" || text == "1" || text == "true") {
                SetMode(Mode::Force);
            } else {
                SetMode(Mode::Auto);
            }
        }
        if (const char* min = std::getenv("WUKI_GPU_MIN")) {
            try {
                SetMinElements(std::stoull(min));
            } catch (...) {
                // 环境变量写坏就当没写，保持默认阈值
            }
        }
    }

    Stats GetStats() {
        Stats s;
        s.ops = g_ops.load(std::memory_order_relaxed);
        s.bytes = g_bytes.load(std::memory_order_relaxed);
        return s;
    }

    void ResetStats() {
        g_ops.store(0, std::memory_order_relaxed);
        g_bytes.store(0, std::memory_order_relaxed);
    }

    void Shutdown() {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_trace && g_phase.calls > 0) {
            const double calls = static_cast<double>(g_phase.calls);
            std::printf("[wgpu] 阶段耗时（%llu 次平均，微秒）: 写参数 %.0f | 建 bind group %.0f | 提交 %.0f | 等待+读回 %.0f\n",
                        static_cast<unsigned long long>(g_phase.calls),
                        g_phase.write / calls / 1000.0, g_phase.bind / calls / 1000.0,
                        g_phase.submit / calls / 1000.0, g_phase.wait / calls / 1000.0);
        }
        if (!g_ctx.ready && !g_ctx.tried) {
            return;
        }
        for (auto& bucket : g_ctx.storagePool) {
            for (WGPUBuffer buffer : bucket.second) {
                wgpuBufferRelease(buffer);
            }
        }
        for (auto& bucket : g_ctx.readbackPool) {
            for (WGPUBuffer buffer : bucket.second) {
                wgpuBufferRelease(buffer);
            }
        }
        const Kernel kernels[] = {g_ctx.dot, g_ctx.dotReduce, g_ctx.axpy, g_ctx.gemv, g_ctx.gemt,
                                  g_ctx.rank1, g_ctx.distance, g_ctx.paired};
        for (const Kernel& k : kernels) {
            if (k.pipeline != nullptr) {
                wgpuComputePipelineRelease(k.pipeline);
            }
            if (k.layout != nullptr) {
                wgpuBindGroupLayoutRelease(k.layout);
            }
        }
        if (g_ctx.uniform != nullptr) wgpuBufferRelease(g_ctx.uniform);
        if (g_ctx.module != nullptr) wgpuShaderModuleRelease(g_ctx.module);
        if (g_ctx.queue != nullptr) wgpuQueueRelease(g_ctx.queue);
        if (g_ctx.device != nullptr) wgpuDeviceRelease(g_ctx.device);
        if (g_ctx.adapter != nullptr) wgpuAdapterRelease(g_ctx.adapter);
        if (g_ctx.instance != nullptr) wgpuInstanceRelease(g_ctx.instance);
        g_ctx = Context{};
        g_ready.store(-1, std::memory_order_release);
    }

    // ========================================================================
    // 内核实现
    // ========================================================================

    bool Dot(const float* a, const float* b, std::size_t n, float* out) {
        if (a == nullptr || b == nullptr || out == nullptr || n == 0) {
            return false;
        }
        const std::uint64_t bytes = static_cast<std::uint64_t>(n) * sizeof(float);
        return guarded(2 * bytes + sizeof(float), [&]() {
            // 趟 1：最多 1024 个 workgroup 各出一份部分和（再多就只是多写几份，没意义）
            const std::uint32_t groups = static_cast<std::uint32_t>(
                std::min<std::size_t>(1024, std::max<std::size_t>(1, (n + 255) / 256)));
            const std::uint64_t partialBytes = static_cast<std::uint64_t>(groups) * sizeof(float);

            Params p1;
            p1.i0 = static_cast<std::uint32_t>(n);
            p1.i1 = groups;

            WGPUBuffer bufA = uploadBuffer(a, bytes);
            WGPUBuffer bufB = uploadBuffer(b, bytes);
            WGPUBuffer bufPartial = uploadBuffer(nullptr, partialBytes);

            std::vector<Output> noOutput;
            bool ok = runKernel(g_ctx.dot, p1,
                                {{bufA, bytes}, {bufB, bytes}, {bufPartial, partialBytes}}, groups, 1,
                                noOutput);

            WGPUBuffer bufOut = nullptr;
            if (ok) {
                // 趟 2：一个 workgroup 把 <=1024 个部分和加完
                Params p2;
                p2.i0 = groups;
                bufOut = uploadBuffer(nullptr, sizeof(float));
                std::vector<Output> outputs{{bufOut, sizeof(float), out, nullptr}};
                ok = runKernel(g_ctx.dotReduce, p2, {{bufPartial, partialBytes}, {bufOut, sizeof(float)}},
                               1, 1, outputs);
            }

            releaseBuffer(bufA, bytes);
            releaseBuffer(bufB, bytes);
            releaseBuffer(bufPartial, partialBytes);
            releaseBuffer(bufOut, sizeof(float));
            return ok;
        });
    }

    bool Axpy(float* y, const float* x, float alpha, std::size_t n) {
        if (y == nullptr || x == nullptr || n == 0) {
            return false;
        }
        const std::uint64_t bytes = static_cast<std::uint64_t>(n) * sizeof(float);
        return guarded(3 * bytes, [&]() {
            // 每个维度最多 65535 个 workgroup；n 更大时靠内核里的跨步循环走第二趟
            const std::uint32_t groups = static_cast<std::uint32_t>(
                std::min<std::size_t>(65535, std::max<std::size_t>(1, (n + 63) / 64)));
            Params p;
            p.i0 = static_cast<std::uint32_t>(n);
            p.i1 = groups * 64u;
            p.f0 = alpha;

            WGPUBuffer bufY = uploadBuffer(y, bytes);
            WGPUBuffer bufX = uploadBuffer(x, bytes);
            std::vector<Output> outputs{{bufY, bytes, y, nullptr}};

            const bool ok = runKernel(g_ctx.axpy, p, {{bufY, bytes}, {bufX, bytes}}, groups, 1, outputs);

            releaseBuffer(bufY, bytes);
            releaseBuffer(bufX, bytes);
            return ok;
        });
    }

    bool Gemv(const float* W, const float* x, const float* b, float* y, std::size_t rows,
              std::size_t cols) {
        if (W == nullptr || x == nullptr || y == nullptr || rows == 0 || cols == 0) {
            return false;
        }
        const std::uint64_t wBytes = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
        const std::uint64_t xBytes = static_cast<std::uint64_t>(rows) * sizeof(float);
        const std::uint64_t yBytes = static_cast<std::uint64_t>(cols) * sizeof(float);
        return guarded(wBytes + xBytes + 2 * yBytes, [&]() {
            Params p;
            p.i0 = static_cast<std::uint32_t>(rows);
            p.i1 = static_cast<std::uint32_t>(cols);

            std::vector<float> zeros;
            const float* bias = b;
            if (bias == nullptr) {
                zeros.assign(cols, 0.0f);
                bias = zeros.data();
            }

            WGPUBuffer bufW = uploadBuffer(W, wBytes);
            WGPUBuffer bufX = uploadBuffer(x, xBytes);
            WGPUBuffer bufB = uploadBuffer(bias, yBytes);
            WGPUBuffer bufY = uploadBuffer(nullptr, yBytes);
            std::vector<Output> outputs{{bufY, yBytes, y, nullptr}};

            const bool ok = runKernel(g_ctx.gemv, p,
                                      {{bufW, wBytes}, {bufX, xBytes}, {bufB, yBytes}, {bufY, yBytes}},
                                      static_cast<std::uint32_t>((cols + 63) / 64), 1, outputs);

            releaseBuffer(bufW, wBytes);
            releaseBuffer(bufX, xBytes);
            releaseBuffer(bufB, yBytes);
            releaseBuffer(bufY, yBytes);
            return ok;
        });
    }

    bool Gemt(const float* W, const float* u, const float* scale, float* out, std::size_t rows,
              std::size_t cols) {
        if (W == nullptr || u == nullptr || out == nullptr || rows == 0 || cols == 0) {
            return false;
        }
        const std::uint64_t wBytes = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
        const std::uint64_t uBytes = static_cast<std::uint64_t>(rows) * sizeof(float);
        const std::uint64_t outBytes = static_cast<std::uint64_t>(cols) * sizeof(float);
        return guarded(wBytes + uBytes + 2 * outBytes, [&]() {
            Params p;
            p.i0 = static_cast<std::uint32_t>(rows);
            p.i1 = static_cast<std::uint32_t>(cols);

            std::vector<float> ones;
            const float* s = scale;
            if (s == nullptr) {
                ones.assign(cols, 1.0f);
                s = ones.data();
            }

            WGPUBuffer bufW = uploadBuffer(W, wBytes);
            WGPUBuffer bufU = uploadBuffer(u, uBytes);
            WGPUBuffer bufS = uploadBuffer(s, outBytes);
            WGPUBuffer bufOut = uploadBuffer(nullptr, outBytes);
            std::vector<Output> outputs{{bufOut, outBytes, out, nullptr}};

            const bool ok = runKernel(g_ctx.gemt, p,
                                      {{bufW, wBytes}, {bufU, uBytes}, {bufS, outBytes}, {bufOut, outBytes}},
                                      static_cast<std::uint32_t>((cols + 63) / 64), 1, outputs);

            releaseBuffer(bufW, wBytes);
            releaseBuffer(bufU, uBytes);
            releaseBuffer(bufS, outBytes);
            releaseBuffer(bufOut, outBytes);
            return ok;
        });
    }

    bool Rank1(float* W, const float* delta, const float* prev, float lr, std::size_t rows,
               std::size_t cols) {
        if (W == nullptr || delta == nullptr || prev == nullptr || rows == 0 || cols == 0) {
            return false;
        }
        const std::uint64_t wBytes = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
        const std::uint64_t dBytes = static_cast<std::uint64_t>(rows) * sizeof(float);
        const std::uint64_t pBytes = static_cast<std::uint64_t>(cols) * sizeof(float);
        return guarded(2 * wBytes + dBytes + pBytes, [&]() {
            Params p;
            p.i0 = static_cast<std::uint32_t>(rows);
            p.i1 = static_cast<std::uint32_t>(cols);
            p.f0 = lr;

            WGPUBuffer bufW = uploadBuffer(W, wBytes);
            WGPUBuffer bufD = uploadBuffer(delta, dBytes);
            WGPUBuffer bufP = uploadBuffer(prev, pBytes);
            std::vector<Output> outputs{{bufW, wBytes, W, nullptr}};

            const bool ok = runKernel(g_ctx.rank1, p, {{bufW, wBytes}, {bufD, dBytes}, {bufP, pBytes}},
                                      static_cast<std::uint32_t>((cols + 15) / 16),
                                      static_cast<std::uint32_t>((rows + 15) / 16), outputs);

            releaseBuffer(bufW, wBytes);
            releaseBuffer(bufD, dBytes);
            releaseBuffer(bufP, pBytes);
            return ok;
        });
    }

    bool DistanceAll(const float* q, const float* table, float* out, std::size_t rows,
                     std::size_t cols) {
        if (q == nullptr || table == nullptr || out == nullptr || rows == 0 || cols == 0) {
            return false;
        }
        const std::uint64_t qBytes = static_cast<std::uint64_t>(cols) * sizeof(float);
        const std::uint64_t tBytes = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
        const std::uint64_t outBytes = static_cast<std::uint64_t>(rows) * sizeof(float);
        return guarded(qBytes + tBytes + 2 * outBytes, [&]() {
            Params p;
            p.i0 = static_cast<std::uint32_t>(rows);
            p.i1 = static_cast<std::uint32_t>(cols);

            WGPUBuffer bufQ = uploadBuffer(q, qBytes);
            WGPUBuffer bufT = uploadBuffer(table, tBytes);
            WGPUBuffer bufOut = uploadBuffer(nullptr, outBytes);
            std::vector<Output> outputs{{bufOut, outBytes, out, nullptr}};

            const bool ok = runKernel(g_ctx.distance, p,
                                      {{bufQ, qBytes}, {bufT, tBytes}, {bufOut, outBytes}},
                                      static_cast<std::uint32_t>((rows + 63) / 64), 1, outputs);

            releaseBuffer(bufQ, qBytes);
            releaseBuffer(bufT, tBytes);
            releaseBuffer(bufOut, outBytes);
            return ok;
        });
    }

    bool PairUpdate(float* a, float* b, float g, std::size_t n) {
        if (a == nullptr || b == nullptr || n == 0) {
            return false;
        }
        const std::uint64_t bytes = static_cast<std::uint64_t>(n) * sizeof(float);
        return guarded(4 * bytes, [&]() {
            const std::uint32_t groups = static_cast<std::uint32_t>(
                std::min<std::size_t>(65535, std::max<std::size_t>(1, (n + 63) / 64)));
            Params p;
            p.i0 = static_cast<std::uint32_t>(n);
            p.i1 = groups * 64u;
            p.f0 = g;

            WGPUBuffer bufA = uploadBuffer(a, bytes);
            WGPUBuffer bufB = uploadBuffer(b, bytes);
            std::vector<Output> outputs{{bufA, bytes, a, nullptr}, {bufB, bytes, b, nullptr}};

            const bool ok = runKernel(g_ctx.paired, p, {{bufA, bytes}, {bufB, bytes}}, groups, 1, outputs);

            releaseBuffer(bufA, bytes);
            releaseBuffer(bufB, bytes);
            return ok;
        });
    }

}  // namespace Wuki::Gpu

#else  // 没编进 WebGPU 支持：全部返回 false，调用方自然走 SIMD

namespace Wuki::Gpu {

    bool Available() {
        return false;
    }
    const char* Error() {
        return "编译时未启用 WebGPU（WUKI_WEBGPU=OFF 或没找到 wgpu-native）";
    }
    std::string Info() {
        return Error();
    }
    Mode GetMode() {
        return Mode::Off;
    }
    void SetMode(Mode) {}
    std::size_t MinElements() {
        return 0;
    }
    void SetMinElements(std::size_t) {}
    bool ShouldUse(std::size_t) {
        return false;
    }
    void ConfigureFromEnv() {}
    Stats GetStats() {
        return {};
    }
    void ResetStats() {}
    void Shutdown() {}

    bool Dot(const float*, const float*, std::size_t, float*) {
        return false;
    }
    bool Axpy(float*, const float*, float, std::size_t) {
        return false;
    }
    bool Gemv(const float*, const float*, const float*, float*, std::size_t, std::size_t) {
        return false;
    }
    bool Gemt(const float*, const float*, const float*, float*, std::size_t, std::size_t) {
        return false;
    }
    bool Rank1(float*, const float*, const float*, float, std::size_t, std::size_t) {
        return false;
    }
    bool DistanceAll(const float*, const float*, float*, std::size_t, std::size_t) {
        return false;
    }
    bool PairUpdate(float*, float*, float, std::size_t) {
        return false;
    }

}  // namespace Wuki::Gpu

#endif
