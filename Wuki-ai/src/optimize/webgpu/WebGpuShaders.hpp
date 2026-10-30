#pragma once
#ifndef WUKI_OPTIMIZE_WEBGPU_SHADERS_HPP
#define WUKI_OPTIMIZE_WEBGPU_SHADERS_HPP

/*
 * 全部计算内核的 WGSL 源码（一个 module，多个 entry point）
 *
 * 为什么放一个 module：wgpu-native 里创建 shader module 是有成本的（编译 + 反射），
 * 7 个内核共用一份源码只编一次，管线各自取 entry point 就行。
 *
 * 绑定约定（C++ 侧的 runKernel 按这个顺序建 bind group）：
 *   binding 0   uniform Params   { i0, i1, f0, i2 }（16 字节，够所有内核用）
 *   binding 1+  storage buffer   各内核对应用到的输入 / 输出
 *
 * 每个 entry point 只声明自己真的会用的 binding —— 管线用的是 auto layout，
 * 没用到的 binding 不会进 layout，多塞一个反而会校验失败。
 *
 * 数值语义和 CPU 侧（Simd::Runtime / Simd::StudyTime）逐元素一致：
 *   - 不做 fast-math、不用 f16；f32 运算顺序可能和 CPU 不同（并行归约），
 *     最后一两位 ulp 的差异是预期的，不要拿它做逐位复现的断言；
 *   - 越界访问靠 WGSL 的 robustness 保证（读到 0 / 写被丢弃），外层 all 判断过了才会 dispatch。
 */
inline constexpr const char* kWukiWgsl = R"WGSL(
struct Params {
    i0: u32,
    i1: u32,
    f0: f32,
    i2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;

// ---------------------------------------------------------------- dot
// 两趟归约：dot_partial 每个 workgroup 出一份部分和，dot_reduce 再把它们加起来。
// 单 workgroup 的版本在 4M 元素上只有 256 个线程，带宽吃不满，实测比 SIMD 还慢。
@group(0) @binding(1) var<storage, read> dot_a: array<f32>;
@group(0) @binding(2) var<storage, read> dot_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dot_out: array<f32>;

var<workgroup> dot_scratch: array<f32, 256>;

// 趟 1：i0 = n，i1 = workgroup 个数（跨步 = 个数 × 256）
@compute @workgroup_size(256)
fn dot_partial(@builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let n = P.i0;
    let stride = P.i1 * 256u;
    var acc = 0.0;
    var i = wid.x * 256u + lid.x;
    loop {
        if (i >= n) { break; }
        acc = acc + dot_a[i] * dot_b[i];
        i = i + stride;
    }

    dot_scratch[lid.x] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            dot_scratch[lid.x] = dot_scratch[lid.x] + dot_scratch[lid.x + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid.x == 0u) {
        dot_out[wid.x] = dot_scratch[0];
    }
}

// 趟 2：i0 = 部分和的个数（独立的 binding，避开 auto layout 的编号空洞）
@group(0) @binding(1) var<storage, read> dot_red_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> dot_red_out: array<f32>;

@compute @workgroup_size(256)
fn dot_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = P.i0;
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= n) { break; }
        acc = acc + dot_red_in[i];
        i = i + 256u;
    }

    dot_scratch[lid.x] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            dot_scratch[lid.x] = dot_scratch[lid.x] + dot_scratch[lid.x + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid.x == 0u) {
        dot_red_out[0] = dot_scratch[0];
    }
}

// ---------------------------------------------------------------- axpy
@group(0) @binding(1) var<storage, read_write> axpy_y: array<f32>;
@group(0) @binding(2) var<storage, read> axpy_x: array<f32>;

// i0 = n，i1 = 总线程数（workgroup 数 × 64）；超过 65535 个 workgroup 时分多趟跨步
@compute @workgroup_size(64)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = P.i0;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        axpy_y[i] = axpy_y[i] + P.f0 * axpy_x[i];
        i = i + P.i1;
    }
}

// ---------------------------------------------------------------- gemv
// y[j] = b[j] + Σ_i W[i*cols + j] * x[i]   （W 行主序 [rows][cols]）
@group(0) @binding(1) var<storage, read> gemv_w: array<f32>;
@group(0) @binding(2) var<storage, read> gemv_x: array<f32>;
@group(0) @binding(3) var<storage, read> gemv_b: array<f32>;
@group(0) @binding(4) var<storage, read_write> gemv_y: array<f32>;

@compute @workgroup_size(64)
fn gemv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = P.i0;
    let cols = P.i1;
    let j = gid.x;
    if (j >= cols) { return; }

    var acc = gemv_b[j];
    for (var i = 0u; i < rows; i = i + 1u) {
        acc = acc + gemv_w[i * cols + j] * gemv_x[i];
    }
    gemv_y[j] = acc;
}

// ---------------------------------------------------------------- gemt
// out[m] = scale[m] * Σ_r W[r*cols + m] * u[r]
@group(0) @binding(1) var<storage, read> gemt_w: array<f32>;
@group(0) @binding(2) var<storage, read> gemt_u: array<f32>;
@group(0) @binding(3) var<storage, read> gemt_scale: array<f32>;
@group(0) @binding(4) var<storage, read_write> gemt_out: array<f32>;

@compute @workgroup_size(64)
fn gemt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = P.i0;
    let cols = P.i1;
    let m = gid.x;
    if (m >= cols) { return; }

    var acc = 0.0;
    for (var r = 0u; r < rows; r = r + 1u) {
        acc = acc + gemt_w[r * cols + m] * gemt_u[r];
    }
    gemt_out[m] = acc * gemt_scale[m];
}

// ---------------------------------------------------------------- rank1
// W[r*cols + m] += lr * delta[r] * prev[m]
@group(0) @binding(1) var<storage, read_write> rank1_w: array<f32>;
@group(0) @binding(2) var<storage, read> rank1_delta: array<f32>;
@group(0) @binding(3) var<storage, read> rank1_prev: array<f32>;

@compute @workgroup_size(16, 16)
fn rank1_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = P.i0;
    let cols = P.i1;
    let r = gid.y;
    let m = gid.x;
    if (r >= rows || m >= cols) { return; }

    let idx = r * cols + m;
    rank1_w[idx] = rank1_w[idx] + P.f0 * rank1_delta[r] * rank1_prev[m];
}

// ---------------------------------------------------------------- distance
// out[j] = Σ_d (q[d] - table[j*cols + d])²
@group(0) @binding(1) var<storage, read> dist_q: array<f32>;
@group(0) @binding(2) var<storage, read> dist_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> dist_out: array<f32>;

@compute @workgroup_size(64)
fn distance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = P.i0;
    let cols = P.i1;
    let j = gid.x;
    if (j >= rows) { return; }

    var acc = 0.0;
    for (var d = 0u; d < cols; d = d + 1u) {
        let diff = dist_q[d] - dist_table[j * cols + d];
        acc = acc + diff * diff;
    }
    dist_out[j] = acc;
}

// ---------------------------------------------------------------- paired
// a[i] += g*b[i]; b[i] += g*a_old[i]（两边都用旧值）
@group(0) @binding(1) var<storage, read_write> pair_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> pair_b: array<f32>;

@compute @workgroup_size(64)
fn paired_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = P.i0;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        let old_a = pair_a[i];
        let old_b = pair_b[i];
        pair_a[i] = old_a + P.f0 * old_b;
        pair_b[i] = old_b + P.f0 * old_a;
        i = i + P.i1;
    }
}
)WGSL";

#endif  // WUKI_OPTIMIZE_WEBGPU_SHADERS_HPP
