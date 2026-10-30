// src/helpers/hip_helper.cu - Frankenstein: the HIP helper worker for an AMD card beside the CUDA cards.
//
// The device side of RemoteExperts (src/core/remote_experts.cpp), lifted behind the worker C ABI
// (strata/helper/strata_helper_abi.h) and compiled as HIP: this file is written in Strata's CUDA shape and the
// force-included hip_compat/cuda_runtime.h renames every cuda* call to hip*, exactly as the rest of the HIP build.
// It lives in its own shared library with every symbol hidden but the entry, so the CUDA engine that dlopens it
// never sees a second copy of the kernels or the HIP runtime's types.
//
// Per layer: the engine hands grouped slot work (slot, start, tok per group); this stages the activation rows, the
// group metadata and runs the same grouped expert kernels the CUDA helper runs, then copies the packed output rows
// back into its pinned host buffer.  finish() waits for the stream and hands that buffer to the engine.
//
// What is deliberately NOT here: zero-copy mapped host buffers (RemoteExperts' STRATA_REMOTE_ZEROCOPY), because the
// mapped path on gfx1030 is unmeasured (the fork note), and the engine's pinned arena, which belongs to the CUDA
// runtime - blobs arrive as pageable host pointers and are staged by hipMemcpy.
#include "strata/helper/strata_helper_abi.h"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
#if defined(STRATA_USE_HIP)
constexpr const char* kBackend = "hip";
#else
constexpr const char* kBackend = "cuda-worker";   // the same file built with nvcc: a CUDA worker for the loader's tests
#endif

struct Meta {   // one contiguous upload per layer, as RemoteMeta
    unsigned long long* ptr;
    int32_t* start;
    int32_t* dst;
    int32_t* tok;
    int32_t* count;
};

int fail(char* err, size_t cap, const std::string& msg, int code = 1) {
    if (err && cap) std::snprintf(err, cap, "%s", msg.c_str());
    return code;
}

bool check(cudaError_t st, const char* what, char* err, size_t cap) {
    if (st == cudaSuccess) return true;
    fail(err, cap, std::string(kBackend) + " worker: " + what + ": " + cudaGetErrorString(st));
    return false;
}

struct DeviceScope {
    int previous = -1; bool ok = false; cudaError_t status = cudaSuccess;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) { previous = -1; return; }
        status = cudaSetDevice(device);
        ok = status == cudaSuccess;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
};
}  // namespace

struct strata_helper_device {
    int device = -1;
    int64_t n_embd = 0, n_ff = 0, max_entries = 0, max_tokens = 0;
    bool native = false;
    // the slot arena
    uint8_t* arena = nullptr;
    std::vector<uint64_t> slot_off;      // n_slots + 1
    std::vector<int64_t> slot_bytes;
    uint64_t arena_bytes = 0;
    // streams and buffers
    cudaStream_t stream = nullptr;
    float* h_x = nullptr;                // pinned: max_tokens * n_embd
    float* h_out = nullptr;              // pinned: max_entries * n_embd
    float* d_x = nullptr;
    float* d_out = nullptr;
    uint8_t* d_q8 = nullptr;
    float* d_scales = nullptr;
    void* d_scratch = nullptr;
    void* d_meta = nullptr; size_t meta_bytes = 0;
    void* h_meta = nullptr;              // pinned staging of the metadata
    Meta dm{}, hm{};
    int64_t cap = 0;                     // max_entries
    // per launch
    int32_t n_entries = 0;
    bool pending = false;
    std::vector<uint8_t> staging;        // verify read-back
    // STRATA_HELPER_TIMING=1: per-launch phase times on stderr (events around the copies and the kernels)
    bool timing = false;
    cudaEvent_t ev[4] = {};
    double stage_us = 0.0;
    int32_t timed_tok = 0;
};

namespace {
size_t meta_layout(int64_t cap, Meta& m, uint8_t* base) {
    size_t off = 0;
    auto take = [&](size_t bytes) { uint8_t* p = base + off; off += (bytes + 255) / 256 * 256; return p; };
    m.ptr = (unsigned long long*) take((size_t) cap * sizeof(unsigned long long));
    m.start = (int32_t*) take((size_t) (cap + 1) * sizeof(int32_t));
    m.dst = (int32_t*) take((size_t) cap * sizeof(int32_t));
    m.tok = (int32_t*) take((size_t) cap * sizeof(int32_t));
    m.count = (int32_t*) take(sizeof(int32_t));
    return off;
}

int w_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return 0; }
    return n;
}

int w_device_name(int device, char* name, size_t cap) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { cudaGetLastError(); return 1; }
    std::snprintf(name, cap, "%s", prop.name);
    return 0;
}

int w_preflight(int device, double* free_gib, char* err, size_t cap) {
    int n = 0;
    if (!check(cudaGetDeviceCount(&n), "cudaGetDeviceCount", err, cap)) return 1;
    if (device < 0 || device >= n) return fail(err, cap, std::string(kBackend) + " worker: device " + std::to_string(device) + " is not visible");
    DeviceScope scope(device);
    if (!scope.ok) return fail(err, cap, std::string(kBackend) + " worker: cudaSetDevice: " + cudaGetErrorString(scope.status));
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err, cap)) return 1;
    if (free_gib) *free_gib = (double) free_b / 1073741824.0;
    return 0;
}

void release(strata_helper_device* d) {
    if (!d) return;
    DeviceScope scope(d->device);
    if (scope.ok) {
        if (d->stream) cudaStreamSynchronize(d->stream);
        if (d->arena) cudaFree(d->arena);
        if (d->d_x) cudaFree(d->d_x);
        if (d->d_out) cudaFree(d->d_out);
        if (d->d_q8) cudaFree(d->d_q8);
        if (d->d_scales) cudaFree(d->d_scales);
        if (d->d_scratch) cudaFree(d->d_scratch);
        if (d->d_meta) cudaFree(d->d_meta);
        if (d->h_x) cudaFreeHost(d->h_x);
        if (d->h_out) cudaFreeHost(d->h_out);
        if (d->h_meta) cudaFreeHost(d->h_meta);
        for (cudaEvent_t& e : d->ev) if (e) cudaEventDestroy(e);
        if (d->stream) cudaStreamDestroy(d->stream);
    }
    delete d;
}

int w_open(int device, const strata_helper_open_params* p, strata_helper_device** out, char* err, size_t cap) {
    if (!p || !out || p->n_slots <= 0 || !p->slot_bytes || p->max_entries <= 0 || p->max_tokens <= 0 || p->n_embd <= 0 || p->n_ff <= 0)
        return fail(err, cap, std::string(kBackend) + " worker: bad open parameters");
    int n = 0;
    if (!check(cudaGetDeviceCount(&n), "cudaGetDeviceCount", err, cap)) return 1;
    if (device < 0 || device >= n) return fail(err, cap, std::string(kBackend) + " worker: device is not visible");
    auto d = std::unique_ptr<strata_helper_device, void (*)(strata_helper_device*)>(new strata_helper_device(), release);
    d->device = device;
    d->n_embd = p->n_embd; d->n_ff = p->n_ff; d->max_entries = p->max_entries; d->max_tokens = p->max_tokens;
    d->native = p->native != 0;
    d->cap = p->max_entries;
    DeviceScope scope(device);
    if (!scope.ok) return fail(err, cap, std::string(kBackend) + " worker: cudaSetDevice: " + cudaGetErrorString(scope.status));
    d->slot_off.resize((size_t) p->n_slots + 1);
    d->slot_bytes.assign(p->slot_bytes, p->slot_bytes + p->n_slots);
    uint64_t off = 0;
    for (int64_t s = 0; s < p->n_slots; ++s) {
        if (p->slot_bytes[s] <= 0) return fail(err, cap, std::string(kBackend) + " worker: a slot of no bytes");
        d->slot_off[(size_t) s] = off;
        off += ((uint64_t) p->slot_bytes[s] + 255) / 256 * 256;
    }
    d->slot_off[(size_t) p->n_slots] = off;
    d->arena_bytes = off;
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err, cap)) return 1;
    if (off + (512ull << 20) > free_b)
        return fail(err, cap, std::string(kBackend) + " worker: " + std::to_string(off >> 20) + " MiB of slots leave less than 512 MiB free");
    const size_t H = (size_t) p->n_embd, FF = (size_t) p->n_ff, CAP = (size_t) p->max_entries, T = (size_t) p->max_tokens;
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) CAP, (int64_t) H, (int64_t) FF),
                                            strata::kernels::native_expert_scratch_bytes((int64_t) CAP, (int64_t) FF));
    Meta probe{}; d->meta_bytes = meta_layout((int64_t) CAP, probe, nullptr);
    const bool ok =
        check(cudaMalloc((void**) &d->arena, off), "slot arena", err, cap) &&
        check(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking), "stream", err, cap) &&
        check(cudaHostAlloc((void**) &d->h_x, T * H * sizeof(float), cudaHostAllocPortable), "input staging", err, cap) &&
        check(cudaHostAlloc((void**) &d->h_out, CAP * H * sizeof(float), cudaHostAllocPortable), "result staging", err, cap) &&
        check(cudaHostAlloc(&d->h_meta, d->meta_bytes, cudaHostAllocPortable), "metadata staging", err, cap) &&
        check(cudaMalloc((void**) &d->d_x, T * H * sizeof(float)), "input", err, cap) &&
        check(cudaMalloc((void**) &d->d_out, CAP * H * sizeof(float)), "result", err, cap) &&
        check(cudaMalloc((void**) &d->d_q8, CAP * (H / 32) * 36), "activation", err, cap) &&
        check(cudaMalloc((void**) &d->d_scales, CAP * (H / 32) * sizeof(float)), "activation scales", err, cap) &&
        check(cudaMalloc(&d->d_scratch, scratch), "scratch", err, cap) &&
        check(cudaMalloc(&d->d_meta, d->meta_bytes), "group metadata", err, cap);
    if (!ok) return 1;
    if (const char* t = std::getenv("STRATA_HELPER_TIMING"); t && t[0] == '1') {
        d->timing = true;
        for (cudaEvent_t& e : d->ev)
            if (!check(cudaEventCreate(&e), "timing event", err, cap)) return 1;
    }
    meta_layout((int64_t) CAP, d->dm, (uint8_t*) d->d_meta);
    meta_layout((int64_t) CAP, d->hm, (uint8_t*) d->h_meta);
    *out = d.release();
    return 0;
}

void w_close(strata_helper_device* d) { release(d); }

bool slot_ok(const strata_helper_device* d, int32_t slot, int64_t bytes) {
    return d && slot >= 0 && (size_t) slot < d->slot_bytes.size() && bytes > 0 && bytes <= d->slot_bytes[(size_t) slot];
}

int w_fill_slot(strata_helper_device* d, int32_t slot, const uint8_t* blob, int64_t bytes, char* err, size_t cap) {
    if (!slot_ok(d, slot, bytes) || !blob) return fail(err, cap, std::string(kBackend) + " worker: fill of a bad slot or size");
    DeviceScope scope(d->device);
    if (!scope.ok) return fail(err, cap, std::string(kBackend) + " worker: device switch failed");
    // blocking, from pageable host memory: the engine's arena is the CUDA runtime's, never registered here
    return check(cudaMemcpy(d->arena + d->slot_off[(size_t) slot], blob, (size_t) bytes, cudaMemcpyHostToDevice), "fill", err, cap) ? 0 : 1;
}

int w_read_slot(strata_helper_device* d, int32_t slot, uint8_t* dst, int64_t bytes, char* err, size_t cap) {
    if (!slot_ok(d, slot, bytes) || !dst) return fail(err, cap, std::string(kBackend) + " worker: read of a bad slot or size");
    DeviceScope scope(d->device);
    if (!scope.ok) return fail(err, cap, std::string(kBackend) + " worker: device switch failed");
    if (d->pending && !check(cudaStreamSynchronize(d->stream), "read wait", err, cap)) return 1;
    return check(cudaMemcpy(dst, d->arena + d->slot_off[(size_t) slot], (size_t) bytes, cudaMemcpyDeviceToHost), "read", err, cap) ? 0 : 1;
}

int w_verify_slot(strata_helper_device* d, int32_t slot, const uint8_t* blob, int64_t bytes, char* err, size_t cap) {
    if (!slot_ok(d, slot, bytes) || !blob) return fail(err, cap, std::string(kBackend) + " worker: verify of a bad slot or size");
    d->staging.resize((size_t) bytes);
    const int rc = w_read_slot(d, slot, d->staging.data(), bytes, err, cap);
    if (rc) return rc;
    for (int64_t i = 0; i < bytes; ++i)
        if (d->staging[(size_t) i] != blob[i])
            return fail(err, cap, std::string(kBackend) + " worker: slot " + std::to_string(slot) + " differs from the arena at byte " + std::to_string(i), 2);
    return 0;
}

int w_begin(strata_helper_device* d, const strata_helper_launch* L, char* err, size_t cap) {
    if (!d || !L || !L->x || L->n_tok <= 0 || L->n_tok > d->max_tokens || L->n_entries < 0 || L->n_entries > d->max_entries ||
        L->n_groups < 0 || L->n_groups > d->max_entries)
        return fail(err, cap, std::string(kBackend) + " worker: a launch outside the opened bounds");
    d->n_entries = L->n_entries;
    d->pending = false;
    if (L->n_entries == 0) { d->pending = true; return 0; }
    const size_t H = (size_t) d->n_embd;
    const auto t_stage = std::chrono::steady_clock::now();
    std::memcpy(d->h_x, L->x, (size_t) L->n_tok * H * sizeof(float));
    // The S2 grouped kernels keep at most GMAX = 8 entries of a group and drop the rest without a word
    // (s2_expert_grouped.cu); the engine never sends more (a verify window is <= 8 tokens), but a longer window
    // must not come back with stale rows, so an S2 group is cut into runs of 8 here, the same slot repeated.
    // Each entry's arithmetic does not depend on its run, so the rows are those of the 8-token windows.
    constexpr int32_t kS2GroupMax = 8;
    int32_t n_groups = 0;
    if (L->start[0] != 0 || L->start[L->n_groups] != L->n_entries)
        return fail(err, cap, std::string(kBackend) + " worker: the groups do not cover the launch's entries");
    for (int32_t g = 0; g < L->n_groups; ++g) {
        const int32_t slot = L->slot[g];
        if (slot < 0 || (size_t) slot >= d->slot_bytes.size()) return fail(err, cap, std::string(kBackend) + " worker: a launch names a slot that does not exist");
        if (L->start[g + 1] < L->start[g]) return fail(err, cap, std::string(kBackend) + " worker: group starts out of order");
        const unsigned long long ptr = (unsigned long long) (d->arena + d->slot_off[(size_t) slot]);
        const int32_t run = d->native ? std::max<int32_t>(1, L->start[g + 1] - L->start[g]) : kS2GroupMax;
        for (int32_t at = L->start[g]; at < L->start[g + 1] || at == L->start[g]; at += run) {
            if (n_groups >= d->cap) return fail(err, cap, std::string(kBackend) + " worker: more groups than the opened entries");
            d->hm.ptr[n_groups] = ptr;
            d->hm.start[n_groups] = at;
            ++n_groups;
            if (at + run >= L->start[g + 1]) break;
        }
    }
    d->hm.start[n_groups] = L->n_entries;
    for (int32_t i = 0; i < L->n_entries; ++i) {
        if (L->tok[i] < 0 || L->tok[i] >= L->n_tok) return fail(err, cap, std::string(kBackend) + " worker: a launch entry names a token outside the window");
        d->hm.dst[i] = i;   // packed output rows, in entry order
        d->hm.tok[i] = L->tok[i];
    }
    *d->hm.count = n_groups;
    DeviceScope scope(d->device);
    if (!scope.ok) return fail(err, cap, std::string(kBackend) + " worker: device switch failed");
    const cudaStream_t s = d->stream;
    if (d->timing) {
        d->stage_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_stage).count();
        d->timed_tok = (int32_t) L->n_tok;
        cudaEventRecord(d->ev[0], s);
    }
    if (!check(cudaMemcpyAsync(d->d_x, d->h_x, (size_t) L->n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, cap) ||
        !check(cudaMemcpyAsync(d->d_meta, d->h_meta, d->meta_bytes, cudaMemcpyHostToDevice, s), "copy group metadata", err, cap))
        return 1;
    if (d->timing) cudaEventRecord(d->ev[1], s);
    if (d->native) {
        strata::kernels::quantize_q8_1_rows(d->d_x, L->n_tok, (int64_t) H, d->d_q8, s);
        auto lay = strata::kernels::native_expert_layout(L->gu_type, L->d_type, d->n_embd, d->n_ff);
        strata::kernels::native_expert_grouped(lay, d->dm.ptr, d->dm.start, d->dm.count, d->dm.dst, d->dm.tok,
                                               n_groups, (int64_t) L->n_entries, d->d_q8, d->d_scratch, d->d_out, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(d->d_x, d->d_q8, d->d_scales, L->n_tok * (int64_t) H, s);
        strata::kernels::moe_grouped_s2(d->dm.ptr, d->dm.start, d->dm.count, d->dm.dst, d->dm.tok,
                                        n_groups, (int64_t) L->n_entries, d->d_q8, d->d_scales, d->d_scratch, d->d_out, s);
    }
    if (!check(cudaGetLastError(), "expert kernels", err, cap)) return 1;
    if (d->timing) cudaEventRecord(d->ev[2], s);
    if (!check(cudaMemcpyAsync(d->h_out, d->d_out, (size_t) L->n_entries * H * sizeof(float), cudaMemcpyDeviceToHost, s), "copy results", err, cap))
        return 1;
    if (d->timing) cudaEventRecord(d->ev[3], s);
    d->pending = true;
    return 0;
}

int w_finish(strata_helper_device* d, const float** rows, char* err, size_t cap) {
    if (!d || !rows) return fail(err, cap, std::string(kBackend) + " worker: bad finish arguments");
    if (!d->pending) return fail(err, cap, std::string(kBackend) + " worker: finish without a begin");
    d->pending = false;
    if (d->n_entries > 0) {
        DeviceScope scope(d->device);
        if (!scope.ok) return fail(err, cap, std::string(kBackend) + " worker: device switch failed");
        if (!check(cudaStreamSynchronize(d->stream), "finish", err, cap)) return 1;
        if (d->timing) {
            float ms[3] = {};
            for (int i = 0; i < 3; ++i) cudaEventElapsedTime(&ms[i], d->ev[i], d->ev[i + 1]);
            std::fprintf(stderr, "helper-timing backend=%s n_tok=%d entries=%d stage_us=%.1f h2d_us=%.1f kernel_us=%.1f d2h_us=%.1f\n",
                         kBackend, (int) d->timed_tok, (int) d->n_entries, d->stage_us, 1e3 * ms[0], 1e3 * ms[1], 1e3 * ms[2]);
        }
    }
    *rows = d->h_out;
    return 0;
}

double w_gib(strata_helper_device* d) { return d ? (double) d->arena_bytes / 1073741824.0 : 0.0; }

const strata_helper_api kApi = {
    STRATA_HELPER_ABI_VERSION, (uint32_t) sizeof(strata_helper_api), kBackend,
    w_device_count, w_device_name, w_preflight, w_open, w_close,
    w_fill_slot, w_verify_slot, w_read_slot, w_begin, w_finish, w_gib,
};
}  // namespace

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const strata_helper_api* strata_helper_api_v1(void) { return &kApi; }
