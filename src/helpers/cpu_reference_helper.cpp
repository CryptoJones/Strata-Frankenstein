// src/helpers/cpu_reference_helper.cpp - Frankenstein: the CPU reference worker.
//
// A helper worker library (strata/helper/strata_helper_abi.h) whose "device" is host memory and whose compute is
// the engine's own CPU expert kernels, one entry at a time.  It exists so the engine side of a foreign helper -
// the loader, the slot table, the grouping, the row return - can be run and compared on a machine with no second
// GPU at all: with this worker enabled under --helper-shadow, every token must be BIT-IDENTICAL to a run without
// it, because the rows it computes are the rows the CPU pool would otherwise have computed with the same functions.
// (Without --helper-shadow the worker is a helper like CUDA1: it is striped with the CUDA helpers' experts and its
// experts leave the GPU plan's PCIe share, so the run legitimately computes other experts on other devices.)  Slow by design
// (no threads, no overlap): a correctness instrument, not a tier anyone should serve from.
#include "strata/helper/strata_helper_abi.h"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {
using namespace strata::kernels::cpu;

constexpr int kVisible = 1;   // one "device": this process's memory

struct Slot {
    std::unique_ptr<uint8_t[]> bytes;
    int64_t size = 0;
};

int fail(char* err, size_t cap, const std::string& msg, int code = 1) {
    if (err && cap) std::snprintf(err, cap, "%s", msg.c_str());
    return code;
}
}  // namespace

struct strata_helper_device {
    std::vector<Slot> slots;
    int64_t total = 0;
    int64_t n_embd = 0, n_ff = 0, max_entries = 0, max_tokens = 0;
    bool native = false;
    // per-launch state
    std::vector<float> rows;          // n_entries * n_embd, the packed result
    std::vector<float> x;             // the activation rows, copied at begin
    // native scratch, one row per token of the window: an expert is applied to ALL its tokens in one call, as the
    // pool's ExpertJobMulti does - some gate/up types take a multi-token kernel from native_gu_mt_min tokens on,
    // which is not bit-equal to the per-token kernel (#152), so one entry at a time would not match the pool
    NativeFmt fmt{};
    int fmt_gu = -1, fmt_d = -1;
    std::vector<uint8_t> act, hq;        // max_tokens * kNativeActBytes / kNativeHBytes
    std::vector<float> ff;               // max_tokens * n_ff
    std::vector<uint8_t> act_done;       // per token of this launch: activation already quantized
    // a native Q2_0 layer (q2_native_kernels): the pool's Q2_0 activations instead of the packed native ones
    std::vector<ActQ> act_q2, h_q2;      // max_tokens each
    std::vector<float> gbuf, ubuf;       // max_tokens * n_ff, the Q2_0 gate and up rows before SwiGLU
    // S2 scratch
    ActQ act_q{};
    ExpertScratch ws{};
    bool pending = false;
    std::string pending_err;
};

namespace {
int w_device_count(void) { return kVisible; }

int w_device_name(int device, char* name, size_t cap) {
    if (device != 0) return 1;
    std::snprintf(name, cap, "cpu-reference (host memory)");
    return 0;
}

int w_preflight(int device, double* free_gib, char* err, size_t cap) {
    if (device != 0) return fail(err, cap, "cpu-reference: only device 0 exists");
    if (free_gib) *free_gib = 64.0;   // a notional budget: the engine sizes an `auto` tier against this
    return 0;
}

int w_open(int device, const strata_helper_open_params* p, strata_helper_device** out, char* err, size_t cap) {
    if (device != 0 || !p || !out) return fail(err, cap, "cpu-reference: bad open arguments");
    if (p->n_embd != H || p->n_ff != FF) return fail(err, cap, "cpu-reference: the CPU kernels are built for H=2560, FF=640");
    auto dev = std::make_unique<strata_helper_device>();
    dev->n_embd = p->n_embd; dev->n_ff = p->n_ff;
    dev->max_entries = p->max_entries; dev->max_tokens = p->max_tokens;
    dev->native = p->native != 0;
    dev->slots.resize((size_t) p->n_slots);
    for (int64_t s = 0; s < p->n_slots; ++s) {
        const int64_t bytes = p->slot_bytes[s];
        if (bytes <= 0) return fail(err, cap, "cpu-reference: a slot of no bytes");
        dev->slots[(size_t) s].bytes.reset(new (std::nothrow) uint8_t[(size_t) bytes]);
        if (!dev->slots[(size_t) s].bytes) return fail(err, cap, "cpu-reference: out of host memory for slot " + std::to_string(s));
        dev->slots[(size_t) s].size = bytes;
        dev->total += bytes;
    }
    dev->rows.assign((size_t) (p->max_entries * p->n_embd), 0.0f);
    dev->x.assign((size_t) (p->max_tokens * p->n_embd), 0.0f);
    dev->act.assign((size_t) p->max_tokens * kNativeActBytes, 0);
    dev->hq.assign((size_t) p->max_tokens * kNativeHBytes, 0);
    dev->ff.assign((size_t) p->max_tokens * (size_t) p->n_ff, 0.0f);
    dev->act_done.assign((size_t) p->max_tokens, 0);
    dev->act_q2.resize((size_t) p->max_tokens);
    dev->h_q2.resize((size_t) p->max_tokens);
    dev->gbuf.assign((size_t) p->max_tokens * (size_t) p->n_ff, 0.0f);
    dev->ubuf.assign((size_t) p->max_tokens * (size_t) p->n_ff, 0.0f);
    *out = dev.release();
    return 0;
}

void w_close(strata_helper_device* dev) { delete dev; }

bool slot_ok(const strata_helper_device* dev, int32_t slot, int64_t bytes) {
    return dev && slot >= 0 && (size_t) slot < dev->slots.size() && bytes > 0 && bytes <= dev->slots[(size_t) slot].size;
}

int w_fill_slot(strata_helper_device* dev, int32_t slot, const uint8_t* blob, int64_t bytes, char* err, size_t cap) {
    if (!slot_ok(dev, slot, bytes) || !blob) return fail(err, cap, "cpu-reference: fill of a bad slot or size");
    std::memcpy(dev->slots[(size_t) slot].bytes.get(), blob, (size_t) bytes);
    return 0;
}

int w_verify_slot(strata_helper_device* dev, int32_t slot, const uint8_t* blob, int64_t bytes, char* err, size_t cap) {
    if (!slot_ok(dev, slot, bytes) || !blob) return fail(err, cap, "cpu-reference: verify of a bad slot or size");
    const uint8_t* have = dev->slots[(size_t) slot].bytes.get();
    for (int64_t i = 0; i < bytes; ++i)
        if (have[i] != blob[i]) return fail(err, cap, "cpu-reference: slot " + std::to_string(slot) + " differs at byte " + std::to_string(i), 2);
    return 0;
}

int w_read_slot(strata_helper_device* dev, int32_t slot, uint8_t* dst, int64_t bytes, char* err, size_t cap) {
    if (!slot_ok(dev, slot, bytes) || !dst) return fail(err, cap, "cpu-reference: read of a bad slot or size");
    std::memcpy(dst, dev->slots[(size_t) slot].bytes.get(), (size_t) bytes);
    return 0;
}

// One expert for the `nt` tokens routed to it, native pack: the pool's row functions over every row, composed as
// run_split_multi_native composes them for one ExpertJobMulti (pool.cpp, phases 5 and 6) - gate/up rows for all the
// tokens, each token's intermediate quantized, down rows for all the tokens.  A Q2_0 layer on an AVX-2 CPU takes the
// pool's Q2_0 branch (q2_native_kernels): act_quant_any activations, q2_rows_any rows, SwiGLU after.  Bitwise the
// pool's result for the same (expert, tokens): every kernel here computes a row independently of the row range the
// pool's tasks split it into.
void native_group(strata_helper_device& d, const uint8_t* blob, const int32_t* toks, int nt, float* const* out_rows) {
    const NativeFmt& f = d.fmt;
    const bool q2_gu = q2_native_kernels(f.gu_type), q2_d = q2_native_kernels(f.d_type);
    const void* acts[MAXT]; const ActQ* acts_q2[MAXT]; float* ffs[MAXT];
    for (int t = 0; t < nt; ++t) {
        const int32_t tok = toks[t];
        uint8_t* a = d.act.data() + (size_t) tok * kNativeActBytes;
        if (!d.act_done[(size_t) tok]) {
            if (q2_gu) act_quant_any(d.x.data() + (size_t) tok * d.n_embd, (int) d.n_embd, d.act_q2[(size_t) tok]);
            else native_quant_act(f, d.x.data() + (size_t) tok * d.n_embd, a);
            d.act_done[(size_t) tok] = 1;
        }
        acts[t] = a;
        acts_q2[t] = &d.act_q2[(size_t) tok];
        ffs[t] = d.ff.data() + (size_t) t * d.n_ff;
    }
    if (q2_gu) {
        float* gp[MAXT]; float* up[MAXT];
        for (int t = 0; t < nt; ++t) {
            gp[t] = d.gbuf.data() + (size_t) t * d.n_ff;
            up[t] = d.ubuf.data() + (size_t) t * d.n_ff;
        }
        const int nbk = (int) (f.n_embd / 64);
        q2_rows_any(blob, f.gu_row, nbk, acts_q2, nt, gp, 0, (int) d.n_ff);
        q2_rows_any(blob + f.up_off, f.gu_row, nbk, acts_q2, nt, up, 0, (int) d.n_ff);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < (int) d.n_ff; ++r)
                ffs[t][r] = (gp[t][r] / (1.f + std::exp(-gp[t][r]))) * up[t][r];
    } else {
        native_gu_rows(f, blob, acts, nt, ffs, 0, (int) d.n_ff);
    }
    if (q2_d) {
        const ActQ* a2[MAXT];
        for (int t = 0; t < nt; ++t) { act_quant_any(ffs[t], (int) d.n_ff, d.h_q2[(size_t) t]); a2[t] = &d.h_q2[(size_t) t]; }
        q2_rows_any(blob + f.down_off, f.d_row, (int) (f.n_ff / 64), a2, nt, out_rows, 0, (int) d.n_embd);
    } else {
        const void* hqs[MAXT];
        for (int t = 0; t < nt; ++t) {
            uint8_t* h = d.hq.data() + (size_t) t * kNativeHBytes;
            native_quant_h(f, ffs[t], h);
            hqs[t] = h;
        }
        native_down_rows(f, blob, hqs, nt, out_rows, 0, (int) d.n_embd);
    }
}

int w_begin(strata_helper_device* dev, const strata_helper_launch* L, char* err, size_t cap) {
    if (!dev || !L) return fail(err, cap, "cpu-reference: bad begin arguments");
    if (L->n_entries < 0 || L->n_entries > dev->max_entries || L->n_tok <= 0 || L->n_tok > dev->max_tokens || !L->x)
        return fail(err, cap, "cpu-reference: a launch outside the opened bounds");
    std::memcpy(dev->x.data(), L->x, (size_t) (L->n_tok * dev->n_embd) * sizeof(float));
    if (dev->native) {
        if (L->gu_type != dev->fmt_gu || L->d_type != dev->fmt_d) {
            std::string why;
            if (!native_fmt(L->gu_type, L->d_type, dev->n_embd, dev->n_ff, dev->fmt, why))
                return fail(err, cap, "cpu-reference: " + why);
            dev->fmt_gu = L->gu_type; dev->fmt_d = L->d_type;
        }
    } else {
        // the S2 pack: one quantized activation per token would be shared across experts in the pool; here each
        // entry quantizes its own, which is the same bytes (the quantizer is a pure function of the row)
    }
    std::fill(dev->act_done.begin(), dev->act_done.end(), 0);
    for (int32_t g = 0; g < L->n_groups; ++g) {
        const int32_t slot = L->slot[g];
        if (slot < 0 || (size_t) slot >= dev->slots.size()) return fail(err, cap, "cpu-reference: a launch names a slot that does not exist");
        const uint8_t* blob = dev->slots[(size_t) slot].bytes.get();
        const int32_t first = L->start[g], last = L->start[g + 1];
        if (last - first > MAXT || last < first) return fail(err, cap, "cpu-reference: a group with more tokens than a window holds");
        for (int32_t i = first; i < last; ++i)
            if (L->tok[i] < 0 || L->tok[i] >= L->n_tok) return fail(err, cap, "cpu-reference: a launch entry names a token outside the window");
        if (dev->native) {
            float* outs[MAXT];
            for (int32_t i = first; i < last; ++i) outs[i - first] = dev->rows.data() + (size_t) i * dev->n_embd;
            native_group(*dev, blob, L->tok + first, (int) (last - first), outs);
        } else {
            for (int32_t i = first; i < last; ++i) {
                act_quant_q8_1(dev->x.data() + (size_t) L->tok[i] * dev->n_embd, (int) dev->n_embd, dev->act_q);
                s2_expert_vnni_q(blob, dev->act_q, dev->rows.data() + (size_t) i * dev->n_embd, dev->ws);
            }
        }
    }
    dev->pending = true;
    return 0;
}

int w_finish(strata_helper_device* dev, const float** rows, char* err, size_t cap) {
    if (!dev || !rows) return fail(err, cap, "cpu-reference: bad finish arguments");
    if (!dev->pending) return fail(err, cap, "cpu-reference: finish without a begin");
    dev->pending = false;
    *rows = dev->rows.data();
    return 0;
}

double w_gib(strata_helper_device* dev) { return dev ? (double) dev->total / 1073741824.0 : 0.0; }

const strata_helper_api kApi = {
    STRATA_HELPER_ABI_VERSION, (uint32_t) sizeof(strata_helper_api), "cpu-reference",
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
