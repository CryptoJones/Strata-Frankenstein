// tests/helper/hip_worker_parity.cpp - Frankenstein: the helper workers against each other on real expert blobs.
//
// Loads up to three worker libraries through strata_helper_api_v1 (dlopen, RTLD_LOCAL, as the engine's loader):
//   --cpu   libstrata_helper_cpu_reference.so   the CPU pool's own expert functions (NOT the GPU kernels' bytes:
//                                               the pool quantizes activations with ggml's vec_dot_type - Q8_K for
//                                               K-quants - where the GPU kernels use q8_1, so its rows differ from
//                                               every GPU worker's by design; --oracle measures both against FP32)
//   --hip   libstrata_helper_hip.so             the HIP worker (an AMD card)
//   --cuda  libstrata_helper_cuda_worker.so     the same hip_helper.cu built with nvcc: the CUDA helper's kernels
// fills every one of them with the same expert blobs, feeds the same activations and routing, and compares the
// packed output rows byte for byte.  Windows: 1 token (decode), 5 (an MTP verify window), 8 (the engine's
// MAXT), and prompt-sized batches (64, 512) beyond what the engine sends a helper today.
//
// The engine sends a helper at most MAXT = 8 tokens and groups entries one group per expert; the CPU reference
// refuses a group of more than MAXT tokens.  So a window above 8 tokens is sent split (a group's entries in
// chunks of at most 8, the same slot repeated) to every worker, and once more UNSPLIT to the GPU workers, whose
// per-entry sums must not depend on the group's size.
//
// Native blobs come from a directory of role slices cut from the GGUF (see --data): for each layer L,
//   L<L>.gate, L<L>.up, L<L>.down = the first N experts of that layer's ffn_{gate,up,down}_exps tensor
// and --layer L:GU:D names its ggml types (12 Q4_K, 13 Q5_K, 14 Q6_K, 7 Q5_1, 8 Q8_0, 6 Q5_0, 2 Q4_0).
// An expert's blob is gate | up | down, the native pack's layout.  --s2 adds synthetic Q2_0 (S2 pack) blobs.
//
// With the CUDA runtime linked (STRATA_PARITY_CUDART) the driver also initializes CUDA itself and keeps a
// copy loop running on the CUDA device while the HIP worker computes (--concurrent), the two-runtimes-in-one-
// process check.  Timing: per call wall times of begin() / finish(), a finish() on an already-finished stream
// (the synchronize latency), and with STRATA_HELPER_TIMING=1 the GPU workers print their own phase times.
//
// With the CPU reference loaded the exit status is 1 by design (see above); the verdict that matters is the
// "hip vs cuda" total.  Build both libraries with -DSTRATA_PORTABLE_EXPF=ON for those to be byte-identical (the
// hardware __expf differs between NVIDIA and AMD; see src/kernels/cuda/portable_expf.cuh).
//
// Exit: 0 all compared pairs byte-identical, 1 a difference (reported), 2 a setup failure.
#include "strata/helper/strata_helper_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(STRATA_PARITY_CUDART)
#include <cuda_runtime.h>
#endif
#if defined(STRATA_PARITY_GGML)
#include "ggml.h"
#endif

namespace {
constexpr int64_t H = 2560, FF = 640, K = 10, MAXT = 8;
using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }

[[noreturn]] void die(const std::string& m) { std::fprintf(stderr, "hip_worker_parity: %s\n", m.c_str()); std::exit(2); }

// bytes of one row of `n` weights of a ggml type
int64_t row_bytes(int type, int64_t n) {
    switch (type) {
        case 2: return n / 32 * 18;    // Q4_0
        case 6: return n / 32 * 22;    // Q5_0
        case 7: return n / 32 * 24;    // Q5_1
        case 8: return n / 32 * 34;    // Q8_0
        case 12: return n / 256 * 144; // Q4_K
        case 13: return n / 256 * 176; // Q5_K
        case 14: return n / 256 * 210; // Q6_K
        default: die("unknown ggml type " + std::to_string(type));
    }
}

struct Worker {
    std::string name;
    void* handle = nullptr;
    const strata_helper_api* api = nullptr;
    strata_helper_device* dev = nullptr;
    int device = 0;
};

void load(Worker& w, const std::string& path) {
    w.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!w.handle) die("dlopen " + path + ": " + dlerror());
    auto entry = (strata_helper_entry_fn) dlsym(w.handle, STRATA_HELPER_ENTRY_SYMBOL);
    if (!entry) die(path + ": no " STRATA_HELPER_ENTRY_SYMBOL);
    w.api = entry();
    if (!w.api || w.api->abi_version != STRATA_HELPER_ABI_VERSION || w.api->struct_size < sizeof(strata_helper_api))
        die(path + ": ABI mismatch");
    const int n = w.api->device_count();
    std::fprintf(stderr, "[%s] backend=%s devices=%d", w.name.c_str(), w.api->backend, n);
    for (int i = 0; i < n; ++i) {
        char nm[256] = {};
        w.api->device_name(i, nm, sizeof nm);
        std::fprintf(stderr, " [%d]=\"%s\"", i, nm);
    }
    std::fprintf(stderr, "\n");
    if (w.device >= n) die(w.name + ": device " + std::to_string(w.device) + " not visible");
    char err[512] = {};
    double free_gib = 0;
    if (w.api->preflight(w.device, &free_gib, err, sizeof err)) die(w.name + " preflight: " + err);
    std::fprintf(stderr, "[%s] preflight ok, %.2f GiB free\n", w.name.c_str(), free_gib);
}

struct Layer {
    int layer = 0, gu = 0, d = 0;
    int64_t blob = 0;
    std::vector<uint8_t> blobs;   // n_expert * blob
};

std::vector<uint8_t> read_file(const std::string& path, int64_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open " + path);
    std::vector<uint8_t> v((size_t) bytes);
    in.read((char*) v.data(), bytes);
    if (in.gcount() != bytes) die(path + ": short (" + std::to_string(in.gcount()) + " of " + std::to_string(bytes) + ")");
    return v;
}

Layer load_native(const std::string& dir, int layer, int gu, int d, int n_expert) {
    Layer L;
    L.layer = layer; L.gu = gu; L.d = d;
    const int64_t gub = FF * row_bytes(gu, H), db = H * row_bytes(d, FF);
    L.blob = 2 * gub + db;
    const std::string p = dir + "/L" + std::to_string(layer);
    auto g = read_file(p + ".gate", gub * n_expert), u = read_file(p + ".up", gub * n_expert),
         dn = read_file(p + ".down", db * n_expert);
    L.blobs.resize((size_t) (L.blob * n_expert));
    for (int e = 0; e < n_expert; ++e) {
        uint8_t* b = L.blobs.data() + (size_t) e * L.blob;
        std::memcpy(b, g.data() + (size_t) e * gub, gub);
        std::memcpy(b + gub, u.data() + (size_t) e * gub, gub);
        std::memcpy(b + 2 * gub, dn.data() + (size_t) e * db, db);
    }
    return L;
}

// Synthetic S2 (Q2_0) blobs: random 2-bit codes, fp16 scales of a trained pack's magnitude.
uint16_t to_half(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    return (uint16_t) (sign | ((uint32_t) e << 10) | (m >> 13));
}

Layer make_s2(int n_expert, uint64_t seed) {
    constexpr int64_t ROW_GU = H * 2 / 8, ROW_D = FF * 2 / 8, SC_GU = H / 64, SC_D = FF / 64;
    constexpr int64_t O_D_CODES = 2 * FF * ROW_GU, O_GU_SCALES = O_D_CODES + H * ROW_D,
                      O_D_SCALES = O_GU_SCALES + 2 * FF * SC_GU * 2, BLOB = O_D_SCALES + H * SC_D * 2;
    static_assert(BLOB == 1382400, "the S2 blob");
    Layer L;
    L.layer = -1; L.blob = BLOB;
    L.blobs.resize((size_t) (BLOB * n_expert));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> sc(0.004f, 0.03f);
    for (int e = 0; e < n_expert; ++e) {
        uint8_t* b = L.blobs.data() + (size_t) e * BLOB;
        for (int64_t i = 0; i < O_GU_SCALES; i += 8) { const uint64_t r = rng(); std::memcpy(b + i, &r, 8); }
        for (int64_t i = O_GU_SCALES; i < BLOB; i += 2) {
            const uint16_t h = to_half((rng() & 1 ? 1.f : -1.f) * sc(rng));
            std::memcpy(b + i, &h, 2);
        }
    }
    return L;
}

// A window's routing: K distinct experts per token, skewed (a few hot experts), as the engine groups it.
struct Launch {
    std::vector<int32_t> slot, start, tok;
    int32_t n_groups() const { return (int32_t) slot.size(); }
    int32_t n_entries() const { return (int32_t) tok.size(); }
};

void route(int64_t n_tok, int n_expert, std::mt19937_64& rng, std::vector<int32_t>& ids) {
    std::vector<double> w((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) w[(size_t) e] = 1.0 / std::pow(e + 4.0, 0.9);
    std::shuffle(w.begin(), w.end(), std::mt19937_64(12345));   // the hot set is fixed per run, not expert 0..k
    ids.resize((size_t) (n_tok * K));
    for (int64_t t = 0; t < n_tok; ++t) {
        std::vector<double> ww = w;
        for (int64_t j = 0; j < K; ++j) {
            std::discrete_distribution<int> pick(ww.begin(), ww.end());
            const int e = pick(rng);
            ids[(size_t) (t * K + j)] = e;
            ww[(size_t) e] = 0.0;
        }
    }
}

// foreign_experts.cpp's grouping: groups in order of first appearance, entries in routed order within a group;
// `split` cuts every group into runs of at most MAXT entries.  `out_row[i]` = the routed index of packed row i.
Launch group(const std::vector<int32_t>& ids, int slot_base, bool split, std::vector<int32_t>* out_row) {
    std::vector<int32_t> order;   // distinct experts by first appearance
    std::map<int32_t, std::vector<int32_t>> rows;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!rows.count(ids[i])) order.push_back(ids[i]);
        rows[ids[i]].push_back((int32_t) i);
    }
    Launch L;
    if (out_row) out_row->clear();
    for (int32_t e : order) {
        const auto& r = rows[e];
        for (size_t at = 0; at < r.size(); at += split ? MAXT : r.size()) {
            L.slot.push_back(slot_base + e);
            L.start.push_back((int32_t) L.tok.size());
            const size_t end = split ? std::min(r.size(), at + MAXT) : r.size();
            for (size_t i = at; i < end; ++i) {
                L.tok.push_back(r[i] / (int32_t) K);
                if (out_row) out_row->push_back(r[i]);
            }
        }
    }
    L.start.push_back((int32_t) L.tok.size());
    return L;
}

// hidden-state-like activations: unit RMS, a few outlier channels, per-token scale
void activations(int64_t n_tok, std::mt19937_64& rng, std::vector<float>& x) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ts(0.5f, 2.0f);
    x.resize((size_t) (n_tok * H));
    for (int64_t t = 0; t < n_tok; ++t) {
        const float s = ts(rng);
        for (int64_t c = 0; c < H; ++c) {
            float v = nd(rng) * s;
            if (c % 397 == 11) v *= 12.f;
            x[(size_t) (t * H + c)] = v;
        }
    }
}

struct Timing { double begin_us = 0, finish_us = 0; };

std::vector<float> run(Worker& w, const Layer& L, const Launch& la, int64_t n_tok, const std::vector<float>& x,
                       Timing* t = nullptr) {
    strata_helper_launch l{};
    l.layer = L.layer; l.n_tok = n_tok; l.x = x.data();
    l.n_groups = la.n_groups(); l.slot = la.slot.data(); l.start = la.start.data(); l.tok = la.tok.data();
    l.n_entries = la.n_entries(); l.gu_type = L.gu; l.d_type = L.d;
    char err[512] = {};
    const auto t0 = Clock::now();
    if (w.api->begin(w.dev, &l, err, sizeof err)) die(w.name + " begin: " + err);
    const auto t1 = Clock::now();
    const float* rows = nullptr;
    if (w.api->finish(w.dev, &rows, err, sizeof err)) die(w.name + " finish: " + err);
    if (t) { t->begin_us = std::chrono::duration<double, std::micro>(t1 - t0).count(); t->finish_us = us_since(t1); }
    return std::vector<float>(rows, rows + (size_t) la.n_entries() * H);
}

struct Diff {
    int64_t rows = 0, rows_same = 0, floats_diff = 0, nonfinite = 0;
    double max_abs = 0, max_rel = 0;   // rel: |a-b| / max|ref| of the row
    int64_t max_ulp = 0;
    bool identical() const { return rows == rows_same && nonfinite == 0; }
    void add(const Diff& o) {
        rows += o.rows; rows_same += o.rows_same; floats_diff += o.floats_diff; nonfinite += o.nonfinite;
        max_abs = std::max(max_abs, o.max_abs); max_rel = std::max(max_rel, o.max_rel); max_ulp = std::max(max_ulp, o.max_ulp);
    }
};

Diff compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Diff d;
    d.rows = (int64_t) (ref.size() / H);
    for (int64_t r = 0; r < d.rows; ++r) {
        const float* a = ref.data() + r * H;
        const float* b = got.data() + r * H;
        for (int64_t c = 0; c < H; ++c) if (!std::isfinite(a[c]) || !std::isfinite(b[c])) ++d.nonfinite;
        if (std::memcmp(a, b, H * sizeof(float)) == 0) { ++d.rows_same; continue; }
        float mx = 0;
        for (int64_t c = 0; c < H; ++c) mx = std::max(mx, std::fabs(a[c]));
        for (int64_t c = 0; c < H; ++c) {
            if (std::memcmp(&a[c], &b[c], 4) == 0) continue;
            ++d.floats_diff;
            const double ad = std::fabs((double) a[c] - (double) b[c]);
            d.max_abs = std::max(d.max_abs, ad);
            if (mx > 0) d.max_rel = std::max(d.max_rel, ad / mx);
            int32_t ia, ib; std::memcpy(&ia, &a[c], 4); std::memcpy(&ib, &b[c], 4);
            if (ia < 0) ia = INT32_MIN - ia;
            if (ib < 0) ib = INT32_MIN - ib;
            d.max_ulp = std::max<int64_t>(d.max_ulp, std::llabs((long long) ia - (long long) ib));
        }
    }
    return d;
}

// The FP32 oracle (--oracle, needs ggml): the expert in double from ggml's own dequantized weights and the raw
// activation, so each worker's distance from the exact product can be told apart from its distance to the others.
struct OracleErr { int64_t rows = 0; double sum_rel = 0, max_rel = 0; };
#if defined(STRATA_PARITY_GGML)
std::vector<float> dequant(int type, const uint8_t* src, int64_t rows, int64_t cols) {
    std::vector<float> out((size_t) (rows * cols));
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    const int64_t rb = row_bytes(type, cols);
    for (int64_t r = 0; r < rows; ++r) tr->to_float(src + r * rb, out.data() + r * cols, cols);
    return out;
}
std::vector<float> oracle(const Layer& L, const std::vector<int32_t>& ids_of_row, const std::vector<int32_t>& tok_of_row,
                          const std::vector<float>& x, int slot_base) {
    std::vector<float> out(ids_of_row.size() * (size_t) H);
    std::map<int32_t, std::vector<size_t>> by_expert;
    for (size_t i = 0; i < ids_of_row.size(); ++i) by_expert[ids_of_row[i] - slot_base].push_back(i);
    const int64_t gub = FF * row_bytes(L.gu, H);
    for (const auto& [e, rows] : by_expert) {
        const uint8_t* b = L.blobs.data() + (size_t) e * L.blob;
        const auto g = dequant(L.gu, b, FF, H), u = dequant(L.gu, b + gub, FF, H), d = dequant(L.d, b + 2 * gub, H, FF);
        std::vector<double> h((size_t) FF);
        for (size_t i : rows) {
            const float* xr = x.data() + (size_t) tok_of_row[i] * H;
            for (int64_t f = 0; f < FF; ++f) {
                double gs = 0, us = 0;
                for (int64_t c = 0; c < H; ++c) { gs += (double) g[(size_t) (f * H + c)] * xr[c]; us += (double) u[(size_t) (f * H + c)] * xr[c]; }
                h[(size_t) f] = gs / (1.0 + std::exp(-gs)) * us;
            }
            for (int64_t r = 0; r < H; ++r) {
                double s = 0;
                for (int64_t f = 0; f < FF; ++f) s += (double) d[(size_t) (r * FF + f)] * h[(size_t) f];
                out[i * (size_t) H + (size_t) r] = (float) s;
            }
        }
    }
    return out;
}
#endif
void oracle_add(OracleErr& o, const std::vector<float>& exact, const std::vector<float>& got) {
    for (size_t r = 0; r < exact.size() / H; ++r) {
        double num = 0, den = 0;
        for (int64_t c = 0; c < H; ++c) {
            const double e = exact[r * H + (size_t) c], g = got[r * H + (size_t) c];
            num += (g - e) * (g - e); den += e * e;
        }
        const double rel = den > 0 ? std::sqrt(num / den) : 0;
        ++o.rows; o.sum_rel += rel; o.max_rel = std::max(o.max_rel, rel);
    }
}

void report(const char* what, const Diff& d) {
    std::printf("  %-28s rows %6lld identical %6lld  %s", what, (long long) d.rows, (long long) d.rows_same,
                d.identical() ? "BYTE-IDENTICAL" : "DIFFERS");
    if (!d.identical())
        std::printf("  (floats %lld, max_abs %.3g, max_rel %.3g, max_ulp %lld, nonfinite %lld)", (long long) d.floats_diff,
                    d.max_abs, d.max_rel, (long long) d.max_ulp, (long long) d.nonfinite);
    std::printf("\n");
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p * (double) v.size()))];
}

void open_worker(Worker& w, const std::vector<const Layer*>& layers, int n_expert, int64_t max_tok, bool native) {
    std::vector<int64_t> sizes;
    for (const Layer* L : layers) for (int e = 0; e < n_expert; ++e) sizes.push_back(L->blob);
    strata_helper_open_params p{};
    p.n_slots = (int64_t) sizes.size(); p.slot_bytes = sizes.data();
    p.max_entries = max_tok * K; p.max_tokens = max_tok; p.n_embd = H; p.n_ff = FF; p.native = native ? 1 : 0;
    char err[512] = {};
    if (w.api->open(w.device, &p, &w.dev, err, sizeof err)) die(w.name + " open: " + err);
    const auto t0 = Clock::now();
    int64_t bytes = 0;
    for (size_t li = 0; li < layers.size(); ++li)
        for (int e = 0; e < n_expert; ++e) {
            const int32_t slot = (int32_t) (li * (size_t) n_expert + (size_t) e);
            const uint8_t* b = layers[li]->blobs.data() + (size_t) e * layers[li]->blob;
            if (w.api->fill_slot(w.dev, slot, b, layers[li]->blob, err, sizeof err)) die(w.name + " fill: " + err);
            bytes += layers[li]->blob;
        }
    const double fill_us = us_since(t0);
    for (size_t li = 0; li < layers.size(); ++li) {
        const int32_t slot = (int32_t) (li * (size_t) n_expert + (size_t) n_expert - 1);
        if (w.api->verify_slot(w.dev, slot, layers[li]->blobs.data() + (size_t) (n_expert - 1) * layers[li]->blob,
                               layers[li]->blob, err, sizeof err))
            die(w.name + " verify: " + err);
    }
    std::fprintf(stderr, "[%s] %s: %zu slots, %.1f MiB filled in %.0f ms (%.2f GB/s pageable H2D), %.3f GiB on device\n",
                 w.name.c_str(), native ? "native" : "s2", sizes.size(), bytes / 1048576.0, fill_us / 1e3,
                 bytes / fill_us / 1e3, w.api->gib(w.dev));
}

#if defined(STRATA_PARITY_CUDART)
void cuda_ok(cudaError_t st, const char* what) {
    if (st != cudaSuccess) die(std::string("cuda ") + what + ": " + cudaGetErrorString(st));
}
void cuda_init(int device) {
    int n = 0;
    cuda_ok(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    std::fprintf(stderr, "[driver] CUDA runtime: %d device(s)", n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        cuda_ok(cudaGetDeviceProperties(&p, i), "props");
        std::fprintf(stderr, " [%d]=\"%s\" sm_%d%d", i, p.name, p.major, p.minor);
    }
    std::fprintf(stderr, "\n");
    cuda_ok(cudaSetDevice(device), "cudaSetDevice");
    cuda_ok(cudaFree(nullptr), "context");
}
#endif

struct Args {
    std::string cpu, hip, cuda, data;
    std::vector<std::string> layers;
    std::vector<int64_t> windows{1, 5, 8, 64, 512};
    int n_expert = 128, reps = 3, timing_reps = 20;
    bool serialize = false, no_copy_loop = false;
    bool s2 = false, s2_cpu = false, concurrent = false, hip_first = false, oracle = false;
    int cuda_device = 0, hip_device = 0;
    uint64_t seed = 20261030;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) die("missing value for " + k); return argv[++i]; };
        if (k == "--cpu") a.cpu = v();
        else if (k == "--hip") a.hip = v();
        else if (k == "--cuda") a.cuda = v();
        else if (k == "--data") a.data = v();
        else if (k == "--layer") a.layers.push_back(v());
        else if (k == "--experts") a.n_expert = std::atoi(v().c_str());
        else if (k == "--reps") a.reps = std::atoi(v().c_str());
        else if (k == "--timing-reps") a.timing_reps = std::atoi(v().c_str());
        else if (k == "--windows") {
            a.windows.clear();
            const std::string s = v();
            for (size_t at = 0; at < s.size();) {
                const size_t c = s.find(',', at);
                a.windows.push_back(std::atoll(s.substr(at, c - at).c_str()));
                if (c == std::string::npos) break;
                at = c + 1;
            }
        }
        else if (k == "--s2") a.s2 = true;
        else if (k == "--s2-cpu") a.s2_cpu = true;
        else if (k == "--serialize") a.serialize = true;        // concurrent phase: finish HIP before CUDA begins
        else if (k == "--no-copy-loop") a.no_copy_loop = true;  // concurrent phase: no CUDA copy loop
        else if (k == "--oracle") a.oracle = true;
        else if (k == "--concurrent") a.concurrent = true;
        else if (k == "--hip-first") a.hip_first = true;
        else if (k == "--cuda-device") a.cuda_device = std::atoi(v().c_str());
        else if (k == "--hip-device") a.hip_device = std::atoi(v().c_str());
        else if (k == "--seed") a.seed = std::strtoull(v().c_str(), nullptr, 10);
        else die("unknown argument " + k);
    }
    if (a.cpu.empty() && a.hip.empty() && a.cuda.empty()) die("name at least one worker (--cpu, --hip, --cuda)");
    return a;
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // a crash keeps every line printed so far
    const Args a = parse(argc, argv);
    const int64_t max_tok = *std::max_element(a.windows.begin(), a.windows.end());

    // ---- the workers, in the order that exercises the runtime init question asked
    Worker cpu{"cpu"}, hip{"hip"}, cuda{"cuda"};
    hip.device = a.hip_device; cuda.device = a.cuda_device;
    if (a.hip_first && !a.hip.empty()) load(hip, a.hip);
#if defined(STRATA_PARITY_CUDART)
    cuda_init(a.cuda_device);
#endif
    if (!a.hip_first && !a.hip.empty()) load(hip, a.hip);
    if (!a.cuda.empty()) load(cuda, a.cuda);
    if (!a.cpu.empty()) load(cpu, a.cpu);
    std::vector<Worker*> all_workers;
    for (Worker* w : {&cpu, &hip, &cuda}) if (w->api) all_workers.push_back(w);

    // ---- the blobs
    std::vector<Layer> native;
    for (const std::string& spec : a.layers) {
        int l = 0, gu = 0, d = 0;
        if (std::sscanf(spec.c_str(), "%d:%d:%d", &l, &gu, &d) != 3) die("--layer L:GU:D, not " + spec);
        if (a.data.empty()) die("--layer needs --data");
        native.push_back(load_native(a.data, l, gu, d, a.n_expert));
        std::fprintf(stderr, "[driver] layer %d: gate/up type %d, down type %d, blob %lld B, %d experts\n", l, gu, d,
                     (long long) native.back().blob, a.n_expert);
    }
    Layer s2;
    if (a.s2) s2 = make_s2(a.n_expert, a.seed ^ 0x52);

    int failures = 0;
    std::map<std::string, Diff> totals;
    std::map<std::string, std::vector<double>> tb, tf;   // name/window -> begin / finish wall times (us)
    std::map<std::string, std::vector<double>> sync_us;
    std::map<std::string, OracleErr> oracle_err;

    auto suite = [&](const std::vector<const Layer*>& layers, bool is_native) {
        // the CPU reference's S2 kernel needs AVX-512 VNNI (SIGILL without it): S2 compares the GPU workers unless --s2-cpu
        const bool use_cpu = cpu.api && (is_native || a.s2_cpu);
        std::vector<Worker*> workers;
        for (Worker* w : all_workers) if (w != &cpu || use_cpu) workers.push_back(w);
        Worker* ref = use_cpu ? &cpu : (cuda.api ? &cuda : &hip);
        for (Worker* w : workers) open_worker(*w, layers, a.n_expert, max_tok, is_native);
        std::mt19937_64 rng(a.seed + (is_native ? 0 : 7));
        for (size_t li = 0; li < layers.size(); ++li) {
            const Layer& L = *layers[li];
            const int slot_base = (int) (li * (size_t) a.n_expert);
            for (int64_t nt : a.windows) {
                for (int rep = 0; rep < a.reps; ++rep) {
                    std::vector<float> x; std::vector<int32_t> ids, order_split, order_whole;
                    activations(nt, rng, x);
                    route(nt, a.n_expert, rng, ids);
                    const Launch split = group(ids, slot_base, true, &order_split);
                    const Launch whole = group(ids, slot_base, false, &order_whole);
                    if (order_split != order_whole) die("split changed the row order");
                    const std::vector<float> r = run(*ref, L, split, nt, x);
                    std::printf("%s layer %d (%d/%d) window %lld rep %d: %d entries, %d groups (%d split)\n",
                                is_native ? "native" : "s2", L.layer, L.gu, L.d, (long long) nt, rep, whole.n_entries(),
                                whole.n_groups(), split.n_groups());
                    std::map<std::string, std::vector<float>> whole_rows;
                    for (Worker* w : workers) {
                        if (w == ref) continue;
                        const std::string key = std::string(is_native ? "native " : "s2 ") + w->name + " vs " + ref->name;
                        // whole groups FIRST: a kernel that skips entries leaves the previous launch's rows in its
                        // output buffer, and after the split launch those would be the right rows
                        std::vector<float> got_whole;
                        if (w != &cpu && whole.n_groups() != split.n_groups()) got_whole = run(*w, L, whole, nt, x);
                        const std::vector<float> got = run(*w, L, split, nt, x);
                        whole_rows[w->name] = got;
                        Diff d = compare(r, got);
                        report((w->name + " vs " + ref->name + " (split)").c_str(), d);
                        totals[key].add(d);
                        if (!d.identical()) ++failures;
                        if (!got_whole.empty()) {
                            Diff dw = compare(r, got_whole);
                            report((w->name + " vs " + ref->name + " (whole)").c_str(), dw);
                            totals[key + " (whole groups)"].add(dw);
                            if (!dw.identical()) ++failures;
                            Diff ds = compare(got, got_whole);   // a worker against itself: group size must not matter
                            totals[std::string(is_native ? "native " : "s2 ") + w->name + " whole vs split"].add(ds);
                            if (!ds.identical()) { report((w->name + " whole vs split").c_str(), ds); ++failures; }
                        }
                    }
#if defined(STRATA_PARITY_GGML)
                    if (a.oracle && is_native && nt <= 64) {   // the exact product, per routed entry in packed order
                        std::vector<int32_t> slot_of_row, tok_of_row;
                        for (int32_t g = 0; g < split.n_groups(); ++g)
                            for (int32_t i = split.start[g]; i < split.start[g + 1]; ++i) {
                                slot_of_row.push_back(split.slot[g]); tok_of_row.push_back(split.tok[i]);
                            }
                        const auto exact = oracle(L, slot_of_row, tok_of_row, x, slot_base);
                        oracle_add(oracle_err[ref->name], exact, r);
                        for (const auto& [name, rows] : whole_rows) oracle_add(oracle_err[name], exact, rows);
                    }
#endif
                    if (hip.api && cuda.api && ref != &hip && ref != &cuda) {   // the real question: HIP vs the CUDA kernels
                        Diff d = compare(whole_rows["cuda"], whole_rows["hip"]);
                        report("hip vs cuda (split)", d);
                        totals[std::string(is_native ? "native " : "s2 ") + "hip vs cuda"].add(d);
                        if (!d.identical()) ++failures;
                    }
                }
                // timing: the GPU workers, the engine's grouping (whole groups)
                for (Worker* w : workers) {
                    if (w == &cpu) continue;
                    std::vector<float> x; std::vector<int32_t> ids;
                    activations(nt, rng, x);
                    route(nt, a.n_expert, rng, ids);
                    const Launch whole = group(ids, slot_base, false, nullptr);
                    const std::string key = std::string(is_native ? "native " : "s2 ") + w->name + " L" +
                                            std::to_string(L.layer) + " n_tok " + std::to_string(nt);
                    run(*w, L, whole, nt, x);   // warm
                    for (int i = 0; i < a.timing_reps; ++i) {
                        Timing t;
                        run(*w, L, whole, nt, x, &t);
                        tb[key].push_back(t.begin_us); tf[key].push_back(t.finish_us);
                    }
                    // finish() on a stream that has long finished: the synchronize call's own latency
                    for (int i = 0; i < 5; ++i) {
                        strata_helper_launch l{};
                        l.layer = L.layer; l.n_tok = nt; l.x = x.data(); l.n_groups = whole.n_groups();
                        l.slot = whole.slot.data(); l.start = whole.start.data(); l.tok = whole.tok.data();
                        l.n_entries = whole.n_entries(); l.gu_type = L.gu; l.d_type = L.d;
                        char err[512] = {};
                        if (w->api->begin(w->dev, &l, err, sizeof err)) die(w->name + " begin: " + err);
                        std::this_thread::sleep_for(std::chrono::milliseconds(30));
                        const float* rows = nullptr;
                        const auto t0 = Clock::now();
                        if (w->api->finish(w->dev, &rows, err, sizeof err)) die(w->name + " finish: " + err);
                        sync_us[w->name].push_back(us_since(t0));
                    }
                }
            }
        }

        // ---- two runtimes at once: the HIP worker and the CUDA side computing concurrently, results unchanged
        if (a.concurrent && hip.api) {
            std::atomic<bool> stop{false};
            std::atomic<long> cuda_iters{0};
            std::thread cuda_load([&] {
#if defined(STRATA_PARITY_CUDART)
                cuda_ok(cudaSetDevice(a.cuda_device), "cudaSetDevice (thread)");
                void* d = nullptr; void* hbuf = nullptr;
                const size_t n = 64u << 20;
                cuda_ok(cudaMalloc(&d, n), "cudaMalloc");
                cuda_ok(cudaMallocHost(&hbuf, n), "cudaMallocHost");
                cudaStream_t s; cuda_ok(cudaStreamCreate(&s), "stream");
                while (!stop.load() && !a.no_copy_loop) {
                    cuda_ok(cudaMemcpyAsync(d, hbuf, n, cudaMemcpyHostToDevice, s), "H2D");
                    cuda_ok(cudaMemsetAsync(d, cuda_iters.load() & 0xff, n, s), "memset");
                    cuda_ok(cudaMemcpyAsync(hbuf, d, n, cudaMemcpyDeviceToHost, s), "D2H");
                    cuda_ok(cudaStreamSynchronize(s), "sync");
                    ++cuda_iters;
                }
                cudaStreamDestroy(s); cudaFree(d); cudaFreeHost(hbuf);
#endif
            });
            // the HIP worker and the CUDA worker launched back to back from one thread, so both cards compute at
            // once (with the copy loop above on the CUDA card too); HIP's rows against the CUDA worker's when it is
            // loaded (the target), else against the HIP worker's own serial result for the same launch
            int bad = 0, n = 0;
            struct Failed { size_t li; int64_t nt; std::vector<float> x; Launch launch; std::vector<float> hip, cuda; };
            std::vector<Failed> failed;
            std::mt19937_64 r3(a.seed + 5);
            const auto t0 = Clock::now();
            while (us_since(t0) < 20e6) {
                for (size_t li = 0; li < layers.size(); ++li) {
                    const Layer& L = *layers[li];
                    const int64_t nt = (n % 3 == 0) ? 1 : (n % 3 == 1) ? 5 : std::min<int64_t>(max_tok, 64);
                    std::vector<float> x; std::vector<int32_t> ids;
                    activations(nt, r3, x); route(nt, a.n_expert, r3, ids);
                    const Launch whole = group(ids, (int) (li * (size_t) a.n_expert), false, nullptr);
                    strata_helper_launch l{};
                    l.layer = L.layer; l.n_tok = nt; l.x = x.data(); l.n_groups = whole.n_groups();
                    l.slot = whole.slot.data(); l.start = whole.start.data(); l.tok = whole.tok.data();
                    l.n_entries = whole.n_entries(); l.gu_type = L.gu; l.d_type = L.d;
                    char err[512] = {};
                    const float *rh = nullptr, *rc = nullptr;
                    if (hip.api->begin(hip.dev, &l, err, sizeof err)) die(std::string("hip begin: ") + err);
                    if (a.serialize && hip.api->finish(hip.dev, &rh, err, sizeof err)) die(std::string("hip finish: ") + err);
                    if (cuda.api && cuda.api->begin(cuda.dev, &l, err, sizeof err)) die(std::string("cuda begin: ") + err);
                    if (!a.serialize && hip.api->finish(hip.dev, &rh, err, sizeof err)) die(std::string("hip finish: ") + err);
                    if (cuda.api && cuda.api->finish(cuda.dev, &rc, err, sizeof err)) die(std::string("cuda finish: ") + err);
                    const std::vector<float> got(rh, rh + (size_t) l.n_entries * H);
                    const std::vector<float> want = rc ? std::vector<float>(rc, rc + (size_t) l.n_entries * H)
                                                       : run(hip, L, whole, nt, x);
                    const Diff d = compare(want, got);
                    if (!d.identical()) {
                        ++bad;
                        failed.push_back({li, nt, x, whole, got, want});
                        // which side moved: each worker again, serially, on the same launch
                        const Diff dh = compare(got, run(hip, L, whole, nt, x));
                        const Diff dc = cuda.api ? compare(want, run(cuda, L, whole, nt, x)) : Diff{};
                        std::printf("  concurrent launch %d (n_tok %lld, %d entries) differs; serial rerun: hip %s, cuda %s\n", n,
                                    (long long) nt, l.n_entries, dh.identical() ? "same" : "CHANGED",
                                    dc.identical() ? "same" : "CHANGED");
                        report("hip vs cuda (concurrent)", d);
                    }
                    ++n;
                }
            }
            stop = true;
            cuda_load.join();
            // the launches that differed, again with both cards quiet and after an unrelated launch: whichever
            // worker now returns other rows was the one that went wrong under load
            for (const Failed& f : failed) {
                const Layer& L = *layers[f.li];
                std::vector<float> x2; std::vector<int32_t> ids2;
                activations(3, r3, x2); route(3, a.n_expert, r3, ids2);
                const Launch other = group(ids2, (int) (f.li * (size_t) a.n_expert), false, nullptr);
                run(hip, L, other, 3, x2);
                if (cuda.api) run(cuda, L, other, 3, x2);
                const std::vector<float> qh = run(hip, L, f.launch, f.nt, f.x);
                const std::vector<float> qc = cuda.api ? run(cuda, L, f.launch, f.nt, f.x) : qh;
                const Diff dh = compare(f.hip, qh), dc = compare(f.cuda, qc), dq = compare(qc, qh);
                // the rows that moved, by the size of their group
                std::map<int, int> moved_by_size, rows_by_size;
                for (int32_t g = 0; g < f.launch.n_groups(); ++g) {
                    const int sz = f.launch.start[g + 1] - f.launch.start[g];
                    for (int32_t i = f.launch.start[g]; i < f.launch.start[g + 1]; ++i) {
                        ++rows_by_size[sz];
                        if (std::memcmp(&f.hip[(size_t) i * H], &qh[(size_t) i * H], H * 4) != 0 ||
                            std::memcmp(&f.cuda[(size_t) i * H], &qc[(size_t) i * H], H * 4) != 0) ++moved_by_size[sz];
                    }
                }
                std::printf("  quiet rerun of a differing launch (n_tok %lld): hip %s, cuda %s, quiet hip vs cuda %s; moved rows by group size:",
                            (long long) f.nt, dh.identical() ? "unchanged" : "CHANGED", dc.identical() ? "unchanged" : "CHANGED",
                            dq.identical() ? "identical" : "DIFFER");
                for (const auto& [sz, m] : moved_by_size) std::printf(" %d:%d/%d", sz, m, rows_by_size[sz]);
                std::printf("\n");
            }
            std::printf("concurrent (%s): %d HIP launches overlapped with %s, %d differ from %s; CUDA copy loop %ld "
                        "iterations\n", is_native ? "native" : "s2", n, cuda.api ? "CUDA worker launches" : "nothing",
                        bad, cuda.api ? "the CUDA worker" : "a serial HIP rerun", cuda_iters.load());
            std::fflush(stdout);
            failures += bad;
        }
        for (Worker* w : workers) { w->api->close(w->dev); w->dev = nullptr; }
    };

    if (!native.empty()) {
        std::vector<const Layer*> ls;
        for (const Layer& L : native) ls.push_back(&L);
        suite(ls, true);
    }
    if (a.s2) suite({&s2}, false);

    std::printf("\n== parity totals\n");
    for (const auto& [k, d] : totals) report(k.c_str(), d);
    if (!oracle_err.empty()) {
        std::printf("\n== distance from the FP32 oracle (relative L2 per output row; windows <= 64)\n");
        for (const auto& [k, o] : oracle_err)
            std::printf("  %-8s rows %6lld  mean %.4g  max %.4g\n", k.c_str(), (long long) o.rows, o.sum_rel / (double) o.rows, o.max_rel);
    }
    std::printf("\n== timing (wall, us): begin() = host staging + enqueue; finish() = wait + return; median / p90\n");
    for (const auto& [k, v] : tb) {
        const auto& f = tf[k];
        std::vector<double> tot(v.size());
        for (size_t i = 0; i < v.size(); ++i) tot[i] = v[i] + f[i];
        std::printf("  %-34s begin %8.1f / %8.1f   finish %8.1f / %8.1f   total %8.1f / %8.1f\n", k.c_str(), median(v),
                    pct(v, 0.9), median(f), pct(f, 0.9), median(tot), pct(tot, 0.9));
    }
    for (const auto& [k, v] : sync_us)
        std::printf("  %-34s finish() on a finished stream: median %.1f us, max %.1f us\n", (k + " sync").c_str(), median(v),
                    *std::max_element(v.begin(), v.end()));
    std::fflush(stdout);
    std::printf("\nhip_worker_parity: %s\n", failures ? "DIFFERENCES FOUND" : "all compared rows byte-identical");
    return failures ? 1 : 0;
}
