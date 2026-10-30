// src/core/foreign_experts.cpp - Frankenstein: the host side of a helper tier behind the worker C ABI.
#include "strata/core/foreign_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strata::core {
namespace {
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

std::string trimmed(const char* buf) {
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}
}  // namespace

// ---------------------------------------------------------------- HelperWorkerLibrary
HelperWorkerLibrary::~HelperWorkerLibrary() {
#if defined(_WIN32)
    if (handle_) FreeLibrary((HMODULE) handle_);
#else
    if (handle_) dlclose(handle_);
#endif
}

std::unique_ptr<HelperWorkerLibrary> HelperWorkerLibrary::load(const std::string& path, std::string& err) {
    std::unique_ptr<HelperWorkerLibrary> lib(new HelperWorkerLibrary());
    lib->path_ = path;
#if defined(_WIN32)
    lib->handle_ = (void*) LoadLibraryA(path.c_str());
    if (!lib->handle_) { err = "helper worker: cannot load " + path; return nullptr; }
    auto entry = (strata_helper_entry_fn) GetProcAddress((HMODULE) lib->handle_, STRATA_HELPER_ENTRY_SYMBOL);
#else
    // RTLD_LOCAL: the worker's symbols stay its own (its kernels are a second copy of the engine's, compiled for
    // another device); RTLD_NOW: an unresolved symbol fails here, by name, not inside a layer.
    lib->handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib->handle_) {
        const char* why = dlerror();
        err = "helper worker: cannot load " + path + ": " + (why ? why : "dlopen failed");
        return nullptr;
    }
    auto entry = (strata_helper_entry_fn) dlsym(lib->handle_, STRATA_HELPER_ENTRY_SYMBOL);
#endif
    if (!entry) { err = "helper worker: " + path + " exports no " STRATA_HELPER_ENTRY_SYMBOL; return nullptr; }
    const strata_helper_api* api = entry();
    if (!api) { err = "helper worker: " + path + " refused this ABI (its entry returned null)"; return nullptr; }
    if (api->abi_version != STRATA_HELPER_ABI_VERSION) {
        err = "helper worker: " + path + " is ABI version " + std::to_string(api->abi_version) + ", the engine needs " +
              std::to_string(STRATA_HELPER_ABI_VERSION);
        return nullptr;
    }
    if (api->struct_size < sizeof(strata_helper_api)) {
        err = "helper worker: " + path + " has a shorter API table than this engine (" +
              std::to_string(api->struct_size) + " < " + std::to_string(sizeof(strata_helper_api)) + " bytes)";
        return nullptr;
    }
    const void* fns[] = {(const void*) api->device_count, (const void*) api->preflight, (const void*) api->open,
                         (const void*) api->close, (const void*) api->fill_slot, (const void*) api->verify_slot,
                         (const void*) api->read_slot, (const void*) api->begin, (const void*) api->finish,
                         (const void*) api->gib};
    for (const void* f : fns) if (!f) { err = "helper worker: " + path + " leaves a required function null"; return nullptr; }
    lib->api_ = api;
    return lib;
}

// ---------------------------------------------------------------- HelperSlotTable
void HelperSlotTable::open(int64_t n_layers, int64_t n_expert, std::vector<int64_t> slot_bytes) {
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    bytes_ = std::move(slot_bytes);
    residency_.assign((size_t) (n_layers * n_expert), -1);
    present_.assign((size_t) n_layers, 0);
    next_free_ = 0;
    total_ = 0;
    for (const int64_t b : bytes_) total_ += b;
}

void HelperSlotTable::close() {
    n_layers_ = n_expert_ = next_free_ = total_ = 0;
    bytes_.clear(); residency_.clear(); present_.clear();
}

int32_t HelperSlotTable::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return -1;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t HelperSlotTable::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return -1;
    if (next_free_ >= (int64_t) bytes_.size()) return -1;
    const int32_t slot = (int32_t) next_free_++;
    residency_[(size_t) (layer * n_expert_ + expert)] = slot;
    present_[(size_t) layer] = 1;
    return slot;
}

// ---------------------------------------------------------------- ForeignExperts
ForeignExperts::~ForeignExperts() { close(); }

bool ForeignExperts::call_failed(int rc, const char* what, std::string& err) const {
    if (rc == 0) return false;
    err = label_ + " experts: " + what + ": " + trimmed(errbuf_) + " (worker code " + std::to_string(rc) + ")";
    return true;
}

bool ForeignExperts::preflight(const HelperWorkerLibrary& lib, int device, double& free_gib, std::string& err) {
    const strata_helper_api& api = lib.api();
    const int count = api.device_count();
    if (device < 0 || device >= count) {
        err = "helper worker " + lib.backend() + ": device " + std::to_string(device) + " is not visible (" +
              std::to_string(count) + " present)";
        return false;
    }
    char buf[512] = {};
    const int rc = api.preflight(device, &free_gib, buf, sizeof buf);
    if (rc != 0) {
        err = "helper worker " + lib.backend() + ": preflight of device " + std::to_string(device) + ": " + trimmed(buf);
        return false;
    }
    return true;
}

void ForeignExperts::close() {
    if (dev_ && lib_) lib_->api().close(dev_);
    dev_ = nullptr;
    device_ = -1;
    table_.close();
    owned_.clear(); group_of_.clear(); group_id_.clear(); slot_.clear(); start_.clear(); tok_.clear();
    original_row_.clear();
    launched_ = false;
    lib_.reset();
}

bool ForeignExperts::open(std::shared_ptr<HelperWorkerLibrary> lib, int device, const std::string& label, int slots,
                          int64_t layers, int64_t experts, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                          const ExpertCache& primary, ExpertSource& source, std::vector<uint8_t>& claimed,
                          std::string& err, bool auto_size) {
    close();
    if (!lib) { err = "helper experts: no worker library"; return false; }
    lib_ = std::move(lib);
    label_ = label;
    const strata_helper_api& api = lib_->api();
    if (device < 0 || device >= api.device_count() || slots <= 0 || ranked.empty() || layers <= 0 || experts <= 0 ||
        claimed.size() != (size_t) layers * (size_t) experts) {
        err = label_ + " experts: need the device, ranked experts and a positive slot count";
        close(); return false;
    }
    device_ = device;
    const auto& lay = strata::kernels::cpu::expert_layout();
    native_ = lay.native;
    // APR (prism.hadamard) packs store their experts in a block-Hadamard-rotated basis: every path that feeds them
    // rotates its activations first (RemoteExperts::begin, the CPU pool, the prompt path).  Worker ABI v1 carries
    // neither the block size nor the sign vectors, and a worker that skips the rotation returns plausible-looking
    // garbage (hadamard.hpp), so such a pack is refused here, as the prompt peer refuses it in Prefill::set_peer.
    if (strata::kernels::cpu::hadamard_spec().any()) {
        err = label_ + " experts: this pack's experts are Hadamard-folded (APR, block " +
              std::to_string(strata::kernels::cpu::hadamard_spec().block) +
              "); helper worker ABI v1 cannot apply the rotation, so a worker cannot serve it";
        close(); return false;
    }
    double free_gib = 0;
    if (!preflight(*lib_, device, free_gib, err)) { close(); return false; }
    const uint64_t free_bytes = (uint64_t) (free_gib * 1073741824.0);

    // The same choice RemoteExperts makes: ranked pairs the primary does not hold and nobody claimed, until the
    // slots run out - or, sized automatically, until 512 MiB would be left on the device.
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) >= 0 || claimed[index] || picked[index]) continue;
        const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
        if (auto_size && needed + bytes + (512ull << 20) > free_bytes) break;
        selected.push_back(pair);
        needed += bytes;
        picked[index] = 1;
        if ((int) selected.size() >= slots) break;
    }
    if (selected.empty()) {
        err = label_ + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                  : " experts: no unclaimed experts remain");
        close(); return false;
    }
    if (needed + (512ull << 20) > free_bytes) {
        err = label_ + " experts: slots leave less than 512 MiB free; reduce --expert-cache-helper";
        close(); return false;
    }
    std::vector<int64_t> sizes;
    sizes.reserve(selected.size());
    for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    table_.open(layers, experts, sizes);

    strata_helper_open_params params{};
    params.n_slots = (int64_t) sizes.size();
    params.slot_bytes = sizes.data();
    params.max_entries = CAP;
    params.max_tokens = strata::kernels::cpu::MAXT;
    params.n_embd = strata::kernels::cpu::H;
    params.n_ff = strata::kernels::cpu::FF;
    params.native = native_ ? 1 : 0;
    if (call_failed(api.open(device, &params, &dev_, errbuf_, sizeof errbuf_), "open", err)) { close(); return false; }

    for (const auto& pair : selected) {
        const int32_t slot = table_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob) { err = label_ + " experts: cache fill failed"; close(); return false; }
        if (call_failed(api.fill_slot(dev_, slot, blob, (int64_t) lay.blob_bytes(pair.first), errbuf_, sizeof errbuf_),
                        "fill", err)) { close(); return false; }
    }
    // The only thing that says the worker holds the bytes it was given: read the first slot back.
    const auto& first = selected.front();
    if (call_failed(api.verify_slot(dev_, table_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                                    (int64_t) lay.blob_bytes(first.first), errbuf_, sizeof errbuf_), "verify", err)) {
        close(); return false;
    }
    for (const auto& pair : selected) claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    owned_.assign((size_t) CAP, 0);
    group_of_.assign((size_t) CAP, -1);
    group_id_.reserve((size_t) CAP); slot_.reserve((size_t) CAP); start_.reserve((size_t) CAP + 1);
    tok_.reserve((size_t) CAP); original_row_.reserve((size_t) CAP);
    computed_ = launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    return true;
}

double ForeignExperts::gib() const { return dev_ && lib_ ? lib_->api().gib(dev_) : 0.0; }

bool ForeignExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                           const int32_t* kind, const int32_t* primary_res, std::string& err) {
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    launched_ = false;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 || layer >= table_.n_layers() ||
        dev_ == nullptr) {
        err = label_ + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); slot_.clear(); start_.clear(); tok_.clear(); original_row_.clear();
    if (!table_.layer_present(layer)) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    const int64_t n_expert = table_.n_expert();
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert + (size_t) e] >= 0)) continue;
        const int32_t slot = table_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            slot_.push_back(slot);
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) tok_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            original_row_.push_back((int32_t) i);
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) tok_.size());

    strata_helper_launch launch{};
    launch.layer = layer;
    launch.n_tok = n_tok;
    launch.x = x;
    launch.n_groups = (int32_t) group_id_.size();
    launch.slot = slot_.data();
    launch.start = start_.data();
    launch.tok = tok_.data();
    launch.n_entries = (int32_t) tok_.size();
    if (native_) {
        const auto& fmt = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
        launch.gu_type = fmt.gu_type;
        launch.d_type = fmt.d_type;
    }
    if (call_failed(lib_->api().begin(dev_, &launch, errbuf_, sizeof errbuf_), "begin", err)) return false;
    launched_ = true;
    ++launched_layers_;
    const uint64_t compact = (uint64_t) tok_.size() * strata::kernels::cpu::H * sizeof(float);
    returned_bytes_ += compact;
    full_row_bytes_ += (uint64_t) n * strata::kernels::cpu::H * sizeof(float);
    return true;
}

bool ForeignExperts::finish(float* out, std::string& err) {
    if (!launched_) return true;
    launched_ = false;
    const auto w0 = std::chrono::steady_clock::now();
    const float* rows = nullptr;
    const int rc = lib_->api().finish(dev_, &rows, errbuf_, sizeof errbuf_);
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    if (call_failed(rc, "finish", err)) return false;
    if (!rows) { err = label_ + " experts: finish returned no rows"; return false; }
    constexpr size_t H = strata::kernels::cpu::H;
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, rows + i * H, H * sizeof(float));
    return true;
}

bool ForeignExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                                       uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (dev_ == nullptr || n == 0) return true;
    std::lock_guard<std::mutex> lock(read_mu_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        const int32_t slot = table_.slot_of(layers[i], experts[i]);
        if (slot < 0) continue;
        if (call_failed(lib_->api().read_slot(dev_, slot, dst[i], (int64_t) lay.blob_bytes(layers[i]), errbuf_, sizeof errbuf_),
                        "cache read", err)) return false;
        hit[i] = 1;
    }
    return true;
}
}  // namespace strata::core
