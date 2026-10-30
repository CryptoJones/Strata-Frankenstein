// include/strata/core/foreign_experts.hpp - Frankenstein: a helper expert tier on a worker library's device.
//
// The host side of a helper that lives behind strata/helper/strata_helper_abi.h: this class keeps the
// (layer, expert) -> slot table, picks which ranked experts the worker gets (the same rule RemoteExperts uses),
// fills and verifies them through the worker, and per layer turns the routed entries into grouped slot work the
// worker computes.  The worker's device runtime is its own; nothing of it reaches this translation unit.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_helper.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/helper/strata_helper_abi.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// A loaded worker library.  One per path; several ForeignExperts may share it (several devices of one backend).
class HelperWorkerLibrary {
public:
    ~HelperWorkerLibrary();
    HelperWorkerLibrary(const HelperWorkerLibrary&) = delete;
    HelperWorkerLibrary& operator=(const HelperWorkerLibrary&) = delete;
    /// dlopen `path` and resolve its API table; null with `err` when it is not a worker of this ABI version.
    static std::unique_ptr<HelperWorkerLibrary> load(const std::string& path, std::string& err);
    const strata_helper_api& api() const { return *api_; }
    const std::string& path() const { return path_; }
    std::string backend() const { return api_->backend ? api_->backend : "?"; }

private:
    HelperWorkerLibrary() = default;
    void* handle_ = nullptr;
    const strata_helper_api* api_ = nullptr;
    std::string path_;
};

/// The host-side slot table of a foreign cache: `(layer, expert)` -> slot, the sizes of the slots, and the layers
/// present.  What ExpertCache keeps beside its device arena, without the arena.
class HelperSlotTable {
public:
    void open(int64_t n_layers, int64_t n_expert, std::vector<int64_t> slot_bytes);
    void close();
    int32_t slot_of(int64_t layer, int64_t expert) const;
    /// Claims the next free slot for (layer, expert); -1 when full.
    int32_t admit(int64_t layer, int64_t expert);
    int64_t slots() const { return (int64_t) bytes_.size(); }
    int64_t resident() const { return next_free_; }
    int64_t slot_bytes(int32_t slot) const { return bytes_[(size_t) slot]; }
    int64_t total_bytes() const { return total_; }
    bool layer_present(int64_t layer) const { return layer >= 0 && layer < n_layers_ && present_[(size_t) layer] != 0; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }

private:
    int64_t n_layers_ = 0, n_expert_ = 0, next_free_ = 0, total_ = 0;
    std::vector<int32_t> residency_;
    std::vector<int64_t> bytes_;
    std::vector<uint8_t> present_;
};

class ForeignExperts final : public IExpertHelper {
public:
    ForeignExperts() = default;
    ~ForeignExperts() override;
    ForeignExperts(const ForeignExperts&) = delete;
    ForeignExperts& operator=(const ForeignExperts&) = delete;

    /// Create the worker device's context before the engine's host arena is registered; reports free GiB.
    static bool preflight(const HelperWorkerLibrary& lib, int device, double& free_gib, std::string& err);

    /// Fill the worker's device `device` with up to `slots` of `ranked` that neither the primary cache holds
    /// nor `claimed` marks (or as many as fit, `auto_size`), from `source`'s blobs.  Marks what it took in
    /// `claimed`.  `label` names the helper in reports ("HIP0").
    bool open(std::shared_ptr<HelperWorkerLibrary> lib, int device, const std::string& label, int slots,
              int64_t layers, int64_t experts, const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source, std::vector<uint8_t>& claimed, std::string& err,
              bool auto_size = false);
    void close();

    std::string label() const override { return label_; }
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
               const int32_t* primary_res, std::string& err) override;
    bool owns(int64_t index) const override { return owned_[(size_t) index] != 0; }
    /// In shadow mode (set_shadow) the worker's experts steer nothing: the GPU plan counts them as CPU misses (the
    /// PCIe share and the plan are the run's without the worker), so this answers false; begin() still claims the
    /// rows that are left to the CPU and that the worker holds.
    bool holds(int64_t layer, int32_t expert) const override { return !shadow_ && table_.slot_of(layer, expert) >= 0; }
    /// --helper-shadow: take only rows the CPU pool would have computed.  With a worker that computes as the pool
    /// does (the CPU reference worker), a run must then be byte-identical to the same run without the worker.
    void set_shadow(bool on) { shadow_ = on; }
    bool shadow() const { return shadow_; }
    bool finish(float* out, std::string& err) override;
    int64_t resident() const override { return table_.resident(); }
    int64_t computed() const override { return computed_; }
    int64_t launched_layers() const override { return launched_layers_; }
    double gib() const override;
    uint64_t returned_bytes() const override { return returned_bytes_; }
    uint64_t full_row_bytes() const override { return full_row_bytes_; }
    double ms_begin() const override { return ms_begin_; }
    double ms_wait() const override { return ms_wait_; }
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err) override;

    const HelperSlotTable& table() const { return table_; }

private:
    bool call_failed(int rc, const char* what, std::string& err) const;

    std::shared_ptr<HelperWorkerLibrary> lib_;
    strata_helper_device* dev_ = nullptr;
    int device_ = -1;
    std::string label_;
    HelperSlotTable table_;
    bool native_ = false;
    bool shadow_ = false;
    int64_t computed_ = 0, launched_layers_ = 0;
    uint64_t returned_bytes_ = 0, full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    char errbuf_[512] = {};
    std::mutex read_mu_;
    // the current window's grouping
    std::vector<uint8_t> owned_;
    std::vector<int32_t> group_of_, group_id_, slot_, start_, tok_, original_row_;
    bool launched_ = false;
};

}  // namespace strata::core
