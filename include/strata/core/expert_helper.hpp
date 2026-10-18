// include/strata/core/expert_helper.hpp - Frankenstein: the engine's view of ONE helper expert tier.
//
// A helper holds a static, profile-filled set of (layer, expert) pairs in its own memory and computes the rows
// routed to them while the CPU pool works on the rest.  CUDA0 never sees the helper's device: input and results
// cross through host memory the helper owns.  `RemoteExperts` (remote_experts.hpp) is the in-process CUDA1..3
// implementation this interface was lifted from; `ForeignExperts` (foreign_experts.hpp) drives a worker library
// loaded at run time through the C ABI in strata/helper/strata_helper_abi.h - a HIP build of the same kernels for
// an AMD card next to the CUDA cards, or the CPU reference worker the tests use.
//
// The per-layer protocol, called from expert_pool_dispatch / _multi (expert_source.cpp):
//   begin(layer, x, ids, ...)  claims the entries this helper holds (owns(i) afterwards) and starts its work;
//   the CPU pool computes the rest;
//   finish(out)                waits and writes the claimed rows of `out` (H floats each, at their routed index).
// `holds(layer, expert)` answers from the helper's own table, live, so an adaptive tier that moves experts later is
// followed rather than a copy taken at load.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

class IExpertHelper {
public:
    virtual ~IExpertHelper() = default;

    /// A short name for the startup and per-request reports ("CUDA1", "HIP0", "cpu-reference").
    virtual std::string label() const = 0;

    /// `kind` is the primary verifier's classification (-1 = CPU candidate), or null on the one-token path.
    /// Entries already served on CUDA0 (`primary_res` >= 0) are excluded.
    virtual bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                       const int32_t* kind, const int32_t* primary_res, std::string& err) = 0;
    /// Whether begin() claimed entry `index` of the current window (index = token * k + routed slot).
    virtual bool owns(int64_t index) const = 0;
    /// Whether this helper's cache holds (layer, expert) right now.
    virtual bool holds(int64_t layer, int32_t expert) const = 0;
    /// Waits for the helper and writes its rows into `out` (H floats per entry, at the entry's routed index).
    virtual bool finish(float* out, std::string& err) = 0;

    // ---- the reports
    virtual int64_t resident() const = 0;          ///< experts held
    virtual int64_t computed() const = 0;          ///< entries computed since open
    virtual int64_t launched_layers() const = 0;   ///< layers with at least one entry since open
    virtual double gib() const = 0;                ///< device memory the cache takes
    virtual uint64_t returned_bytes() const = 0;   ///< bytes of packed rows returned
    virtual uint64_t full_row_bytes() const = 0;   ///< what full rows would have been
    virtual double ms_begin() const = 0;           ///< host time in begin(), cumulative
    virtual double ms_wait() const = 0;            ///< host time waiting in finish(), cumulative

    /// STRATA_PREFILL_HELPER_SOURCE: copy up to `n` cached blobs into `dst[i]` and wait; `hit[i]` is set where
    /// this helper held (layer, expert).  Misses are the caller's.  False with `err` on a device failure; the
    /// hits copied so far are complete.  Safe from several threads.
    virtual bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                                   uint8_t* hit, std::string& err) = 0;
};

}  // namespace strata::core
