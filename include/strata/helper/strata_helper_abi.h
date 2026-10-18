/* include/strata/helper/strata_helper_abi.h - Frankenstein: the C ABI between the engine and a helper worker.
 *
 * WHY A C ABI AND A SEPARATE LIBRARY.  Strata is compiled for ONE device backend: on HIP the CUDA runtime names are
 * renamed by a force-included header, so one binary cannot hold CUDA and HIP code.  A helper worker is a shared
 * library compiled for its own backend with every symbol hidden except `strata_helper_api_v1`, loaded with dlopen by
 * the CUDA engine: two runtimes, two copies of the kernels, no shared C++ types across the boundary.  Everything
 * here is plain C: integers, pointers to host memory the ENGINE owns (read during the call only) or the WORKER owns
 * (valid until the next call that replaces it), and error text copied into a caller's buffer.
 *
 * MEMORY RULES.  The engine never hands the worker memory registered with its own device runtime, and the worker
 * never hands the engine device memory.  Blobs arrive as pageable host pointers (the expert arena); the worker
 * stages them itself.  The activation `x` is pageable host memory the worker copies at begin(); the rows it returns
 * are its own host buffer, pinned or not, that the engine copies out of in finish().
 *
 * SLOTS.  The engine keeps the (layer, expert) -> slot table.  The worker only knows slot numbers and the bytes of
 * each slot it was asked to allocate at open(); a native pack's blobs differ per layer, and a profile-filled tier
 * never moves an expert to another layer's slot, so each slot keeps its first size.
 *
 * VERSIONING.  `abi_version` is the contract; `struct_size` lets a newer engine refuse a shorter table.  A worker
 * built for another version returns NULL from `strata_helper_api_v1`, and the engine says so by name.
 */
#ifndef STRATA_HELPER_ABI_H
#define STRATA_HELPER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRATA_HELPER_ABI_VERSION 1u
#define STRATA_HELPER_ENTRY_SYMBOL "strata_helper_api_v1"

/* One open device of one worker: opaque to the engine. */
typedef struct strata_helper_device strata_helper_device;

/* The geometry of one open: how many slots of which sizes, and the bounds of a launch. */
typedef struct strata_helper_open_params {
    int64_t n_slots;             /* slots to allocate */
    const int64_t* slot_bytes;   /* n_slots entries: bytes of each slot (a blob of its layer, 256-aligned by the worker) */
    int64_t max_entries;         /* most (token, routed expert) entries one launch carries */
    int64_t max_tokens;          /* most tokens one launch carries */
    int64_t n_embd;              /* H: floats per activation row and per output row */
    int64_t n_ff;                /* the expert's intermediate width */
    int32_t native;              /* 1: native (GGUF i-quant / k-quant) experts; 0: the Q2_0 S2 pack */
} strata_helper_open_params;

/* One layer's work: the entries this helper was given, grouped by expert.
 * Group g computes slot `slot[g]` for entries start[g] .. start[g+1]-1; entry i reads activation row tok[i]
 * and produces packed output row i.  The engine maps packed rows back to their routed index itself. */
typedef struct strata_helper_launch {
    int64_t layer;
    int64_t n_tok;               /* activation rows in `x` */
    const float* x;              /* host, n_tok * n_embd floats, valid during begin() only */
    int32_t n_groups;
    const int32_t* slot;         /* n_groups */
    const int32_t* start;        /* n_groups + 1 */
    const int32_t* tok;          /* n_entries */
    int32_t n_entries;
    /* the layer's native format (ignored when the open was not native) */
    int32_t gu_type;             /* ggml type of gate/up */
    int32_t d_type;              /* ggml type of down */
} strata_helper_launch;

/* Every function returns 0 on success; otherwise a non-zero code and a NUL-terminated message in err (cap bytes). */
typedef struct strata_helper_api {
    uint32_t abi_version;        /* STRATA_HELPER_ABI_VERSION */
    uint32_t struct_size;        /* sizeof(strata_helper_api) of the worker */
    const char* backend;         /* "hip", "cuda", "cpu-reference", ... */

    int (*device_count)(void);
    int (*device_name)(int device, char* name, size_t cap);
    /* Create the device's context before the engine registers its host arena; reports free memory in GiB. */
    int (*preflight)(int device, double* free_gib, char* err, size_t cap);
    int (*open)(int device, const strata_helper_open_params* params, strata_helper_device** out,
                char* err, size_t cap);
    void (*close)(strata_helper_device* dev);

    /* Blocking: copy `bytes` of `host_blob` into `slot`. */
    int (*fill_slot)(strata_helper_device* dev, int32_t slot, const uint8_t* host_blob, int64_t bytes,
                     char* err, size_t cap);
    /* Blocking: read `slot` back and compare with `host_blob` byte for byte; non-zero if they differ. */
    int (*verify_slot)(strata_helper_device* dev, int32_t slot, const uint8_t* host_blob, int64_t bytes,
                       char* err, size_t cap);
    /* Blocking: copy `bytes` of `slot` into `dst` (the prompt path's helper-cache reads). */
    int (*read_slot)(strata_helper_device* dev, int32_t slot, uint8_t* dst, int64_t bytes, char* err, size_t cap);

    /* Start one layer's work; returns once the work is queued (asynchronous where the backend allows). */
    int (*begin)(strata_helper_device* dev, const strata_helper_launch* launch, char* err, size_t cap);
    /* Wait for the layer; `*rows` then points at n_entries * n_embd floats the worker owns, valid until the
     * next begin() or close(). */
    int (*finish)(strata_helper_device* dev, const float** rows, char* err, size_t cap);

    /* Device memory the open slots take, in GiB. */
    double (*gib)(strata_helper_device* dev);
} strata_helper_api;

/* The one exported symbol of a worker library. */
typedef const strata_helper_api* (*strata_helper_entry_fn)(void);

#ifdef __cplusplus
}
#endif

#endif /* STRATA_HELPER_ABI_H */
