// tests/core/helper_abi_test.cpp - Frankenstein: the worker ABI through the engine's loader, on the CPU reference worker.
//
// No model, no GPU: the test loads the worker library the build produced, opens a few slots of known sizes, fills
// them with known bytes, verifies through the worker's own compare, reads them back, and checks the slot table.
// A launch with real experts is the engine's end-to-end check (bit-identical tokens with the worker enabled).
#include "strata/core/foreign_experts.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef STRATA_HELPER_CPU_REFERENCE_PATH
#define STRATA_HELPER_CPU_REFERENCE_PATH ""
#endif

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
    else std::fprintf(stderr, "ok: %s\n", what);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : STRATA_HELPER_CPU_REFERENCE_PATH;
    std::string err;
    auto lib = strata::core::HelperWorkerLibrary::load(path, err);
    check(lib != nullptr, ("load " + path + (lib ? "" : ": " + err)).c_str());
    if (!lib) return 1;
    check(lib->backend() == "cpu-reference", "backend name");
    const strata_helper_api& api = lib->api();
    check(api.abi_version == STRATA_HELPER_ABI_VERSION, "abi version");
    check(api.device_count() == 1, "one device");
    double free_gib = 0;
    check(strata::core::ForeignExperts::preflight(*lib, 0, free_gib, err), "preflight device 0");
    check(!strata::core::ForeignExperts::preflight(*lib, 7, free_gib, err), "preflight refuses device 7");

    // open three slots of different sizes, as a native pack's layers would ask
    const std::vector<int64_t> sizes{4096, 8192, 4096};
    strata_helper_open_params p{};
    p.n_slots = (int64_t) sizes.size(); p.slot_bytes = sizes.data();
    p.max_entries = 80; p.max_tokens = 8; p.n_embd = 2560; p.n_ff = 640; p.native = 1;
    strata_helper_device* dev = nullptr;
    char buf[256] = {};
    check(api.open(0, &p, &dev, buf, sizeof buf) == 0 && dev != nullptr, "open");
    if (!dev) return 1;
    std::vector<uint8_t> a(4096), b(8192), back(8192);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (uint8_t) (i * 7 + 1);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t) (i * 13 + 5);
    check(api.fill_slot(dev, 0, a.data(), 4096, buf, sizeof buf) == 0, "fill slot 0");
    check(api.fill_slot(dev, 1, b.data(), 8192, buf, sizeof buf) == 0, "fill slot 1");
    check(api.fill_slot(dev, 1, b.data(), 9000, buf, sizeof buf) != 0, "fill refuses more bytes than the slot");
    check(api.fill_slot(dev, 5, a.data(), 4096, buf, sizeof buf) != 0, "fill refuses a slot that does not exist");
    check(api.verify_slot(dev, 1, b.data(), 8192, buf, sizeof buf) == 0, "verify slot 1 matches");
    b[100] ^= 0xFF;
    check(api.verify_slot(dev, 1, b.data(), 8192, buf, sizeof buf) != 0, "verify detects a changed byte");
    b[100] ^= 0xFF;
    check(api.read_slot(dev, 1, back.data(), 8192, buf, sizeof buf) == 0 && std::memcmp(back.data(), b.data(), 8192) == 0,
          "read slot 1 back");
    check(api.gib(dev) > 0.0, "gib reports the slots");
    const float* rows = nullptr;
    check(api.finish(dev, &rows, buf, sizeof buf) != 0, "finish without begin is refused");
    api.close(dev);

    // the host slot table: admission, lookup, layer presence
    strata::core::HelperSlotTable t;
    t.open(4, 16, sizes);
    check(t.slot_of(1, 3) < 0, "empty table: not resident");
    check(t.admit(1, 3) == 0 && t.admit(2, 9) == 1 && t.admit(1, 4) == 2, "admit in order");
    check(t.admit(3, 0) < 0, "admit refuses when full");
    check(t.slot_of(1, 3) == 0 && t.slot_of(2, 9) == 1 && t.slot_of(1, 4) == 2 && t.slot_of(3, 0) < 0, "slot_of");
    check(t.layer_present(1) && t.layer_present(2) && !t.layer_present(0) && !t.layer_present(3), "layer presence");
    check(t.slot_of(99, 0) < 0 && t.slot_of(0, 99) < 0, "bounds");
    check(t.resident() == 3 && t.total_bytes() == 16384 && t.slot_bytes(1) == 8192, "sizes");
    std::fprintf(stderr, "%s\n", fails ? "helper_abi_test: FAILED" : "helper_abi_test: all checks passed");
    return fails ? 1 : 0;
}
