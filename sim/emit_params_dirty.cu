#include "emit_params.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

__constant__ sim::EmitParams g_emitParams;

static uint64_t hash64(const void* d, size_t n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
    uint64_t h = 1469598103934665603ull;
    while (n--) { h ^= *p++; h *= 1099511628211ull; }
    return h;
}

extern "C" bool SetEmitParamsIfChanged(const sim::EmitParams* h, cudaStream_t s, bool enableCheck) {
    static sim::EmitParams prev{};
    static uint64_t prevHash = 0;
    if (!enableCheck) {
        SetEmitParamsAsync(h, s);
        return true;
    }
    uint64_t hval = hash64(h, sizeof(*h));
    if (hval == prevHash) return false;
    SetEmitParamsAsync(h, s);
    prev = *h;
    prevHash = hval;
    return true;
}