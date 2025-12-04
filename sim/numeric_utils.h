#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "parameters.h"

// Numeric approximation utilities shared across modules.
// Keeps consistent tolerance checks for scalars, vectors, and structs.
namespace sim {

    // Approximate equality for scalars:
    // allows relative error eps * max(1, |a|, |b|)
    static inline bool approxEq(float a, float b, float eps = 1e-6f) {
        float da = fabsf(a - b);
        float ma = fmaxf(fabsf(a), fabsf(b));
        return da <= eps * fmaxf(1.0f, ma);
    }

    // Component-wise approximate equality for float3
    static inline bool approxEq3(float3 a, float3 b, float eps = 1e-6f) {
        return approxEq(a.x, b.x, eps) && approxEq(a.y, b.y, eps) && approxEq(a.z, b.z, eps);
    }

    // Grid parameter equality (mins/maxs/cellSize/dim)
    static inline bool gridEqual(const GridBounds& a, const GridBounds& b, float eps = 1e-6f) {
        return approxEq3(a.mins, b.mins, eps) &&
               approxEq3(a.maxs, b.maxs, eps) &&
               approxEq(a.cellSize, b.cellSize, eps) &&
               (a.dim.x == b.dim.x) && (a.dim.y == b.dim.y) && (a.dim.z == b.dim.z);
    }

    // Relaxed comparison of kernel coefficients:
    // uses relative threshold 'rel' (default 2%)
    static inline bool kernelEqualRelaxed(const KernelCoeffs& a, const KernelCoeffs& b, float rel = 0.02f) {
        auto nearRel = [&](float x, float y) {
            float mx = fmaxf(fabsf(x), fabsf(y));
            if (mx < 1e-9f) return true;        // treat near-zero as equal
            return fabsf(x - y) <= rel * mx;    // relative tolerance
        };
        return nearRel(a.h, a.h) && nearRel(a.h, b.h) && nearRel(a.inv_h, b.inv_h) &&
               nearRel(a.h2, b.h2) && nearRel(a.poly6, b.poly6) &&
               nearRel(a.spiky, b.spiky) && nearRel(a.visc, b.visc);
    }

}
