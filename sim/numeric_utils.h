#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "parameters.h"

// 数值近似与结构比较工具，统一供各模块使用（原先散落于 simulator.cpp）
namespace sim {
    // 标量近似相等：允许相对误差 eps * max(1, |a|, |b|)
    static inline bool approxEq(float a, float b, float eps = 1e-6f) {
        float da = fabsf(a - b);
        float ma = fmaxf(fabsf(a), fabsf(b));
        return da <= eps * fmaxf(1.0f, ma);
    }
    // float3 分量逐一近似相等
    static inline bool approxEq3(float3 a, float3 b, float eps = 1e-6f) {
        return approxEq(a.x, b.x, eps) && approxEq(a.y, b.y, eps) && approxEq(a.z, b.z, eps);
    }
    // 网格参数整体比较（含 mins/maxs/cellSize/dim）
    static inline bool gridEqual(const GridBounds& a, const GridBounds& b, float eps = 1e-6f) {
        return approxEq3(a.mins, b.mins, eps) && approxEq3(a.maxs, b.maxs, eps) && approxEq(a.cellSize, b.cellSize, eps)
            && (a.dim.x == b.dim.x) && (a.dim.y == b.dim.y) && (a.dim.z == b.dim.z);
    }
    // Kernel 系数宽松比较：采用相对阈值 rel（默认 2%）
    static inline bool kernelEqualRelaxed(const KernelCoeffs& a, const KernelCoeffs& b, float rel = 0.02f) {
        auto nearRel = [&](float x, float y) {
            float mx = fmaxf(fabsf(x), fabsf(y));
            if (mx < 1e-9f) return true;
            return fabsf(x - y) <= rel * mx;
        };
        return nearRel(a.h, b.h) && nearRel(a.inv_h, b.inv_h) && nearRel(a.h2, b.h2)
            && nearRel(a.poly6, b.poly6) && nearRel(a.spiky, b.spiky) && nearRel(a.visc, b.visc);
    }
}
