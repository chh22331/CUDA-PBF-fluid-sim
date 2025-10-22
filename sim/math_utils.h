#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace sim {
// 基础 float3 工具函数（统一定义，避免 simulator.cpp 多处重复 lambda）
static inline float3 make3(float x, float y, float z) { return make_float3(x, y, z); }
static inline float3 add3(float3 a, float3 b) { return make3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline float3 sub3(float3 a, float3 b) { return make3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline float3 mul3(float3 a, float s) { return make3(a.x * s, a.y * s, a.z * s); }
static inline float  dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float  len3(float3 a) { return std::sqrt(dot3(a, a)); }
static inline float3 cross3(float3 a, float3 b) {
    return make3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}
static inline float3 normalize3(float3 v) {
    float n2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (n2 <= 1e-20f) return make3(0, 0, 0);
    float inv = 1.0f / std::sqrt(n2);
    return make3(v.x * inv, v.y * inv, v.z * inv);
}
} // namespace sim
