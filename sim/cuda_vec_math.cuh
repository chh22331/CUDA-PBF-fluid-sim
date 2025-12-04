#pragma once

#include <cuda_runtime.h>
#include <math.h>

__host__ __device__ inline float3 to_float3(const float4& a)
{
    return make_float3(a.x, a.y, a.z);
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}

__host__ __device__ inline float3& operator-=(float3& a, const float3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}

__host__ __device__ inline float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3& operator*=(float3& a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; return a;
}

__host__ __device__ inline float3 operator/(const float3& a, float s)
{
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__host__ __device__ inline float3& operator/=(float3& a, float s)
{
    float inv = 1.0f / s;
    a.x *= inv; a.y *= inv; a.z *= inv; return a;
}

__host__ __device__ inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// ===== float3 basic functions =====

__host__ __device__ inline float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length2(const float3& a)
{
    return dot(a, a);
}

__host__ __device__ inline float length(const float3& a)
{
    return sqrtf(length2(a));
}

__host__ __device__ inline float3 normalize(const float3& a)
{
    float len = length(a);
    return (len > 0.0f) ? (a / len) : make_float3(0.0f, 0.0f, 0.0f);
}

__host__ __device__ inline float3 hadamard(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 clamp(const float3& v, float minv, float maxv)
{
    float3 r;
    r.x = fminf(fmaxf(v.x, minv), maxv);
    r.y = fminf(fmaxf(v.y, minv), maxv);
    r.z = fminf(fmaxf(v.z, minv), maxv);
    return r;
}