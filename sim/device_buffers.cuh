#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "parameters.h"
#include "../engine/core/console.h"

namespace sim {

    struct Half4 { __half x, y, z, w; };

    // 前置包装函数声明（实现于 device_buffers_kernels.cu）
    void PackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N, cudaStream_t s);
    void UnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N, cudaStream_t s);

    struct DeviceBuffers {
        // FP32 主数据
        float4* d_pos = nullptr;
        float4* d_vel = nullptr;
        float4* d_pos_pred = nullptr;
        float* d_lambda = nullptr;
        float4* d_delta = nullptr;

        // 半精镜像
        Half4* d_pos_h4 = nullptr;
        Half4* d_vel_h4 = nullptr;
        Half4* d_pos_pred_h4 = nullptr;
        Half4* d_prev_pos_h4 = nullptr;

        // 网格 / 排序
        uint32_t* d_cellKeys = nullptr;
        uint32_t* d_cellKeys_sorted = nullptr;
        uint32_t* d_indices = nullptr;
        uint32_t* d_indices_sorted = nullptr;
        uint32_t* d_cellStart = nullptr;
        uint32_t* d_cellEnd = nullptr;

        // 压缩网格
        uint32_t* d_cellUniqueKeys = nullptr;
        uint32_t* d_cellOffsets = nullptr;
        uint32_t* d_compactCount = nullptr;
        uint32_t  compactCapacity = 0;

        // 排序临时
        void* d_sortTemp = nullptr;
        size_t    sortTempBytes = 0;

        uint32_t  capacity = 0;
        bool      posPredExternal = false;

        // 精度标志
        bool usePosHalf = false;
        bool useVelHalf = false;
        bool usePosPredHalf = false;

        bool anyHalf() const { return usePosHalf || useVelHalf || usePosPredHalf; }
        bool hasPrevSnapshot() const { return d_prev_pos_h4 != nullptr; }

        // 分配接口
        void allocate(uint32_t cap) { 
            allocateInternal(cap, false, false, false); 
            if (console::Instance().debug.usePrevPosHalfSnapshot) {
                if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; }
                cudaMalloc((void**)&d_prev_pos_h4, sizeof(sim::Half4) * capacity);
            }
        }
        void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
            bool posH = (prec.positionStore == sim::NumericType::FP16_Packed ||
                prec.positionStore == sim::NumericType::FP16);
            bool velH = (prec.velocityStore == sim::NumericType::FP16_Packed ||
                prec.velocityStore == sim::NumericType::FP16);
            bool predH = (prec.predictedPosStore == sim::NumericType::FP16_Packed ||
                prec.predictedPosStore == sim::NumericType::FP16);
            allocateInternal(cap, posH, velH, predH);
            if (console::Instance().debug.usePrevPosHalfSnapshot) {
                if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; }
                cudaMalloc((void**)&d_prev_pos_h4, sizeof(sim::Half4) * capacity);
            }
        }
        void allocatePrevPosSnapshot(uint32_t cap) {
            if (cap == 0) return;
            if (!d_prev_pos_h4) {
                cudaMalloc((void**)&d_prev_pos_h4, sizeof(sim::Half4) * cap);
            }
        }
        void freePrevPosSnapshot() {
            if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; }
        }
        void allocateInternal(uint32_t cap, bool posH, bool velH, bool predH);

        void ensureCompactCapacity(uint32_t N);
        void ensureSortTemp(size_t bytes);

        void resizeGridRanges(uint32_t numCells);
        void allocGridRanges(uint32_t numCells);

        void bindExternalPosPred(float4* ptr) {
            if (d_pos_pred && !posPredExternal) {
                cudaFree(d_pos_pred);
            }
            d_pos_pred = ptr;
            posPredExternal = true;
        }
        void detachExternalPosPred() {
            d_pos_pred = nullptr;
            posPredExternal = false;
        }

        void release();

        // 半精打包/解包
        void packAllToHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            if (usePosHalf && d_pos && d_pos_h4)       PackFloat4ToHalf4(d_pos, d_pos_h4, N, s);
            if (useVelHalf && d_vel && d_vel_h4)       PackFloat4ToHalf4(d_vel, d_vel_h4, N, s);
            if (usePosPredHalf && d_pos_pred && d_pos_pred_h4) PackFloat4ToHalf4(d_pos_pred, d_pos_pred_h4, N, s);
        }
        void unpackAllFromHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            if (usePosHalf && d_pos && d_pos_h4)       UnpackHalf4ToFloat4(d_pos_h4, d_pos, N, s);
            if (useVelHalf && d_vel && d_vel_h4)       UnpackHalf4ToFloat4(d_vel_h4, d_vel, N, s);
            if (usePosPredHalf && d_pos_pred && d_pos_pred_h4) UnpackHalf4ToFloat4(d_pos_pred_h4, d_pos_pred, N, s);
        }

        ~DeviceBuffers() { release(); }
    };

    // ---- 内联实现 ----
    inline void DeviceBuffers::allocateInternal(uint32_t cap, bool posH, bool velH, bool predH) {
        if (cap == 0) cap = 1;
        if (cap == capacity &&
            posH == usePosHalf && velH == useVelHalf && predH == usePosPredHalf)
            return;
        release();
        capacity = cap;
        usePosHalf = posH;
        useVelHalf = velH;
        usePosPredHalf = predH;

        cudaMalloc((void**)&d_pos, sizeof(float4) * capacity);
        cudaMalloc((void**)&d_vel, sizeof(float4) * capacity);
        if (!posPredExternal) {
            cudaMalloc((void**)&d_pos_pred, sizeof(float4) * capacity);
        }
        cudaMalloc((void**)&d_lambda, sizeof(float) * capacity);
        cudaMalloc((void**)&d_delta, sizeof(float4) * capacity);
        cudaMalloc((void**)&d_cellKeys, sizeof(uint32_t) * capacity);
        cudaMalloc((void**)&d_cellKeys_sorted, sizeof(uint32_t) * capacity);
        cudaMalloc((void**)&d_indices, sizeof(uint32_t) * capacity);
        cudaMalloc((void**)&d_indices_sorted, sizeof(uint32_t) * capacity);

        if (usePosHalf)      cudaMalloc((void**)&d_pos_h4, sizeof(Half4) * capacity);
        if (useVelHalf)      cudaMalloc((void**)&d_vel_h4, sizeof(Half4) * capacity);
        if (usePosPredHalf)  cudaMalloc((void**)&d_pos_pred_h4, sizeof(Half4) * capacity);
    }

    inline void DeviceBuffers::allocGridRanges(uint32_t numCells) {
        cudaMalloc((void**)&d_cellStart, sizeof(uint32_t) * numCells);
        cudaMalloc((void**)&d_cellEnd, sizeof(uint32_t) * numCells);
    }

    inline void DeviceBuffers::resizeGridRanges(uint32_t numCells) {
        if (d_cellStart) cudaFree(d_cellStart);
        if (d_cellEnd)   cudaFree(d_cellEnd);
        d_cellStart = d_cellEnd = nullptr;
        allocGridRanges(numCells);
    }

    inline void DeviceBuffers::ensureCompactCapacity(uint32_t N) {
        uint32_t need = (N == 0u) ? 1u : N;
        if (need <= compactCapacity && d_cellUniqueKeys && d_cellOffsets && d_compactCount) return;
        if (d_cellUniqueKeys) cudaFree(d_cellUniqueKeys);
        if (d_cellOffsets)    cudaFree(d_cellOffsets);
        if (d_compactCount)   cudaFree(d_compactCount);
        cudaMalloc((void**)&d_cellUniqueKeys, sizeof(uint32_t) * need);
        cudaMalloc((void**)&d_cellOffsets, sizeof(uint32_t) * (need + 1));
        cudaMalloc((void**)&d_compactCount, sizeof(uint32_t) * 1);
        compactCapacity = need;
    }

    inline void DeviceBuffers::ensureSortTemp(size_t bytes) {
        if (bytes <= sortTempBytes) return;
        if (d_sortTemp) cudaFree(d_sortTemp);
        cudaMalloc(&d_sortTemp, bytes);
        sortTempBytes = bytes;
    }

    inline void DeviceBuffers::release() {
        // 使用泛型 lambda 以支持不同指针类型
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };

        fre(d_pos);
        if (!posPredExternal) fre(d_pos_pred);
        fre(d_vel);
        fre(d_lambda);
        fre(d_delta);
        fre(d_cellKeys);
        fre(d_cellKeys_sorted);
        fre(d_indices);
        fre(d_indices_sorted);
        fre(d_cellStart);
        fre(d_cellEnd);
        fre(d_cellUniqueKeys);
        fre(d_cellOffsets);
        fre(d_compactCount);
        fre(d_sortTemp);

        fre(d_pos_h4);
        fre(d_vel_h4);
        fre(d_pos_pred_h4);

        sortTempBytes = 0;
        compactCapacity = 0;
        capacity = 0;
        posPredExternal = false;
        usePosHalf = useVelHalf = usePosPredHalf = false;
    }

} // namespace sim