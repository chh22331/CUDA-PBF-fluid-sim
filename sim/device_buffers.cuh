#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); abort(); } } while(0)
#endif

struct DeviceBuffers {
    // 主状态
    float4* d_pos = nullptr;
    float4* d_pos_pred = nullptr;
    float4* d_vel = nullptr;

    // 邻域/排序
    uint32_t* d_indices = nullptr;
    uint32_t* d_indices_sorted = nullptr;
    uint32_t* d_cellKeys = nullptr;
    uint32_t* d_cellKeys_sorted = nullptr;
    uint32_t* d_cellStart = nullptr;
    uint32_t* d_cellEnd = nullptr;

    // PBF
    float* d_lambda = nullptr;
    float4* d_delta = nullptr;

    // sort 临时缓存
    void* d_sortTemp = nullptr;
    size_t sortTempBytes = 0;

    uint32_t capacity = 0;
    uint32_t numCellsCapacity = 0;

    // 外部绑定标志：d_pos_pred 来源于外部共享资源时，禁止在 free/resize 时 cudaFree
    bool posPredExternal = false;

    void allocate(uint32_t N);
    void allocGridRanges(uint32_t numCells);
    void resizeGridRanges(uint32_t numCells);
    void ensureSortTemp(size_t bytes);
    void freeAll();

    // 绑定/解绑外部 d_pos_pred 指针（绑定时释放原自有内存以避免泄漏）
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
};

namespace sim {

    struct DeviceBuffers {
        // 主状态（SoA，float4 对齐）
        float4* d_pos = nullptr;        // xyz: 位置, w: 保留
        float4* d_vel = nullptr;        // xyz: 速度, w: 保留
        float4* d_pos_pred = nullptr;   // 预测位置
        float* d_lambda = nullptr;      // PBF λ
        float4* d_delta = nullptr;      // 位置修正 Δx

        // 哈希 / 排序（稠密路径 + 压缩路径共享输入）
        uint32_t* d_cellKeys = nullptr;         // 每粒子cell key
        uint32_t* d_cellKeys_sorted = nullptr;  // 排序后的cell key
        uint32_t* d_indices = nullptr;          // 每粒子原始索引
        uint32_t* d_indices_sorted = nullptr;   // 排序后的索引

        // 稠密网格范围（保留以向后兼容旧路径）
        uint32_t* d_cellStart = nullptr;        // 每cell起始（exclusive prefix）
        uint32_t* d_cellEnd = nullptr;          // 每cell结束

        // —— 压缩网格段表（基于非空 cell） ——
        uint32_t* d_cellUniqueKeys = nullptr;   // M
        uint32_t* d_cellOffsets = nullptr;      // M+1
        uint32_t* d_compactCount = nullptr;     // 设备端单元素，记录 M
        uint32_t  compactCapacity = 0;          // 以 N 为上界（M <= N）

        // 临时与统计
        void* d_sortTemp = nullptr; size_t sortTempBytes = 0;

        uint32_t capacity = 0;

        // 外部绑定标志：若为 true，release/allocate 不应释放/重分配 d_pos_pred
        bool posPredExternal = false;

        void allocate(uint32_t cap) {
            if (cap == capacity) return;

            // 释放除 d_pos_pred（当为外部绑定时保留）之外的所有资源
            release();

            capacity = cap;
            CUDA_CHECK(cudaMalloc((void**)&d_pos, sizeof(float4) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_vel, sizeof(float4) * capacity));

            if (!posPredExternal) {
                CUDA_CHECK(cudaMalloc((void**)&d_pos_pred, sizeof(float4) * capacity));
            }
            CUDA_CHECK(cudaMalloc((void**)&d_lambda, sizeof(float) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_delta, sizeof(float4) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_cellKeys, sizeof(uint32_t) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_cellKeys_sorted, sizeof(uint32_t) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_indices, sizeof(uint32_t) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_indices_sorted, sizeof(uint32_t) * capacity));
            // grid 尺寸根据参数设置，由外部创建/重配 cellStart/End
        }

        void allocGridRanges(uint32_t numCells) {
            CUDA_CHECK(cudaMalloc((void**)&d_cellStart, sizeof(uint32_t) * numCells));
            CUDA_CHECK(cudaMalloc((void**)&d_cellEnd, sizeof(uint32_t) * numCells));
        }

        // 运行时重配 cellStart/End（当 numCells 变化时调用）
        void resizeGridRanges(uint32_t numCells) {
            if (d_cellStart) CUDA_CHECK(cudaFree(d_cellStart));
            if (d_cellEnd)   CUDA_CHECK(cudaFree(d_cellEnd));
            d_cellStart = d_cellEnd = nullptr;
            allocGridRanges(numCells);
        }

        // 确保压缩段表容量（按 N 上界分配 M<=N 的缓冲）
        void ensureCompactCapacity(uint32_t N) {
            uint32_t need = (N == 0u) ? 1u : N;
            if (need <= compactCapacity && d_cellUniqueKeys && d_cellOffsets && d_compactCount) return;
            if (d_cellUniqueKeys) CUDA_CHECK(cudaFree(d_cellUniqueKeys));
            if (d_cellOffsets)    CUDA_CHECK(cudaFree(d_cellOffsets));
            if (d_compactCount)   CUDA_CHECK(cudaFree(d_compactCount));
            compactCapacity = 0;
            // M <= N，offsets 需要 M+1，按 N+1 分配；compactCount 为 1
            CUDA_CHECK(cudaMalloc((void**)&d_cellUniqueKeys, sizeof(uint32_t) * need));
            CUDA_CHECK(cudaMalloc((void**)&d_cellOffsets,    sizeof(uint32_t) * (need + 1)));
            CUDA_CHECK(cudaMalloc((void**)&d_compactCount,   sizeof(uint32_t) * 1));
            compactCapacity = need;
        }

        void ensureSortTemp(size_t bytes) {
            if (bytes <= sortTempBytes) return;
            if (d_sortTemp) CUDA_CHECK(cudaFree(d_sortTemp));
            sortTempBytes = 0;
            CUDA_CHECK(cudaMalloc(&d_sortTemp, bytes));
            sortTempBytes = bytes;
        }

        // 仅在非外部绑定时释放 d_pos_pred；其他资源总是释放
        void release() {
            auto fre = [](void* p) { if (p) CUDA_CHECK(cudaFree(p)); };
            fre(d_pos); d_pos = nullptr;
            if (!posPredExternal) { fre(d_pos_pred); d_pos_pred = nullptr; }
            fre(d_vel); d_vel = nullptr;

            fre(d_lambda); d_lambda = nullptr;
            fre(d_delta);  d_delta = nullptr;

            fre(d_cellKeys); d_cellKeys = nullptr;
            fre(d_cellKeys_sorted); d_cellKeys_sorted = nullptr;
            fre(d_indices); d_indices = nullptr;
            fre(d_indices_sorted); d_indices_sorted = nullptr;
            fre(d_cellStart); d_cellStart = nullptr;
            fre(d_cellEnd);   d_cellEnd = nullptr;

            fre(d_cellUniqueKeys); d_cellUniqueKeys = nullptr;
            fre(d_cellOffsets);    d_cellOffsets = nullptr;
            fre(d_compactCount);   d_compactCount = nullptr;
            compactCapacity = 0;

            fre(d_sortTemp); d_sortTemp = nullptr; sortTempBytes = 0;

            capacity = 0;
        }

        // 绑定/解绑外部 d_pos_pred 指针（绑定时释放原自有内存以避免泄漏）
        void bindExternalPosPred(float4* ptr) {
            if (d_pos_pred && !posPredExternal) {
                CUDA_CHECK(cudaFree(d_pos_pred));
            }
            d_pos_pred = ptr;
            posPredExternal = true;
        }
        void detachExternalPosPred() {
            d_pos_pred = nullptr;
            posPredExternal = false;
        }

        ~DeviceBuffers() { release(); }
    };

} // namespace sim