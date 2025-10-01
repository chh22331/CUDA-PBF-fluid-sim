#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); abort(); } } while(0)
#endif

namespace sim {

    struct DeviceBuffers {
        // 主状态（SoA，float4 对齐）
        float4* d_pos = nullptr;        // xyz: 位置, w: 保留
        float4* d_vel = nullptr;        // xyz: 速度, w: 保留
        float4* d_pos_pred = nullptr;   // 预测位置
        float* d_lambda = nullptr;      // PBF λ
        float4* d_delta = nullptr;      // 位置修正 Δx

        // 哈希 / 排序
        uint32_t* d_cellKeys = nullptr;         // 每粒子cell key
        uint32_t* d_cellKeys_sorted = nullptr;  // 排序后的cell key
        uint32_t* d_indices = nullptr;          // 每粒子原始索引
        uint32_t* d_indices_sorted = nullptr;   // 排序后的索引
        uint32_t* d_cellStart = nullptr;        // 每cell起始（exclusive prefix）
        uint32_t* d_cellEnd = nullptr;          // 每cell结束

        // 临时与统计
        void* d_sortTemp = nullptr; size_t sortTempBytes = 0;

        uint32_t capacity = 0;

        void allocate(uint32_t cap) {
            if (cap == capacity) return;
            release();
            capacity = cap;
            CUDA_CHECK(cudaMalloc((void**)&d_pos, sizeof(float4) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_vel, sizeof(float4) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_pos_pred, sizeof(float4) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_lambda, sizeof(float) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_delta, sizeof(float4) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_cellKeys, sizeof(uint32_t) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_cellKeys_sorted, sizeof(uint32_t) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_indices, sizeof(uint32_t) * capacity));
            CUDA_CHECK(cudaMalloc((void**)&d_indices_sorted, sizeof(uint32_t) * capacity));
            // grid 尺寸根据参数设置，由外部创建 cellStart/End
        }

        void allocGridRanges(uint32_t numCells) {
            CUDA_CHECK(cudaMalloc((void**)&d_cellStart, sizeof(uint32_t) * numCells));
            CUDA_CHECK(cudaMalloc((void**)&d_cellEnd, sizeof(uint32_t) * numCells));
        }

        void ensureSortTemp(size_t bytes) {
            if (bytes <= sortTempBytes) return;
            if (d_sortTemp) CUDA_CHECK(cudaFree(d_sortTemp));
            sortTempBytes = 0;
            CUDA_CHECK(cudaMalloc(&d_sortTemp, bytes));
            sortTempBytes = bytes;
        }

        void release() {
            auto fre = [](void* p) { if (p) CUDA_CHECK(cudaFree(p)); };
            fre(d_pos); fre(d_vel); fre(d_pos_pred); fre(d_lambda); fre(d_delta);
            fre(d_cellKeys); fre(d_cellKeys_sorted);
            fre(d_indices); fre(d_indices_sorted);
            fre(d_cellStart); fre(d_cellEnd); fre(d_sortTemp);
            d_pos = d_vel = d_pos_pred = nullptr; d_lambda = nullptr; d_delta = nullptr;
            d_cellKeys = d_cellKeys_sorted = nullptr;
            d_indices = d_indices_sorted = nullptr;
            d_cellStart = d_cellEnd = nullptr; d_sortTemp = nullptr; sortTempBytes = 0;
            capacity = 0;
        }

        ~DeviceBuffers() { release(); }
    };

} // namespace sim