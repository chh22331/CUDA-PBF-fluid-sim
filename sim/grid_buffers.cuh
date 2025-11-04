#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {
    struct GridBuffers {
        // Per-particle arrays (size = capacity)
        uint32_t* d_cellKeys = nullptr;
        uint32_t* d_cellKeys_sorted = nullptr;
        uint32_t* d_indices = nullptr;
        uint32_t* d_indices_sorted = nullptr;

        // Per-cell arrays (size = numCells)
        uint32_t* d_cellStart = nullptr;
        uint32_t* d_cellEnd = nullptr;

        // Compacted grid
        uint32_t* d_cellUniqueKeys = nullptr;
        uint32_t* d_cellOffsets = nullptr; // size = compactCapacity + 1
        uint32_t* d_compactCount = nullptr; // size = 1
        uint32_t  compactCapacity = 0;

        // Hash (direct) mapping: cellKey -> compact index (size = numCells)
        // 0xFFFFFFFF表示该 cell为空（未出现在压缩列表中）
        uint32_t* d_keyToCompact = nullptr;

        // Sort temporary storage
        void*   d_sortTemp = nullptr;
        size_t  sortTempBytes = 0;

        uint32_t capacity = 0; // particle capacity for key/index arrays
        uint32_t numCells = 0; // current cell count (for memset etc.)

        void allocateIndices(uint32_t cap) {
            if (cap == 0) cap = 1;
            if (cap == capacity) return;
            releaseIndices();
            capacity = cap;
            cudaMalloc((void**)&d_cellKeys, sizeof(uint32_t) * capacity);
            cudaMalloc((void**)&d_cellKeys_sorted, sizeof(uint32_t) * capacity);
            cudaMalloc((void**)&d_indices, sizeof(uint32_t) * capacity);
            cudaMalloc((void**)&d_indices_sorted, sizeof(uint32_t) * capacity);
        }

        void allocGridRanges(uint32_t nCells) {
            numCells = nCells;
            cudaMalloc((void**)&d_cellStart, sizeof(uint32_t) * numCells);
            cudaMalloc((void**)&d_cellEnd, sizeof(uint32_t) * numCells);
            // allocate hash mapping buffer
            if (d_keyToCompact) cudaFree(d_keyToCompact);
            cudaMalloc((void**)&d_keyToCompact, sizeof(uint32_t) * numCells);
            cudaMemset(d_keyToCompact, 0xFF, sizeof(uint32_t) * numCells);
        }
        void resizeGridRanges(uint32_t nCells) {
            if (d_cellStart) cudaFree(d_cellStart);
            if (d_cellEnd)   cudaFree(d_cellEnd);
            if (d_keyToCompact) cudaFree(d_keyToCompact);
            d_cellStart = d_cellEnd = nullptr; d_keyToCompact = nullptr;
            numCells = nCells;
            if (numCells) allocGridRanges(numCells);
        }

        void ensureCompactCapacity(uint32_t N) {
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

        void ensureSortTemp(size_t bytes) {
            if (bytes <= sortTempBytes) return;
            if (d_sortTemp) cudaFree(d_sortTemp);
            cudaMalloc(&d_sortTemp, bytes);
            sortTempBytes = bytes;
        }

        void releaseIndices() {
            auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            fre(d_cellKeys); fre(d_cellKeys_sorted); fre(d_indices); fre(d_indices_sorted);
        }
        void releaseGrid() {
            auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            fre(d_cellStart); fre(d_cellEnd);
            fre(d_cellUniqueKeys); fre(d_cellOffsets); fre(d_compactCount);
            fre(d_keyToCompact);
            fre(d_sortTemp);
            sortTempBytes = 0; compactCapacity = 0; numCells = 0;
        }
        void releaseAll() { releaseIndices(); releaseGrid(); capacity = 0; }
        ~GridBuffers() { releaseAll(); }
    };
} // namespace sim
