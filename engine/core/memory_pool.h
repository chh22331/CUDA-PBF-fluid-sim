#pragma once
#include <cstdint>
#include <vector>

namespace core {
struct GpuAllocation { void* ptr = nullptr; uint64_t size = 0; uint64_t offset = 0; };

class MemoryPool {
public:
    bool initialize() { return true; }
    void shutdown() {}
    GpuAllocation allocate(uint64_t size, uint64_t alignment) { (void)alignment; return {nullptr, size, 0}; }
};
}
