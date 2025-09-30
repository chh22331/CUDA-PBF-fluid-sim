#pragma once
#include <cstdint>

namespace interop {
struct InteropHandles {
    void* sharedParticleBuffer = nullptr; // placeholder for HANDLE
    void* sharedSimDoneFence = nullptr;   // placeholder for HANDLE
};

class CudaD3D12Interop {
public:
    bool import(const InteropHandles&) { return true; }
    void* particlesDevicePtr() const { return nullptr; }
    void waitGfx() {}
    void signalSimDone() {}
};
}
