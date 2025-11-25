#pragma once
#include <unordered_map>
#include <cstring>
#include "parameters.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "phase_pipeline.h"

namespace sim {

struct KernelVariant { bool halfPos=false; bool halfVel=false; bool compact=false; };

// unified launch signature placeholder; real implementation will dispatch to existing extern "C" kernels.
using KernelLaunchFn = void(*)(DeviceBuffers&, GridBuffers&, const SimParams&, cudaStream_t);

struct KernelKey {
    PhaseType phase; bool halfPos; bool halfVel; bool compact;
    bool operator==(const KernelKey& o) const { return phase==o.phase && halfPos==o.halfPos && halfVel==o.halfVel && compact==o.compact; }
};

struct KernelKeyHash { size_t operator()(const KernelKey& k) const { return ((size_t)k.phase*131u) ^ (k.halfPos?0x1u:0) ^ (k.halfVel?0x2u:0) ^ (k.compact?0x4u:0); } };

class KernelDispatcher {
public:
    void registerKernel(PhaseType ph, bool halfPos, bool halfVel, bool compact, KernelLaunchFn fn){ m_map[{ph,halfPos,halfVel,compact}] = fn; }
    KernelLaunchFn find(const KernelVariant& v, PhaseType ph) const {
        KernelKey key{ph,v.halfPos,v.halfVel,v.compact};
        auto it=m_map.find(key); if(it!=m_map.end()) return it->second;
        // fallback: ignore halfVel, then halfPos
        KernelKey k2{ph,v.halfPos,false,v.compact}; it=m_map.find(k2); if(it!=m_map.end()) return it->second;
        KernelKey k3{ph,false,false,v.compact}; it=m_map.find(k3); if(it!=m_map.end()) return it->second;
        return nullptr;
    }
private:
    std::unordered_map<KernelKey,KernelLaunchFn,KernelKeyHash> m_map;
};

} // namespace sim
