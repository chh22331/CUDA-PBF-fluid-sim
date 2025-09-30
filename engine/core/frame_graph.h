#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

namespace core {
struct ResourceDesc { std::string name; };
using ResourceId = uint32_t;

struct PassDesc {
    std::string name;
    std::vector<ResourceId> reads;
    std::vector<ResourceId> writes;
    std::function<void()> execute;
};

class FrameGraph {
public:
    ResourceId addResource(const ResourceDesc& r) { resources.push_back(r); return (ResourceId)(resources.size()-1); }
    void addPass(const PassDesc& p) { passes.push_back(p); }
    void compile() {}
    template<typename AddStatFn>
    void execute(AddStatFn addStat) {
        for (auto& p: passes) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (p.execute) p.execute();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            addStat(p.name, ms);
        }
    }
private:
    std::vector<ResourceDesc> resources;
    std::vector<PassDesc> passes;
};
}
