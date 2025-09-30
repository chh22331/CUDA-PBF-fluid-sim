#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

namespace core {

struct StatRow { std::string name; double ms = 0.0; };

class CpuTimer {
public:
    void begin() { t0 = Clock::now(); }
    double endMs() const { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
};

class Profiler {
public:
    void beginFrame(uint64_t frame) { (void)frame; rows.clear(); }
    void endFrame() {}
    void addRow(const std::string& name, double ms) { rows.push_back({name, ms}); }
    void flushCsv(const std::string& path, uint64_t frameIndex) {
        std::ofstream f(path, frameIndex==0 ? std::ios::trunc : std::ios::app);
        if (!f) return;
        if (frameIndex==0) {
            f << "frame";
            for (auto& r: rows) f << "," << r.name;
            f << "\n";
        }
        f << frameIndex;
        for (auto& r: rows) f << "," << r.ms;
        f << "\n";
    }
private:
    std::vector<StatRow> rows;
};

}
