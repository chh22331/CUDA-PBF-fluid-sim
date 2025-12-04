#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>

namespace core {

struct StatRow { std::string name; double ms = 0.0; };
// Text fields such as profile/device names.
struct TextRow { std::string name; std::string value; };

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
    void beginFrame(uint64_t frame) { (void)frame; rows.clear(); texts.clear(); }
    void endFrame() {}
    void addRow(const std::string& name, double ms) { rows.push_back({name, ms}); }
    void addCounter(const std::string& name, int64_t v) { rows.push_back({name, static_cast<double>(v)}); }
    void addText(const std::string& name, const std::string& v) { texts.push_back({name, v}); }

    void flushCsv(const std::string& path, uint64_t frameIndex) {
        std::ofstream f(path, frameIndex == 0 ? std::ios::trunc : std::ios::app);
        if (!f) return;

        if (frameIndex == 0) {
            f << "frame";
            // Numeric column.
            for (auto& r : rows) f << "," << r.name;
            // Text column.
            for (auto& t : texts) f << "," << t.name;
            f << "\n";
        }

        f << frameIndex;
        // Numeric column.
        for (auto& r : rows) f << "," << r.ms;
        // Text column (quoted with doubled inner quotes).
        for (auto& t : texts) {
            std::string v = t.value;
            std::replace(v.begin(), v.end(), '\n', ' ');
            // Basic CSV escaping: double quotes become doubled.
            std::string out; out.reserve(v.size() + 2);
            out.push_back('"');
            for (char c : v) {
                if (c == '"') out.push_back('"');
                out.push_back(c);
            }
            out.push_back('"');
            f << "," << out;
        }
        f << "\n";
    }

private:
    std::vector<StatRow> rows;
    std::vector<TextRow> texts;
};

} // namespace core
