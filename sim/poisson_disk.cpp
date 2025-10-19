#include "poisson_disk.h"
#include <algorithm>
#include <cmath>

namespace sim {
    static inline float2 sample_annulus(float rMin, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float R0 = rMin;
        const float R1 = 2.0f * rMin;
        const float u = U(rng);
        const float r = std::sqrt(R0 * R0 + u * (R1 * R1 - R0 * R0));
        const float th = 6.28318530718f * U(rng);
        return make_float2(r * std::cos(th), r * std::sin(th));
    }
    static inline float2 sample_in_disk(float R, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float r = R * std::sqrt(U(rng));
        const float th = 6.28318530718f * U(rng);
        return make_float2(r * std::cos(th), r * std::sin(th));
    }

    void poisson_disk_in_circle(float R, float rMin, int target,
                                std::mt19937& rng,
                                std::vector<float2>& outPts) {
        outPts.clear();
        if (R <= 0.0f || rMin <= 0.0f || target <= 0) return;
        const float cell = rMin / 1.41421356237f;
        const float side = 2.0f * R;
        const int gw = std::max(1, (int)std::ceil(side / cell));
        const int gh = gw;
        const int nCells = gw * gh;
        std::vector<int> grid(nCells, -1);
        std::vector<int> active;
        active.reserve(target);
        outPts.reserve(target);
        auto toGrid = [&](const float2& p, int& gx, int& gy) {
            const float x = p.x + R;
            const float y = p.y + R;
            gx = std::clamp((int)std::floor(x / cell), 0, gw - 1);
            gy = std::clamp((int)std::floor(y / cell), 0, gh - 1);
        };
        auto gridIdx = [&](int gx, int gy) { return gy * gw + gx; };
        {
            float2 p0 = sample_in_disk(R, rng);
            int gx, gy; toGrid(p0, gx, gy);
            grid[gridIdx(gx, gy)] = 0;
            outPts.push_back(p0);
            active.push_back(0);
        }
        std::uniform_int_distribution<int> pick;
        const int kCandidates = 30;
        const float rMin2 = rMin * rMin;
        while (!active.empty() && (int)outPts.size() < target) {
            pick = std::uniform_int_distribution<int>(0, (int)active.size() - 1);
            const int aidx = pick(rng);
            const int pIndex = active[aidx];
            const float2 p = outPts[pIndex];
            bool found = false;
            for (int k = 0; k < kCandidates && (int)outPts.size() < target; ++k) {
                const float2 off = sample_annulus(rMin, rng);
                const float2 q = make_float2(p.x + off.x, p.y + off.y);
                if ((q.x * q.x + q.y * q.y) > R * R) continue;
                int gx, gy; toGrid(q, gx, gy);
                bool ok = true;
                for (int yy = std::max(0, gy - 2); yy <= std::min(gh - 1, gy + 2) && ok; ++yy) {
                    for (int xx = std::max(0, gx - 2); xx <= std::min(gw - 1, gx + 2) && ok; ++xx) {
                        const int gi = grid[gridIdx(xx, yy)];
                        if (gi < 0) continue;
                        const float2 other = outPts[gi];
                        const float dx = other.x - q.x;
                        const float dy = other.y - q.y;
                        if (dx * dx + dy * dy < rMin2) ok = false;
                    }
                }
                if (!ok) continue;
                const int newIndex = (int)outPts.size();
                outPts.push_back(q);
                active.push_back(newIndex);
                grid[gridIdx(gx, gy)] = newIndex;
                found = true;
            }
            if (!found) {
                active[aidx] = active.back();
                active.pop_back();
            }
        }
    }
} // namespace sim
