#pragma once
#include <cstdint>

namespace sim {
struct Parameters { int solver_iterations = 1; };

class Simulator {
public:
    bool initialize(const Parameters&) { return true; }
    void step(float /*dt*/) {}
};
}
