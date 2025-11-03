#pragma once
#include "console.h"
#include "../../sim/parameters.h"

namespace console {
 // Phase1 refactor: precision mapping extracted from console.cpp
 // Map user Simulation::precision config + useMixedPrecision flags to final sim::SimPrecision.
 sim::SimPrecision MapPrecision(const RuntimeConsole::Simulation& simCfg);
}
