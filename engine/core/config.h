#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include "../core/console.h"

namespace config {

    // Hot-reload state and whitelist.
    struct State {
        std::string path = "config.json";
        std::filesystem::file_time_type lastWrite{};
        // Whitelist declared in config (fallback to defaults when empty).
        std::vector<std::string> hotReloadWhitelist;
        // Name of the active profile if provided.
        std::string activeProfile;
        // Aggregated error text for UI display.
        std::string lastError;
        // Emit the missing-file warning only once.
        bool missingWarnedOnce = false;
    };

    // Load RuntimeConsole from JSON (profile overrides apply; missing fields keep their values).
    // Returns true on load success even if some fields fail; errors are written to state.lastError/errOut.
    bool LoadFile(const std::string& path, console::RuntimeConsole& io, State* state = nullptr, std::string* errOut = nullptr);

    // mtime-driven hot reload that applies whitelisted updates incrementally.
    // Returns true when a hot reload occurred (fields may still be unchanged).
    bool TryHotReload(State& state, console::RuntimeConsole& io, std::string* errOut = nullptr);

    // Default whitelist derived from the requirements doc.
    inline const std::vector<std::string>& DefaultWhitelist() {
        static const std::vector<std::string> k = {
            "simulation.solver_iterations",
            "simulation.max_neighbors",
            "performance.sort_frequency",
            "performance.neighbor_cap",
            "viewer.point_size_px",
            "viewer.color_mode",
            "profile",
            "system.frame_cap_enabled",
            "system.frame_cap_fps"
        };
        return k;
    }

} // namespace config
