#pragma once
#include <cstdint>
#include <string>

struct AppConfig {
    // Minimal M0 configuration; extend later.
    uint32_t width = 1280;
    uint32_t height = 720;
    bool use_compute_thickness = false;
    std::string csv_path = "stats.csv";
};

namespace config {
    AppConfig LoadDefault();
}
