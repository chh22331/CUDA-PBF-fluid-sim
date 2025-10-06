#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include "../core/console.h"

namespace config {

    // 热加载状态与白名单
    struct State {
        std::string path = "config.json";
        std::filesystem::file_time_type lastWrite{};
        // 配置中声明的白名单，若为空则使用默认白名单
        std::vector<std::string> hotReloadWhitelist;
        // 当前生效 profile 名（若配置存在）
        std::string activeProfile;
        // 简单错误累计文本（可用于 UI 展示）
        std::string lastError;
        // 仅在文件缺失时首轮提示一次，后续静默
        bool missingWarnedOnce = false;
    };

    // 从 JSON 文件加载到 RuntimeConsole（应用 profile 覆盖；若 JSON 缺失字段，则保持现有值）
    // 返回 true 表示加载成功（即使部分字段解析失败）；错误详情写入 state.lastError/errOut
    bool LoadFile(const std::string& path, console::RuntimeConsole& io, State* state = nullptr, std::string* errOut = nullptr);

    // 基于 mtime 的热加载：当检测到配置文件更新时，按白名单对 io 进行增量更新
    // 返回 true 表示执行过热加载（不代表字段都有变化）；false 表示未变化或失败
    bool TryHotReload(State& state, console::RuntimeConsole& io, std::string* errOut = nullptr);

    // 默认白名单（需求文档）
    inline const std::vector<std::string>& DefaultWhitelist() {
        static const std::vector<std::string> k = {
            "simulation.solver_iterations",
            "simulation.max_neighbors",
            "performance.sort_frequency",
            "performance.neighbor_cap",
            "viewer.point_size_px",
            "viewer.color_mode",
            "profile"
        };
        return k;
    }

} // namespace config