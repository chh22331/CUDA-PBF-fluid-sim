#include "config.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <charconv>

namespace {

// —— 轻量 JSON 读取工具（仅服务于本项目字段；非通用 JSON 解析器） ——

struct JsonDoc {
    std::string text;
    bool ok = false;
};

static JsonDoc ReadTextFile(const std::string& path, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    JsonDoc d{};
    if (!f) {
        if (err) *err = "config: failed to open file: " + path;
        return d;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    d.text = ss.str();
    d.ok = true;
    return d;
}

// 去除空白帮助
static inline bool isSpace(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
static void skipSpaces(const std::string& s, size_t& i) {
    while (i < s.size() && isSpace(s[i])) ++i;
}

// 在父节起点后查找 key（"key" 形式）；若 parentKey 为空，则全局搜索
static size_t findKey(const std::string& s, const std::string& key, size_t from = 0) {
    // 查找 "key"
    const std::string pat = "\"" + key + "\"";
    return s.find(pat, from);
}

// 定位对象节："section": { ... }，返回大括号内 [l,r) 范围起点
static bool findObjectSection(const std::string& s, const std::string& section, size_t& outBegin, size_t& outEnd) {
    size_t pos = findKey(s, section, 0);
    if (pos == std::string::npos) return false;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return false;
    // 跳到 '{'
    pos = s.find('{', pos);
    if (pos == std::string::npos) return false;
    size_t start = pos + 1;
    int depth = 1;
    for (size_t i = start; i < s.size(); ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') {
            --depth;
            if (depth == 0) {
                outBegin = start;
                outEnd = i; // 不包含 '}'
                return true;
            }
        }
    }
    return false;
}

// 在指定对象范围内查找标量 value（number/bool/string 简单形式）
template <typename T>
static bool parseNumberInObject(const std::string& s, size_t objBegin, size_t objEnd, const std::string& key, T& outVal) {
    size_t k = findKey(s, key, objBegin);
    if (k == std::string::npos || k >= objEnd) return false;
    size_t colon = s.find(':', k);
    if (colon == std::string::npos || colon >= objEnd) return false;
    size_t i = colon + 1; skipSpaces(s, i);
    // 支持整数/浮点
    size_t j = i;
    // 允许负号和小数/指数
    while (j < objEnd) {
        char c = s[j];
        if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') { ++j; continue; }
        break;
    }
    if (j == i) return false;
    // 利用 from_chars 解析为 double/float/int
    T val{};
    auto fcres = std::from_chars(s.data() + i, s.data() + j, val);
    if (fcres.ec != std::errc()) {
        // 对 float 使用 std::strtod 简易兜底
        if constexpr (std::is_same_v<T, float>) {
            char* endp = nullptr;
            val = static_cast<float>(std::strtod(s.c_str() + i, &endp));
            if (endp != s.c_str() + j) return false;
        } else if constexpr (std::is_same_v<T, double>) {
            char* endp = nullptr;
            val = std::strtod(s.c_str() + i, &endp);
            if (endp != s.c_str() + j) return false;
        } else {
            return false;
        }
    }
    outVal = val;
    return true;
}

static bool parseBoolInObject(const std::string& s, size_t objBegin, size_t objEnd, const std::string& key, bool& outVal) {
    size_t k = findKey(s, key, objBegin);
    if (k == std::string::npos || k >= objEnd) return false;
    size_t colon = s.find(':', k);
    if (colon == std::string::npos || colon >= objEnd) return false;
    size_t i = colon + 1; skipSpaces(s, i);
    if (i >= objEnd) return false;
    if (s.compare(i, 4, "true") == 0) { outVal = true; return true; }
    if (s.compare(i, 5, "false") == 0) { outVal = false; return true; }
    return false;
}

static bool parseStringInObject(const std::string& s, size_t objBegin, size_t objEnd, const std::string& key, std::string& outVal) {
    size_t k = findKey(s, key, objBegin);
    if (k == std::string::npos || k >= objEnd) return false;
    size_t colon = s.find(':', k);
    if (colon == std::string::npos || colon >= objEnd) return false;
    size_t i = s.find('"', colon);
    if (i == std::string::npos || i >= objEnd) return false;
    size_t j = s.find('"', i + 1);
    if (j == std::string::npos || j > objEnd) return false;
    outVal = s.substr(i + 1, j - (i + 1));
    return true;
}

static bool parseFloatArrayNInObject(const std::string& s, size_t objBegin, size_t objEnd, const std::string& key, float* outVals, int n) {
    size_t k = findKey(s, key, objBegin);
    if (k == std::string::npos || k >= objEnd) return false;
    size_t colon = s.find(':', k);
    if (colon == std::string::npos || colon >= objEnd) return false;
    size_t i = s.find('[', colon);
    if (i == std::string::npos || i >= objEnd) return false;
    size_t j = s.find(']', i + 1);
    if (j == std::string::npos || j > objEnd) return false;
    // 解析 n 个 float
    size_t p = i + 1;
    for (int idx = 0; idx < n; ++idx) {
        skipSpaces(s, p);
        // 读取一个数
        size_t q = p;
        while (q < j) {
            char c = s[q];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') { ++q; continue; }
            break;
        }
        if (q == p) return false;
        char* endp = nullptr;
        outVals[idx] = static_cast<float>(std::strtod(s.c_str() + p, &endp));
        p = s.find(',', q);
        if (p == std::string::npos || p > j) p = j; else ++p;
    }
    return true;
}

static bool parseStringArrayInObject(const std::string& s, size_t objBegin, size_t objEnd, const std::string& key, std::vector<std::string>& outVals) {
    size_t k = findKey(s, key, objBegin);
    if (k == std::string::npos || k >= objEnd) return false;
    size_t colon = s.find(':', k);
    if (colon == std::string::npos || colon >= objEnd) return false;
    size_t i = s.find('[', colon);
    if (i == std::string::npos || i >= objEnd) return false;
    size_t j = s.find(']', i + 1);
    if (j == std::string::npos || j > objEnd) return false;
    size_t p = i + 1;
    while (p < j) {
        size_t q1 = s.find('"', p);
        if (q1 == std::string::npos || q1 >= j) break;
        size_t q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > j) break;
        outVals.emplace_back(s.substr(q1 + 1, q2 - (q1 + 1)));
        p = s.find(',', q2);
        if (p == std::string::npos || p > j) break; else ++p;
    }
    return !outVals.empty();
}

// 从 profiles 中提取 profileName 对应的对象范围
static bool findProfileObject(const std::string& s, const std::string& profileName, size_t& outBegin, size_t& outEnd) {
    size_t profBegin, profEnd;
    if (!findObjectSection(s, "profiles", profBegin, profEnd)) return false;
    // 在 profiles 对象中查找 "profileName": { ... }
    size_t keyPos = findKey(s, profileName, profBegin);
    if (keyPos == std::string::npos || keyPos >= profEnd) return false;
    size_t colon = s.find(':', keyPos);
    if (colon == std::string::npos || colon >= profEnd) return false;
    size_t lb = s.find('{', colon);
    if (lb == std::string::npos || lb >= profEnd) return false;
    int depth = 1;
    size_t start = lb + 1;
    for (size_t i = start; i < profEnd; ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') {
            --depth;
            if (depth == 0) {
                outBegin = start;
                outEnd = i;
                return true;
            }
        }
    }
    return false;
}

static int mapSortFrequencyToN(const std::string& freq) {
    if (freq == "every_step") return 1;
    if (freq == "every_2_steps") return 2;
    // auto_threshold 暂用 1（后续加入阈值逻辑）
    return 1;
}

} // anon

namespace config {

    static void applySystemSection(const std::string& s, console::RuntimeConsole& io, std::vector<std::string>& whitelistOut) {
        size_t b, e;
        if (!findObjectSection(s, "system", b, e)) return;

        float res2[2]{};
        if (parseFloatArrayNInObject(s, b, e, "resolution", res2, 2)) {
            io.app.width = static_cast<uint32_t>((((1.0f) > (res2[0])) ? (1.0f) : (res2[0])));
            io.app.height = static_cast<uint32_t>((((1.0f) > (res2[1])) ? (1.0f) : (res2[1])));
        }
        bool vs = false;
        if (parseBoolInObject(s, b, e, "vsync", vs)) io.app.vsync = vs;

        std::string csv;
        if (parseStringInObject(s, b, e, "csv_stats_path", csv)) io.app.csv_path = csv;

        // hot_reload_whitelist
        std::vector<std::string> wl;
        if (parseStringArrayInObject(s, b, e, "hot_reload_whitelist", wl)) {
            whitelistOut = wl;
        }
    }

    static void applyViewerSection(const std::string& s, console::RuntimeConsole& io) {
        size_t b, e;
        if (!findObjectSection(s, "viewer", b, e)) return;
        bool en = true;
        if (parseBoolInObject(s, b, e, "enabled", en)) io.viewer.enabled = en;

        float ps = 0.f;
        if (parseNumberInObject<float>(s, b, e, "point_size_px", ps)) io.viewer.point_size_px = ps;

        float bg[4]{};
        if (parseFloatArrayNInObject(s, b, e, "background_color", bg, 3)) {
            io.viewer.background_color[0] = bg[0];
            io.viewer.background_color[1] = bg[1];
            io.viewer.background_color[2] = bg[2];
            io.viewer.background_color[3] = 1.0f;
        }
    }

    static void applySimulationSection(const std::string& s, console::RuntimeConsole& io) {
        size_t b, e;
        if (!findObjectSection(s, "simulation", b, e)) return;

        float cfl = 0.f;
        if (parseNumberInObject<float>(s, b, e, "cfl", cfl)) io.sim.cfl = cfl;

        float rho0 = 0.f;
        if (parseNumberInObject<float>(s, b, e, "rest_density", rho0)) io.sim.restDensity = rho0;

        float h = 0.f;
        if (parseNumberInObject<float>(s, b, e, "kernel_radius", h)) {
            io.sim.smoothingRadius = h;
            io.sim.cellSize = h;
        }

        int K = 0;
        if (parseNumberInObject<int>(s, b, e, "solver_iterations", K)) io.sim.solverIters = K;

        int cap = 0;
        if (parseNumberInObject<int>(s, b, e, "max_neighbors", cap)) io.sim.maxNeighbors = cap;
    }

    static void applyPerformanceSection(const std::string& s, console::RuntimeConsole& io) {
        size_t b, e;
        if (!findObjectSection(s, "performance", b, e)) return;

        float mul = 0.f;
        if (parseNumberInObject<float>(s, b, e, "grid_cell_size_multiplier", mul)) {
            io.perf.grid_cell_size_multiplier = mul;
        }

        std::string sf;
        if (parseStringInObject(s, b, e, "sort_frequency", sf)) {
            io.sim.sortEveryN = mapSortFrequencyToN(sf);
        }

        int nc = 0;
        if (parseNumberInObject<int>(s, b, e, "neighbor_cap", nc)) {
            io.perf.neighbor_cap = nc;
            io.sim.maxNeighbors = (((io.sim.maxNeighbors) < (nc)) ? (io.sim.maxNeighbors) : (nc));
        }

        int tbs = 0;
        if (parseNumberInObject<int>(s, b, e, "launch_bounds_tbs", tbs)) io.perf.launch_bounds_tbs = tbs;
        int mbs = 0;
        if (parseNumberInObject<int>(s, b, e, "min_blocks_per_sm", mbs)) io.perf.min_blocks_per_sm = mbs;

        bool graphs = true;
        if (parseBoolInObject(s, b, e, "use_cuda_graphs", graphs)) io.perf.use_cuda_graphs = graphs;
    }

    static void applyProfileOverrides(const std::string& s, const std::string& profile, console::RuntimeConsole& io) {
        if (profile.empty()) return;
        size_t pb, pe;
        if (!findProfileObject(s, profile, pb, pe)) return;

        // profiles.<name>.simulation.solver_iterations/max_neighbors
        size_t sb, se;
        if (findObjectSection(s.substr(pb, pe - pb), "simulation", sb, se)) {
            sb += pb; se += pb;
            int K = 0;
            if (parseNumberInObject<int>(s, sb, se, "solver_iterations", K)) io.sim.solverIters = K;
            int cap = 0;
            if (parseNumberInObject<int>(s, sb, se, "max_neighbors", cap)) io.sim.maxNeighbors = cap;
        }

        // profiles.<name>.performance.neighbor_cap/mixed_precision
        if (findObjectSection(s.substr(pb, pe - pb), "performance", sb, se)) {
            sb += pb; se += pb;
            int nc = 0;
            if (parseNumberInObject<int>(s, sb, se, "neighbor_cap", nc)) {
                io.perf.neighbor_cap = nc;
                io.sim.maxNeighbors = (((io.sim.maxNeighbors) < (nc)) ? (io.sim.maxNeighbors) : (nc));
            }
            // mixed_precision 仅占位（当前不改变 sim.useMixedPrecision 的枚举）
            std::string mp;
            if (parseStringInObject(s, sb, se, "mixed_precision", mp)) {
                // 可根据 mp=off|fp16_mid|fp16_more 映射为 bool/枚举；当前保持占位
                io.sim.useMixedPrecision = (mp != "off");
            }
        }
    }

    static void applyProfileName(const std::string& s, std::string& outProfile) {
        // 顶层 "profile": "balanced"
        size_t b=0, e= s.size();
        std::string prof;
        if (parseStringInObject(s, b, e, "profile", prof)) {
            outProfile = prof;
        }
    }

    static bool shouldApplyField(const std::vector<std::string>& whitelist, const std::string& dottedKey) {
        if (whitelist.empty()) return false;
        return std::find(whitelist.begin(), whitelist.end(), dottedKey) != whitelist.end();
    }

    // 局部热加载：仅应用白名单字段（最小实现）
    static void applyHotReloadWhitelisted(const std::string& s, const std::vector<std::string>& whitelist, console::RuntimeConsole& io, std::string& activeProfile) {
        size_t bSim, eSim, bPerf, ePerf, bView, eView;
        if (findObjectSection(s, "simulation", bSim, eSim)) {
            if (shouldApplyField(whitelist, "simulation.solver_iterations")) {
                int K = 0; if (parseNumberInObject<int>(s, bSim, eSim, "solver_iterations", K)) io.sim.solverIters = K;
            }
            if (shouldApplyField(whitelist, "simulation.max_neighbors")) {
                int cap = 0; if (parseNumberInObject<int>(s, bSim, eSim, "max_neighbors", cap)) io.sim.maxNeighbors = cap;
            }
        }
        if (findObjectSection(s, "performance", bPerf, ePerf)) {
            if (shouldApplyField(whitelist, "performance.sort_frequency")) {
                std::string sf; if (parseStringInObject(s, bPerf, ePerf, "sort_frequency", sf)) io.sim.sortEveryN = mapSortFrequencyToN(sf);
            }
            if (shouldApplyField(whitelist, "performance.neighbor_cap")) {
                int nc=0; if (parseNumberInObject<int>(s, bPerf, ePerf, "neighbor_cap", nc)) {
                    io.perf.neighbor_cap = nc;
                    io.sim.maxNeighbors = (((io.sim.maxNeighbors) < (nc)) ? (io.sim.maxNeighbors) : (nc));
                }
            }
        }
        if (findObjectSection(s, "viewer", bView, eView)) {
            if (shouldApplyField(whitelist, "viewer.point_size_px")) {
                float ps=0.f; if (parseNumberInObject<float>(s, bView, eView, "point_size_px", ps)) io.viewer.point_size_px = ps;
            }
            // viewer.color_mode 暂未落地到渲染器，先忽略
        }
        if (shouldApplyField(whitelist, "profile")) {
            std::string prof; applyProfileName(s, prof);
            if (!prof.empty() && prof != activeProfile) {
                activeProfile = prof;
                applyProfileOverrides(s, activeProfile, io);
            }
        }
    }

    bool LoadFile(const std::string& path, console::RuntimeConsole& io, State* state, std::string* errOut) {
        std::string err;
        auto doc = ReadTextFile(path, &err);
        if (!doc.ok) {
            if (errOut) *errOut = err;
            if (state) state->lastError = err;
            return false;
        }

        // 基本节应用
        std::vector<std::string> wl; // 从文件读取的白名单
        applySystemSection(doc.text, io, wl);
        applyViewerSection(doc.text, io);
        applySimulationSection(doc.text, io);
        applyPerformanceSection(doc.text, io);

        // profile 名与覆盖
        std::string profName;
        applyProfileName(doc.text, profName);
        if (!profName.empty()) {
            applyProfileOverrides(doc.text, profName, io);
        }

        // 维护 state
        if (state) {
            state->path = path;
            std::error_code ec;
            state->lastWrite = std::filesystem::last_write_time(path, ec);
            if (ec) { /* ignore */ }
            state->hotReloadWhitelist = wl.empty() ? DefaultWhitelist() : wl;
            state->activeProfile = profName;
            state->lastError.clear();
        }
        if (errOut) errOut->clear();
        return true;
    }

    bool TryHotReload(State& state, console::RuntimeConsole& io, std::string* errOut) {
        std::error_code ec;
        auto cur = std::filesystem::last_write_time(state.path, ec);
        if (ec) {
            std::string err = "config: last_write_time failed: " + state.path;
            if (errOut) *errOut = err;
            state.lastError = err;
            return false;
        }
        if (state.lastWrite == std::filesystem::file_time_type{}) {
            state.lastWrite = cur;
            return false;
        }
        if (cur <= state.lastWrite) return false;

        // 文件已更新 → 读取文本并应用白名单字段
        std::string err;
        auto doc = ReadTextFile(state.path, &err);
        if (!doc.ok) {
            if (errOut) *errOut = err;
            state.lastError = err;
            return false;
        }

        // 应用热加载白名单字段
        const auto& wl = state.hotReloadWhitelist.empty() ? DefaultWhitelist() : state.hotReloadWhitelist;
        applyHotReloadWhitelisted(doc.text, wl, io, state.activeProfile);

        // 更新时间戳
        state.lastWrite = cur;
        state.lastError.clear();
        if (errOut) errOut->clear();
        return true;
    }

} // namespace config
