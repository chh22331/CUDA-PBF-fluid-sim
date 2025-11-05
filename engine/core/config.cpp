#include "config.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <system_error>
#include <unordered_map>
#include <cstdio>

#if __has_include(<nlohmann/json.hpp>)
    #include <nlohmann/json.hpp>
    using json = nlohmann::json;
    #define PBFX_HAVE_JSON 1
#else
    #define PBFX_HAVE_JSON 0
#endif

namespace fs = std::filesystem;

namespace {

    static inline std::string slurpFile(const std::string& path, std::error_code& ec) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) { ec = std::make_error_code(std::errc::no_such_file_or_directory); return {}; }
        std::ostringstream oss; oss << ifs.rdbuf();
        ec.clear();
        return oss.str();
    }

    static inline bool fileTime(const std::string& path, fs::file_time_type& out, std::error_code& ec) {
        ec.clear();
        if (!fs::exists(path, ec)) return false;
        out = fs::last_write_time(path, ec);
        return !ec;
    }

    static void ApplyWhitelistDiff(const std::vector<std::string>& whitelist,
                                   const console::RuntimeConsole& tmp,
                                   console::RuntimeConsole& io,
                                   config::State& st) {
        for (const auto& k : whitelist) {
            if (k == "simulation.solver_iterations") {
                io.sim.solverIters = tmp.sim.solverIters;
            } else if (k == "simulation.max_neighbors") {
                io.sim.maxNeighbors = tmp.sim.maxNeighbors;
            } else if (k == "performance.neighbor_cap") {
                io.perf.neighbor_cap = tmp.perf.neighbor_cap;
            } else if (k == "viewer.point_size_px") {
                io.viewer.point_size_px = tmp.viewer.point_size_px;
            } else if (k == "performance.sort_frequency") {
                // 保留占位（未实现）
            } else if (k == "viewer.color_mode") {
                // 未实现
            } else if (k == "profile") {
            }
        }
    }

#if !PBFX_HAVE_JSON
    static bool parseBool(const std::string& s, size_t& pos, bool& out) {
        while (pos < s.size() && isspace((unsigned char)s[pos])) ++pos;
        if (s.compare(pos, 4, "true") == 0) { out = true; pos += 4; return true; }
        if (s.compare(pos, 5, "false") == 0) { out = false; pos += 5; return true; }
        return false;
    }
    static bool parseNumber(const std::string& s, size_t& pos, double& out) {
        while (pos < s.size() && isspace((unsigned char)s[pos])) ++pos;
        size_t start = pos;
        bool seen = false;
        while (pos < s.size()) {
            char c = s[pos];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') { ++pos; seen = true; }
            else break;
        }
        if (!seen) return false;
        try {
            out = std::stod(s.substr(start, pos - start));
            return true;
        } catch (...) { return false; }
    }
    static bool parseString(const std::string& s, size_t& pos, std::string& out) {
        while (pos < s.size() && isspace((unsigned char)s[pos])) ++pos;
        if (pos >= s.size() || s[pos] != '"') return false;
        ++pos;
        std::string res;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '\\') {
                if (pos < s.size()) {
                    char esc = s[pos++];
                    res.push_back(esc);
                }
            } else if (c == '"') {
                out = std::move(res);
                return true;
            } else {
                res.push_back(c);
            }
        }
        return false;
    }
    static bool findObject(const std::string& s, const std::string& key, size_t from, size_t& objBegin, size_t& objEnd) {
        size_t kpos = s.find("\"" + key + "\"", from);
        if (kpos == std::string::npos) return false;
        size_t colon = s.find(':', kpos);
        if (colon == std::string::npos) return false;
        size_t brace = s.find('{', colon);
        if (brace == std::string::npos) return false;
        int depth = 0; size_t i = brace;
        for (; i < s.size(); ++i) {
            if (s[i] == '{') ++depth;
            else if (s[i] == '}') { --depth; if (depth == 0) { objBegin = brace + 1; objEnd = i; return true; } }
        }
        return false;
    }
    static bool findArray(const std::string& s, const std::string& key, size_t from, size_t& arrBegin, size_t& arrEnd) {
        size_t kpos = s.find("\"" + key + "\"", from);
        if (kpos == std::string::npos) return false;
        size_t colon = s.find(':', kpos);
        if (colon == std::string::npos) return false;
        size_t lb = s.find('[', colon);
        if (lb == std::string::npos) return false;
        int depth = 0; size_t i = lb;
        for (; i < s.size(); ++i) {
            if (s[i] == '[') ++depth;
            else if (s[i] == ']') { --depth; if (depth == 0) { arrBegin = lb + 1; arrEnd = i; return true; } }
        }
        return false;
    }
    static bool findNumberInObject(const std::string& obj, const std::string& key, double& out) {
        size_t kpos = obj.find("\"" + key + "\"");
        if (kpos == std::string::npos) return false;
        size_t colon = obj.find(':', kpos);
        if (colon == std::string::npos) return false;
        size_t p = colon + 1;
        return parseNumber(obj, p, out);
    }
    static bool findBoolInObject(const std::string& obj, const std::string& key, bool& out) {
        size_t kpos = obj.find("\"" + key + "\"");
        if (kpos == std::string::npos) return false;
        size_t colon = obj.find(':', kpos);
        if (colon == std::string::npos) return false;
        size_t p = colon + 1;
        return parseBool(obj, p, out);
    }
    static bool findStringInObject(const std::string& obj, const std::string& key, std::string& out) {
        size_t kpos = obj.find("\"" + key + "\"");
        if (kpos == std::string::npos) return false;
        size_t colon = obj.find(':', kpos);
        if (colon == std::string::npos) return false;
        size_t p = colon + 1;
        return parseString(obj, p, out);
    }
#endif

} // namespace

namespace config {

    static void MergeIntoConsole_Safe(console::RuntimeConsole& cc,
#if PBFX_HAVE_JSON
        const json& j,
        State* state
#else
        const std::string& raw,
        State* state
#endif
    ) {
#if PBFX_HAVE_JSON
        if (j.contains("system") && j["system"].is_object()) {
            const auto& s = j["system"];
            if (s.contains("resolution") && s["resolution"].is_array() && s["resolution"].size() >= 2) {
                cc.app.width = s["resolution"][0].get<uint32_t>();
                cc.app.height = s["resolution"][1].get<uint32_t>();
            }
            if (s.contains("vsync")) cc.app.vsync = s["vsync"].get<bool>();
            if (s.contains("csv_stats_path")) cc.app.csv_path = s["csv_stats_path"].get<std::string>();
        }
        if (j.contains("viewer") && j["viewer"].is_object()) {
            const auto& v = j["viewer"];
            if (v.contains("enabled")) cc.viewer.enabled = v["enabled"].get<bool>();
            if (v.contains("point_size_px")) cc.viewer.point_size_px = v["point_size_px"].get<float>();
            if (v.contains("fixed_color") && v["fixed_color"].is_array() && v["fixed_color"].size() >= 3) {
                cc.viewer.fixed_color[0] = v["fixed_color"][0].get<float>();
                cc.viewer.fixed_color[1] = v["fixed_color"][1].get<float>();
                cc.viewer.fixed_color[2] = v["fixed_color"][2].get<float>();
            }
            if (v.contains("background_color") && v["background_color"].is_array() && v["background_color"].size() >= 3) {
                cc.viewer.background_color[0] = v["background_color"][0].get<float>();
                cc.viewer.background_color[1] = v["background_color"][1].get<float>();
                cc.viewer.background_color[2] = v["background_color"][2].get<float>();
            }
        }
        if (j.contains("performance") && j["performance"].is_object()) {
            const auto& p = j["performance"];
            if (p.contains("grid_cell_size_multiplier")) cc.perf.grid_cell_size_multiplier = p["grid_cell_size_multiplier"].get<float>();
            if (p.contains("neighbor_cap")) cc.perf.neighbor_cap = p["neighbor_cap"].get<int>();
            if (p.contains("launch_bounds_tbs")) cc.perf.launch_bounds_tbs = p["launch_bounds_tbs"].get<int>();
            if (p.contains("min_blocks_per_sm")) cc.perf.min_blocks_per_sm = p["min_blocks_per_sm"].get<int>();
            // use_cuda_graphs 已移除：忽略输入配置
            if (p.contains("use_hashed_grid")) cc.perf.use_hashed_grid = p["use_hashed_grid"].get<bool>();
            if (p.contains("sort_compact_every_n")) cc.perf.sort_compact_every_n = p["sort_compact_every_n"].get<int>();
            if (p.contains("compact_binary_search")) cc.perf.compact_binary_search = p["compact_binary_search"].get<bool>();
            if (p.contains("log_grid_compact_stats")) cc.perf.log_grid_compact_stats = p["log_grid_compact_stats"].get<bool>();
        }
        if (j.contains("simulation") && j["simulation"].is_object()) {
            const auto& s = j["simulation"];
            if (s.contains("cfl")) cc.sim.cfl = s["cfl"].get<float>();
            if (s.contains("rest_density")) cc.sim.restDensity = s["rest_density"].get<float>();
            if (s.contains("kernel_radius")) cc.sim.smoothingRadius = s["kernel_radius"].get<float>();
            if (s.contains("solver_iterations")) cc.sim.solverIters = s["solver_iterations"].get<int>();
            if (s.contains("max_neighbors")) cc.sim.maxNeighbors = s["max_neighbors"].get<int>();
            if (s.contains("xsph_c")) cc.sim.xsph_c = s["xsph_c"].get<float>();
        }
        if (j.contains("profile") && j["profile"].is_string()) {
            if (state) state->activeProfile = j["profile"].get<std::string>();
        }
        if (state && !state->activeProfile.empty() && j.contains("profiles") && j["profiles"].is_object()) {
            const auto& pr = j["profiles"];
            auto it = pr.find(state->activeProfile);
            if (it != pr.end() && it->is_object()) {
                const auto& P = *it;
                if (P.contains("simulation") && P["simulation"].is_object()) {
                    const auto& ps = P["simulation"];
                    if (ps.contains("solver_iterations")) cc.sim.solverIters = ps["solver_iterations"].get<int>();
                    if (ps.contains("max_neighbors")) cc.sim.maxNeighbors = ps["max_neighbors"].get<int>();
                }
                if (P.contains("performance") && P["performance"].is_object()) {
                    const auto& pp = P["performance"];
                    if (pp.contains("neighbor_cap")) cc.perf.neighbor_cap = pp["neighbor_cap"].get<int>();
                }
            }
        }
#else
        size_t objB = 0, objE = 0;
        if (findObject(raw, "system", 0, objB, objE)) {
            std::string sys = raw.substr(objB, objE - objB);
            size_t ab = 0, ae = 0;
            if (findArray(sys, "resolution", 0, ab, ae)) {
                std::string arr = sys.substr(ab, ae - ab);
                size_t p = 0; double a = 0, b = 0;
                if (parseNumber(arr, p, a)) {
                    while (p < arr.size() && arr[p] != ',') ++p;
                    if (p < arr.size()) ++p;
                    if (parseNumber(arr, p, b)) {
                        if (a > 0 && b > 0) { cc.app.width = (uint32_t)a; cc.app.height = (uint32_t)b; }
                    }
                }
            }
            bool vs = false;
            if (findBoolInObject(sys, "vsync", vs)) cc.app.vsync = vs;
            std::string csv;
            if (findStringInObject(sys, "csv_stats_path", csv)) cc.app.csv_path = csv;
        }
        if (findObject(raw, "viewer", 0, objB, objE)) {
            std::string vw = raw.substr(objB, objE - objB);
            bool en = false; if (findBoolInObject(vw, "enabled", en)) cc.viewer.enabled = en;
            double ps = 0.0; if (findNumberInObject(vw, "point_size_px", ps)) cc.viewer.point_size_px = float(ps);
        }
        if (findObject(raw, "performance", 0, objB, objE)) {
            std::string pf = raw.substr(objB, objE - objB);
            double gmul = 0.0; if (findNumberInObject(pf, "grid_cell_size_multiplier", gmul)) cc.perf.grid_cell_size_multiplier = float(gmul);
            double ncap = 0.0; if (findNumberInObject(pf, "neighbor_cap", ncap)) cc.perf.neighbor_cap = int(ncap);
            bool hashed = false; if (findBoolInObject(pf, "use_hashed_grid", hashed)) cc.perf.use_hashed_grid = hashed;
            double every = 0.0; if (findNumberInObject(pf, "sort_compact_every_n", every)) cc.perf.sort_compact_every_n = int(every);
            // use_cuda_graphs 已移除，不再解析
        }
        if (findObject(raw, "simulation", 0, objB, objE)) {
            std::string sm = raw.substr(objB, objE - objB);
            double cfl = 0.0; if (findNumberInObject(sm, "cfl", cfl)) cc.sim.cfl = float(cfl);
            double rd = 0.0;  if (findNumberInObject(sm, "rest_density", rd)) cc.sim.restDensity = float(rd);
            double h  = 0.0;  if (findNumberInObject(sm, "kernel_radius", h)) cc.sim.smoothingRadius = float(h);
            double K  = 0.0;  if (findNumberInObject(sm, "solver_iterations", K)) cc.sim.solverIters = int(K);
            double MN = 0.0;  if (findNumberInObject(sm, "max_neighbors", MN)) cc.sim.maxNeighbors = int(MN);
            double X  = 0.0;  if (findNumberInObject(sm, "xsph_c", X)) cc.sim.xsph_c = float(X);
        }
        {
            size_t p = raw.find("\"profile\"");
            if (p != std::string::npos) {
                size_t colon = raw.find(':', p);
                if (colon != std::string::npos) {
                    std::string v; size_t q = colon + 1;
                    if (parseString(raw, q, v)) {
                        if (state) state->activeProfile = v;
                    }
                }
            }
        }
#endif
        (void)state;
    }

    static const std::vector<std::string>& ResolveWhitelist(const State* state) {
        if (state && !state->hotReloadWhitelist.empty()) return state->hotReloadWhitelist;
        return DefaultWhitelist();
    }

    bool LoadFile(const std::string& path, console::RuntimeConsole& io, State* state, std::string* errOut) {
        if (state) state->path = path;
        std::error_code ec;
        std::string raw = slurpFile(path, ec);
        if (ec || raw.empty()) {
            if (errOut) {
                if (ec) *errOut = "config: open failed: " + ec.message();
                else     *errOut = "config: empty or unreadable file";
            }
            if (state) {
                if (!state->missingWarnedOnce) {
                    state->lastError = (errOut ? *errOut : std::string("config missing."));
                }
                state->missingWarnedOnce = true;
            }
            return false;
        }

        if (state) {
            std::error_code ec2;
            fs::file_time_type ft;
            if (fileTime(path, ft, ec2)) state->lastWrite = ft;
        }

#if PBFX_HAVE_JSON
        try {
            json j = json::parse(raw);
            MergeIntoConsole_Safe(io, j, state);
        } catch (const std::exception& e) {
            if (errOut) *errOut = std::string("config: json parse error: ") + e.what();
            if (state) state->lastError = *errOut;
            return false;
        }
#else
        MergeIntoConsole_Safe(io, raw, state);
#endif
        return true;
    }

    bool TryHotReload(State& state, console::RuntimeConsole& io, std::string* errOut) {
        std::error_code ec;
        fs::file_time_type ft;
        if (!fileTime(state.path, ft, ec) || ec) return false;
        if (state.lastWrite == fs::file_time_type{} || ft <= state.lastWrite) return false;

        console::RuntimeConsole tmp = io;
        State tmpState = state;
        std::string err;
        if (!LoadFile(state.path, tmp, &tmpState, &err)) {
            if (errOut) *errOut = err;
            return false;
        }

        const auto& wl = ResolveWhitelist(&state);
        ApplyWhitelistDiff(wl, tmp, io, state);

        state.lastWrite = tmpState.lastWrite;
        state.activeProfile = tmpState.activeProfile;
        return true;
    }

} // namespace config
