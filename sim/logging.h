#pragma once
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include "../engine/core/console.h"

namespace sim {
    enum class LogChannel : uint8_t {
        Error,
        Warn,
        Hint,
        Diag,
        Perf,
        Precision,
        HotReload
    };

    inline bool LogChannelEnabled(const console::RuntimeConsole& c, LogChannel ch) {
        using LC = LogChannel;
        switch (ch) {
        case LC::Error:     return c.debug.printErrors;
        case LC::Warn:      return c.debug.printWarnings;
        case LC::Hint:      return c.debug.printHints;
        case LC::Diag:      return c.debug.printDiagnostics;
        case LC::HotReload: return c.debug.printHotReload;
        case LC::Precision: return true; 
        case LC::Perf:      return true;  
        default:            return true;
        }
    }

    inline const char* LogChannelPrefix(LogChannel ch) {
        switch (ch) {
        case LogChannel::Error:     return "[Error]"; 
        case LogChannel::Warn:      return "[Warn]"; 
        case LogChannel::Hint:      return "[Hint]"; 
        case LogChannel::Diag:      return "[Diag]"; 
        case LogChannel::Perf:      return "[Perf]"; 
        case LogChannel::Precision: return "[Precision]"; 
        case LogChannel::HotReload: return "[HotReload]"; 
        default:                    return "[Log]"; 
        }
    }

    inline void Log(LogChannel ch, const char* fmt, ...) {
        const auto& c = console::Instance();
        if (!LogChannelEnabled(c, ch)) return;
        std::va_list args; va_start(args, fmt);
        std::fprintf(stderr, "%s ", LogChannelPrefix(ch));
        std::vfprintf(stderr, fmt, args);
        std::fprintf(stderr, "\n");
        va_end(args);
    }
}
