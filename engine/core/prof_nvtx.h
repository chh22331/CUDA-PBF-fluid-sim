#pragma once
// 统一 NVTX 标记（可在运行时关闭）
#include <cstdint>
#include <nvtx3/nvtx3.hpp>
 

namespace prof {

    // 颜色生成（ARGB）
    static inline uint32_t Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) {
        return (uint32_t(a) << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
    }

    // 全局开关（由外部根据 RuntimeConsole 决定）
    inline bool& NvtxEnabledFlag() {
        static bool g = true;
        return g;
    }
    inline void SetNvtxEnabled(bool v) { NvtxEnabledFlag() = v; }

#if defined(ENABLE_NVTX)
    struct Range {
        Range(const char* name, uint32_t color = Color(0x30, 0x90, 0xFF)) {
            if (!NvtxEnabledFlag()) return;
            nvtxEventAttributes_t ev{};
            ev.version = NVTX_VERSION;
            ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            ev.colorType = NVTX_COLOR_ARGB;
            ev.color = color;
            ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
            ev.message.ascii = name;
            nvtxRangePushEx(&ev);
            m_active = true;
        }
        ~Range() {
            if (m_active) nvtxRangePop();
        }
    private:
        bool m_active = false;
    };
    inline void Mark(const char* name, uint32_t color = Color(0xAA, 0xAA, 0xAA)) {
        if (!NvtxEnabledFlag()) return;
        nvtxEventAttributes_t ev{};
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = color;
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = name;
        nvtxMarkEx(&ev);
    }
#else
    struct Range { Range(const char*, uint32_t = 0) {} };
    inline void Mark(const char*, uint32_t = 0) {}
#endif

} // namespace prof