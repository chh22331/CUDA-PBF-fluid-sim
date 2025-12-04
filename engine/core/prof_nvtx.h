#pragma once
#include <cstdint>
#include <optional>

#if defined(ENABLE_NVTX)
    #include <nvtx3/nvtx3.hpp>
#endif

namespace prof {

    // Color palette (ARGB).
    static inline uint32_t Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) {
        return (uint32_t(a) << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
    }

    // Runtime toggle.
    inline bool& NvtxEnabledFlag() {
        static bool g = true;
        return g;
    }
    inline void SetNvtxEnabled(bool v) { NvtxEnabledFlag() = v; }

#if defined(ENABLE_NVTX)

    // Shared domain makes Nsight Systems filtering easier.
    inline nvtxDomainHandle_t NvtxDomain() {
        static nvtxDomainHandle_t h = nvtxDomainCreateA("PBF-X");
        return h;
    }

    // RAII range helper leveraging NVTX3 C++ for color/name metadata.
    class Range {
    public:
        Range(const char* name, uint32_t argb = Color(0x30, 0x90, 0xFF)) {
            if (!NvtxEnabledFlag()) return;
            nvtxEventAttributes_t ev{};
            ev.version = NVTX_VERSION;
            ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            ev.colorType = NVTX_COLOR_ARGB;
            ev.color = argb;
            ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
            ev.message.ascii = name;
            // Use the domain variant to keep lanes separate.
            nvtxDomainRangePushEx(NvtxDomain(), &ev);
            m_active = true;
        }
        ~Range() {
            if (m_active) {
                nvtxDomainRangePop(NvtxDomain());
            }
        }
        Range(const Range&) = delete;
        Range& operator=(const Range&) = delete;
    private:
        bool m_active = false;
    };

    // Instantaneous event markers.
    inline void Mark(const char* name, uint32_t argb = Color(0xAA, 0xAA, 0xAA)) {
        if (!NvtxEnabledFlag()) return;
        nvtxEventAttributes_t ev{};
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = argb;
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = name;
        nvtxDomainMarkEx(NvtxDomain(), &ev);
    }

    // Optional lightweight wrappers over NVTX3 C++ APIs.
    class Scoped {
    public:
        explicit Scoped(const char* name, uint32_t argb = Color(0x50, 0x50, 0xC0)) {
            if (!NvtxEnabledFlag()) return;
            nvtxEventAttributes_t ev{};
            ev.version = NVTX_VERSION;
            ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            ev.colorType = NVTX_COLOR_ARGB;
            ev.color = argb;
            ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
            ev.message.ascii = name;
            nvtxDomainRangePushEx(NvtxDomain(), &ev);
            m_active = true;
        }
        ~Scoped() {
            if (m_active) nvtxDomainRangePop(NvtxDomain());
        }
    private:
        bool m_active = false;
    };

#else
    // No-op shims when NVTX is disabled.
    class Range {
    public: Range(const char*, uint32_t = 0) {}
    };
    class Scoped {
    public: Scoped(const char*, uint32_t = 0) {}
    };
    inline void Mark(const char*, uint32_t = 0) {}
#endif

} // namespace prof
