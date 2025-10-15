#pragma once
#include <cstdint>
#include <optional>

#if defined(ENABLE_NVTX)
    // 引入 NVTX3 C & C++ 接口
    //#include <nvtx3/nvtx3.h>
    #include <nvtx3/nvtx3.hpp>
#endif

namespace prof {

    // 颜色 (ARGB)
    static inline uint32_t Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) {
        return (uint32_t(a) << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
    }

    // 运行时开关
    inline bool& NvtxEnabledFlag() {
        static bool g = true;
        return g;
    }
    inline void SetNvtxEnabled(bool v) { NvtxEnabledFlag() = v; }

#if defined(ENABLE_NVTX)

    // 可选：统一一个 domain，便于在 Nsight Systems 里过滤
    inline nvtxDomainHandle_t NvtxDomain() {
        static nvtxDomainHandle_t h = nvtxDomainCreateA("PBF-X");
        return h;
    }

    // RAII 范围（优先使用 NVTX3 C++，若需要携带自定义 color+name）
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
            // 使用 domain 版本，方便区分
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

    // 简单事件标记（瞬时 Mark）
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

    // 也可以提供 NVTX3 C++ 的简洁封装（可选）：
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
    // NVTX 关闭时的空实现
    class Range {
    public: Range(const char*, uint32_t = 0) {}
    };
    class Scoped {
    public: Scoped(const char*, uint32_t = 0) {}
    };
    inline void Mark(const char*, uint32_t = 0) {}
#endif

} // namespace prof