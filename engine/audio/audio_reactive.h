#pragma once
#include <cstdint>
#include <vector>
#include <mutex>
#include <string>
#include "../../sim/audio_frame.h"
#include "../../audio/kissfft/kiss_fftr.h"

struct ma_device;
struct ma_decoder;

namespace audio {

class AudioReactiveSystem {
public:
    struct Settings {
        bool     enabled = true;
        uint32_t sampleRate = 48000;
        uint32_t channels = 1;
        bool     preferLoopback = true;
        bool     fallbackToCapture = true;
        uint32_t fftSize = 2048;
        uint32_t ringBufferMs = 250;
        uint32_t keyCount = sim::kAudioKeyCount;
        float    minFrequencyHz = 27.5f;
        float    maxFrequencyHz = 4200.0f;
        float    intensityGain = 12.0f;
        float    noiseGateDb = -55.0f;
        float    smoothingAttackSec = 0.05f;
        float    smoothingReleaseSec = 0.18f;
        float    beatThreshold = 2.25f;
        float    beatHoldSeconds = 0.08f;
        float    beatReleaseSeconds = 0.45f;
        float    globalEnergyEmaSeconds = 0.35f;
        bool     debugPrint = false;
    };

    AudioReactiveSystem() = default;
    ~AudioReactiveSystem();

    bool initialize(const Settings& settings);
    void shutdown();
    void tick(double dtSeconds);
    void updateGainAndGate(float intensityGain, float noiseGateDb);

    bool isActive() const { return m_active; }
    const sim::AudioFrameData& frameData() const { return m_frame; }

private:
    void pushSamples(const float* data, uint32_t frames, uint32_t channels);
    bool consumeSamples(std::vector<float>& dst);
    void processSpectrum(const std::vector<float>& samples);
    void applySmoothing(const std::vector<float>& targets, double dtSeconds);
    void updateBeatLogic(double dtSeconds);
    void resetFrameData();
    static void PlaybackThunk(ma_device* device, void* output, const void* input, uint32_t frameCount);

private:
    Settings m_settings{};
    bool     m_active = false;

    std::vector<float> m_ringBuffer;
    size_t   m_ringRead = 0;
    size_t   m_ringWrite = 0;
    size_t   m_ringCount = 0;
    std::mutex m_ringMutex;

    std::vector<float> m_windowScratch;
    std::vector<float> m_fftInput;
    std::vector<float> m_sampleScratch;
    std::vector<float> m_fftMagnitudes;
    std::vector<kiss_fft_cpx> m_fftOutput;
    std::vector<float> m_smoothed;
    std::vector<float> m_targets;
    std::vector<uint32_t> m_binCounts;

    kiss_fftr_cfg m_fftCfg = nullptr;
    bool    m_reportedStarved = false;

    double  m_timeSinceBeat = 0.0;
    float   m_energyEma = 0.0f;
    uint32_t m_frameSeed = 0;
    float   m_gateLinear = 0.0f;

    sim::AudioFrameData m_frame{};

    struct DebugFftStats {
        bool     valid = false;
        uint32_t bins = 0;
        uint32_t hits = 0;
        float    peakPre = 0.0f;
        float    peakPost = 0.0f;
        float    avgPre = 0.0f;
    } m_debugFft{};
    struct DebugSummary {
        uint64_t fftFrames = 0;
        uint64_t binsTotal = 0;
        uint64_t hitsTotal = 0;
        uint64_t beats = 0;
        float    peakGlobalEnergy = 0.0f;
    } m_debugSummary{};

    ma_device*  m_device = nullptr;
    ma_decoder* m_decoder = nullptr;
    std::string m_audioFilePath;
};

} // namespace audio
