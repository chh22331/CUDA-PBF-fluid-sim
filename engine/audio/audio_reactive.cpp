// Pull in stb_vorbis declarations before enabling Vorbis decoders in miniaudio.
#define STB_VORBIS_HEADER_ONLY
#include "../../audio/miniaudio/extras/stb_vorbis.c"
// Clean up stb_vorbis's short macros that collide with Windows headers.
#undef PLAYBACK_MONO
#undef PLAYBACK_LEFT
#undef PLAYBACK_RIGHT
#undef L
#undef C
#undef R

#define MA_ENABLE_VORBIS
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#include "../../audio/miniaudio/miniaudio.h"

#include "audio_reactive.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <cstdarg>
#include <string>
#include <filesystem>
#include <array>
#include <cctype>

namespace {

namespace fs = std::filesystem;

inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

inline float hannWindow(uint32_t i, uint32_t n) {
    if (n <= 1) return 1.0f;
    const float twoPi = 6.28318530717958647692f;
    return 0.5f - 0.5f * std::cos(twoPi * float(i) / float(n - 1));
}

inline float smoothingAlpha(double dt, double timeConstant) {
    if (timeConstant <= 0.0) return 1.0f;
    double a = 1.0 - std::exp(-dt / timeConstant);
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;
    return static_cast<float>(a);
}

bool equalsIgnoreCase(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        unsigned char ca = static_cast<unsigned char>(a[i]);
        unsigned char cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

fs::path ResolveAudioFilePath() {
    static constexpr std::array<const char*, 11> kCommonExts{
        "wav", "mp3", "flac", "ogg", "opus", "aac", "m4a", "wma", "aiff", "aif", "caf"
    };
    std::error_code ec;
    fs::path audioDir = fs::path("audio");
    for (const char* ext : kCommonExts) {
        fs::path candidate = audioDir / (std::string("audio.") + ext);
        if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            return candidate;
        }
    }
    if (!fs::exists(audioDir, ec) || !fs::is_directory(audioDir, ec)) {
        return {};
    }
    for (const auto& entry : fs::directory_iterator(audioDir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        std::string stem = entry.path().stem().string();
        if (equalsIgnoreCase(stem, "audio")) {
            return entry.path();
        }
    }
    return {};
}

} // namespace

namespace audio {

namespace {

void DebugLog(const AudioReactiveSystem::Settings& cfg, const char* fmt, ...) {
    if (!cfg.debugPrint) return;
    std::printf("[Audio][dbg] ");
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::printf("\n");
}

} // namespace

AudioReactiveSystem::~AudioReactiveSystem() {
    shutdown();
}

void AudioReactiveSystem::PlaybackThunk(ma_device* device, void* output, const void* input, uint32_t frameCount) {
    (void)input;
    const uint32_t fallbackChannels = (device && device->playback.channels > 0) ? device->playback.channels : 1u;
    auto* self = reinterpret_cast<AudioReactiveSystem*>(device ? device->pUserData : nullptr);
    if (!self || !output || !device) {
        if (output) std::memset(output, 0, static_cast<size_t>(frameCount) * fallbackChannels * sizeof(float));
        return;
    }
    uint32_t channels = device->playback.channels;
    if (channels == 0) channels = 1;
    float* out = static_cast<float*>(output);
    ma_uint64 framesRemaining = frameCount;
    while (framesRemaining > 0) {
        if (!self->m_decoder) {
            std::memset(out, 0, static_cast<size_t>(framesRemaining) * channels * sizeof(float));
            self->pushSamples(out, static_cast<uint32_t>(framesRemaining), channels);
            framesRemaining = 0;
            break;
        }
        ma_uint64 framesRead = 0;
        ma_result readRes = ma_decoder_read_pcm_frames(self->m_decoder, out, framesRemaining, &framesRead);
        if (readRes != MA_SUCCESS) {
            std::memset(out, 0, static_cast<size_t>(framesRemaining) * channels * sizeof(float));
            self->pushSamples(out, static_cast<uint32_t>(framesRemaining), channels);
            framesRemaining = 0;
            break;
        }
        if (framesRead == 0) {
            ma_result seekResult = ma_decoder_seek_to_pcm_frame(self->m_decoder, 0);
            if (seekResult != MA_SUCCESS) {
                std::memset(out, 0, static_cast<size_t>(framesRemaining) * channels * sizeof(float));
                self->pushSamples(out, static_cast<uint32_t>(framesRemaining), channels);
                framesRemaining = 0;
                break;
            }
            continue;
        }
        self->pushSamples(out, static_cast<uint32_t>(framesRead), channels);
        out += framesRead * channels;
        framesRemaining -= framesRead;
    }
    if (framesRemaining > 0) {
        std::memset(out, 0, static_cast<size_t>(framesRemaining) * channels * sizeof(float));
        self->pushSamples(out, static_cast<uint32_t>(framesRemaining), channels);
    }
}

bool AudioReactiveSystem::initialize(const Settings& settings) {
    shutdown();

    Settings cfg = settings;
    cfg.keyCount = std::clamp<uint32_t>(cfg.keyCount == 0 ? sim::kAudioKeyCount : cfg.keyCount, 1u, sim::kAudioKeyCount);
    if ((cfg.fftSize == 0) || (cfg.fftSize & (cfg.fftSize - 1)) != 0) {
        std::fprintf(stderr, "[Audio] fftSize must be a power of two (got %u)\n", cfg.fftSize);
        return false;
    }
    cfg.channels = (cfg.channels == 0) ? 2u : cfg.channels;
    cfg.minFrequencyHz = std::max(10.0f, cfg.minFrequencyHz);
    cfg.maxFrequencyHz = std::max(cfg.minFrequencyHz + 10.0f, cfg.maxFrequencyHz);

    DebugLog(cfg, "initialize: enabled=%d targetRate=%uHz ch=%u fft=%u ring=%ums",
        cfg.enabled ? 1 : 0, cfg.sampleRate, cfg.channels, cfg.fftSize, cfg.ringBufferMs);

    if (!cfg.enabled) {
        m_settings = cfg;
        m_gateLinear = std::pow(10.0f, cfg.noiseGateDb / 20.0f);
        DebugLog(cfg, "initialize skipped: audio disabled.");
        resetFrameData();
        return true;
    }

    fs::path audioPath = ResolveAudioFilePath();
    ma_decoder* decoder = nullptr;
    ma_uint64 totalFrames = 0;
    bool decoderReady = false;
    bool decoderInitialized = false;

    if (!audioPath.empty()) {
        m_audioFilePath = audioPath.u8string();
        ma_decoder_config decCfg = ma_decoder_config_init(ma_format_f32, 0, 0);
        decoder = new(std::nothrow) ma_decoder();
        if (!decoder) {
            std::fprintf(stderr, "[Audio] decoder allocation failed.\n");
        } else {
#ifdef _WIN32
            ma_result decRes = ma_decoder_init_file_w(audioPath.c_str(), &decCfg, decoder);
#else
            ma_result decRes = ma_decoder_init_file(audioPath.string().c_str(), &decCfg, decoder);
#endif
            if (decRes != MA_SUCCESS) {
                std::fprintf(stderr, "[Audio] failed to open %s (ma_result=%d)\n", m_audioFilePath.c_str(), decRes);
            } else {
                decoderInitialized = true;
                if (decoder->outputSampleRate == 0 || decoder->outputChannels == 0) {
                    std::fprintf(stderr, "[Audio] decoder reported invalid format for %s\n", m_audioFilePath.c_str());
                } else if (ma_decoder_get_length_in_pcm_frames(decoder, &totalFrames) != MA_SUCCESS || totalFrames == 0) {
                    std::fprintf(stderr, "[Audio] %s contains no audio data.\n", m_audioFilePath.c_str());
                } else {
                    decoderReady = true;
                }
            }
        }
    } else {
        if (cfg.debugPrint) {
            std::printf("[Audio][dbg] audio/audio.* file not found, using silent source.\n");
        }
    }

    if (!decoderReady) {
        if (decoder) {
            if (decoderInitialized) {
                ma_decoder_uninit(decoder);
            }
            delete decoder;
            decoder = nullptr;
        }
        m_audioFilePath.clear();
        if (cfg.sampleRate == 0) cfg.sampleRate = 48000;
        cfg.channels = std::max(1u, cfg.channels);
    } else {
        cfg.sampleRate = decoder->outputSampleRate;
        cfg.channels = decoder->outputChannels;
        DebugLog(cfg, "audio file resolved: %s (%u Hz, %u ch, %llu frames)",
            m_audioFilePath.c_str(), cfg.sampleRate, cfg.channels,
            static_cast<unsigned long long>(totalFrames));
    }
    m_decoder = decoderReady ? decoder : nullptr;

    const size_t minRingSamples = static_cast<size_t>(cfg.sampleRate * (cfg.ringBufferMs / 1000.0f));
    size_t ringSamples = std::max<size_t>(minRingSamples, cfg.fftSize * 2ull);
    if (ringSamples == 0) ringSamples = cfg.fftSize * 2ull;

    m_ringBuffer.assign(ringSamples, 0.0f);
    m_ringRead = m_ringWrite = m_ringCount = 0;
    m_reportedStarved = false;
    DebugLog(cfg, "ring buffer configured: %zu samples (%.1f ms window)", ringSamples,
        double(ringSamples) / double(std::max(1u, cfg.sampleRate)) * 1000.0);

    m_windowScratch.resize(cfg.fftSize);
    for (uint32_t i = 0; i < cfg.fftSize; ++i) {
        m_windowScratch[i] = hannWindow(i, cfg.fftSize);
    }

    m_fftInput.assign(cfg.fftSize, 0.0f);
    m_sampleScratch.assign(cfg.fftSize, 0.0f);
    m_fftOutput.resize(cfg.fftSize / 2 + 1);
    m_fftMagnitudes.assign(cfg.fftSize / 2 + 1, 0.0f);
    m_targets.assign(sim::kAudioKeyCount, 0.0f);
    m_smoothed.assign(sim::kAudioKeyCount, 0.0f);

    m_fftCfg = kiss_fftr_alloc(static_cast<int>(cfg.fftSize), 0, nullptr, nullptr);
    if (!m_fftCfg) {
        std::fprintf(stderr, "[Audio] kiss_fftr_alloc failed for N=%u\n", cfg.fftSize);
        return false;
    }

    ma_device_config devCfg = ma_device_config_init(ma_device_type_playback);
    devCfg.playback.format = ma_format_f32;
    devCfg.playback.channels = cfg.channels;
    devCfg.sampleRate = cfg.sampleRate;
    devCfg.dataCallback = AudioReactiveSystem::PlaybackThunk;
    devCfg.pUserData = this;

    ma_device* dev = new(std::nothrow) ma_device();
    if (!dev) {
        std::fprintf(stderr, "[Audio] playback device allocation failed.\n");
        return false;
    }
    if (ma_device_init(nullptr, &devCfg, dev) != MA_SUCCESS) {
        std::fprintf(stderr, "[Audio] playback device init failed.\n");
        delete dev;
        return false;
    }
    if (ma_device_start(dev) != MA_SUCCESS) {
        std::fprintf(stderr, "[Audio] playback device start failed.\n");
        ma_device_uninit(dev);
        delete dev;
        return false;
    }
    m_device = dev;
    m_settings = cfg;
    m_gateLinear = std::pow(10.0f, cfg.noiseGateDb / 20.0f);

    resetFrameData();
    m_active = true;
    if (cfg.debugPrint) {
        std::printf("[Audio] playback started: %s (%u Hz, %u ch, fft=%u)\n",
            m_audioFilePath.c_str(),
            cfg.sampleRate, cfg.channels, cfg.fftSize);
    }
    return true;
}

void AudioReactiveSystem::shutdown() {
    DebugLog(m_settings, "shutdown requested (device=%s)", m_device ? "active" : "none");
    if (m_device) {
        ma_device_stop(m_device);
        ma_device_uninit(m_device);
        delete m_device;
        m_device = nullptr;
    }
    if (m_decoder) {
        ma_decoder_uninit(m_decoder);
        delete m_decoder;
        m_decoder = nullptr;
    }
    m_audioFilePath.clear();
    if (m_fftCfg) {
        kiss_fftr_free(m_fftCfg);
        m_fftCfg = nullptr;
    }
    m_active = false;
    m_ringBuffer.clear();
    m_fftInput.clear();
    m_sampleScratch.clear();
    m_fftOutput.clear();
    m_fftMagnitudes.clear();
    m_windowScratch.clear();
    m_targets.assign(sim::kAudioKeyCount, 0.0f);
    m_smoothed.assign(sim::kAudioKeyCount, 0.0f);
    resetFrameData();
    if (m_settings.debugPrint && m_debugSummary.fftFrames > 0) {
        float avgHitPct = (m_debugSummary.binsTotal > 0)
            ? (100.0f * double(m_debugSummary.hitsTotal) / double(m_debugSummary.binsTotal))
            : 0.0f;
        DebugLog(m_settings,
            "summary: fftFrames=%llu bins=%llu hits=%llu (%.1f%%) beats=%llu peakGlobal=%.4f",
            static_cast<unsigned long long>(m_debugSummary.fftFrames),
            static_cast<unsigned long long>(m_debugSummary.binsTotal),
            static_cast<unsigned long long>(m_debugSummary.hitsTotal),
            avgHitPct,
            static_cast<unsigned long long>(m_debugSummary.beats),
            m_debugSummary.peakGlobalEnergy);
    }
}

void AudioReactiveSystem::tick(double dtSeconds) {
    if (dtSeconds <= 0.0) dtSeconds = 1.0 / 60.0;

    if (m_sampleScratch.size() != m_settings.fftSize) {
        m_sampleScratch.assign(m_settings.fftSize, 0.0f);
    }
    bool fftUpdated = false;
    if (m_fftCfg) {
        // Only refresh the spectrum when we have a full block of fresh samples; otherwise
        // keep the previous targets so the smoothed envelope does not drop to zero.
        if (consumeSamples(m_sampleScratch)) {
            processSpectrum(m_sampleScratch);
            fftUpdated = true;
        }
        else {
            m_debugFft.valid = false;
        }
    }
    else {
        std::fill(m_targets.begin(), m_targets.end(), 0.0f);
        m_debugFft.valid = false;
    }
    applySmoothing(m_targets, dtSeconds);
    updateBeatLogic(dtSeconds);
    if (m_settings.debugPrint && fftUpdated && m_debugFft.valid) {
        float hitPct = (m_debugFft.bins > 0)
            ? (100.0f * float(m_debugFft.hits) / float(m_debugFft.bins))
            : 0.0f;
        ++m_debugSummary.fftFrames;
        m_debugSummary.binsTotal += m_debugFft.bins;
        m_debugSummary.hitsTotal += m_debugFft.hits;
        if (m_frame.globalEnergy > m_debugSummary.peakGlobalEnergy)
            m_debugSummary.peakGlobalEnergy = m_frame.globalEnergy;
        DebugLog(m_settings,
            "fft[%llu]: bins=%u hits=%u (%.1f%%) peakPre=%.4f avgPre=%.4f peakPost=%.4f global=%.4f beat=%u gate=%.4f",
            static_cast<unsigned long long>(m_debugSummary.fftFrames),
            m_debugFft.bins,
            m_debugFft.hits,
            hitPct,
            m_debugFft.peakPre,
            m_debugFft.avgPre,
            m_debugFft.peakPost,
            m_frame.globalEnergy,
            m_frame.isBeat,
            m_gateLinear);
        m_debugFft.valid = false;
    }
}

void AudioReactiveSystem::updateGainAndGate(float intensityGain, float noiseGateDb) {
    m_settings.intensityGain = intensityGain;
    m_settings.noiseGateDb = noiseGateDb;
    m_gateLinear = std::pow(10.0f, m_settings.noiseGateDb / 20.0f);
}

void AudioReactiveSystem::pushSamples(const float* data, uint32_t frames, uint32_t channels) {
    if (!data || channels == 0 || m_ringBuffer.empty()) return;
    std::lock_guard<std::mutex> lock(m_ringMutex);
    const size_t cap = m_ringBuffer.size();
    for (uint32_t f = 0; f < frames; ++f) {
        float mono = 0.0f;
        for (uint32_t c = 0; c < channels; ++c) {
            mono += data[f * channels + c];
        }
        mono /= float(channels);
        m_ringBuffer[m_ringWrite] = mono;
        m_ringWrite = (m_ringWrite + 1) % cap;
        if (m_ringCount < cap) {
            ++m_ringCount;
        }
        else {
            m_ringRead = (m_ringRead + 1) % cap;
        }
    }
}

bool AudioReactiveSystem::consumeSamples(std::vector<float>& dst) {
    const size_t needed = dst.size();
    if (needed == 0 || m_ringBuffer.empty()) return false;
    std::lock_guard<std::mutex> lock(m_ringMutex);
    if (m_ringCount < needed) {
        if (m_settings.debugPrint && !m_reportedStarved) {
            DebugLog(m_settings, "sample starvation: have=%zu need=%zu ring=%zu", m_ringCount, needed, m_ringBuffer.size());
            m_reportedStarved = true;
        }
        return false;
    }
    const size_t cap = m_ringBuffer.size();
    for (size_t i = 0; i < needed; ++i) {
        dst[i] = m_ringBuffer[m_ringRead];
        m_ringRead = (m_ringRead + 1) % cap;
    }
    m_ringCount -= needed;
    if (m_reportedStarved && m_settings.debugPrint) {
        DebugLog(m_settings, "sample stream recovered: consumed=%zu remaining=%zu", needed, m_ringCount);
    }
    m_reportedStarved = false;
    return true;
}

void AudioReactiveSystem::processSpectrum(const std::vector<float>& samples) {
    const uint32_t N = m_settings.fftSize;
    for (uint32_t i = 0; i < N; ++i) {
        float s = (i < samples.size()) ? samples[i] : 0.0f;
        m_fftInput[i] = s * ((i < m_windowScratch.size()) ? m_windowScratch[i] : 1.0f);
    }
    kiss_fftr(m_fftCfg, m_fftInput.data(), m_fftOutput.data());
    std::fill(m_targets.begin(), m_targets.end(), 0.0f);
    if (m_binCounts.size() != sim::kAudioKeyCount) {
        m_binCounts.assign(sim::kAudioKeyCount, 0u);
    }
    else {
        std::fill(m_binCounts.begin(), m_binCounts.end(), 0u);
    }

    uint32_t binsConsidered = 0;
    uint32_t binsAboveGate = 0;
    float peakPre = 0.0f;
    float peakPost = 0.0f;
    float sumPre = 0.0f;

    const float fftNorm = (N > 0) ? (2.0f / float(N)) : 0.0f;
    constexpr float kHannEnergy = 0.5f;
    const float magnitudeScale = (kHannEnergy > 0.0f) ? (fftNorm / kHannEnergy) : 0.0f;

    const float freqStep = float(m_settings.sampleRate) / float(N);
    const float freqMin = std::max(1e-3f, m_settings.minFrequencyHz);
    const float freqMax = std::max(freqMin + 1e-3f, m_settings.maxFrequencyHz);
    const float logMin = std::log(freqMin);
    const float logRange = std::max(1e-6f, std::log(freqMax) - logMin);
    for (uint32_t i = 0; i <= N / 2; ++i) {
        float freq = freqStep * float(i);
        if (freq < freqMin || freq > freqMax) continue;
        float logFreq = std::log(std::max(freq, freqMin));
        float norm = (logFreq - logMin) / logRange;
        norm = std::min(std::max(norm, 0.0f), 0.9999f);
        uint32_t key = std::min<uint32_t>(static_cast<uint32_t>(norm * float(m_settings.keyCount)), m_settings.keyCount - 1);
        float re = m_fftOutput[i].r;
        float im = m_fftOutput[i].i;
        float mag = std::sqrt(re * re + im * im) * magnitudeScale;
        m_targets[key] += mag;
        m_binCounts[key] += 1;
        ++binsConsidered;
        sumPre += mag;
        if (mag > peakPre) peakPre = mag;
    }

    for (uint32_t k = 0; k < m_settings.keyCount; ++k) {
        float avg = (m_binCounts[k] > 0) ? (m_targets[k] / float(m_binCounts[k])) : 0.0f;
        avg = std::max(avg, 0.0f);
        bool passedGate = avg > m_gateLinear;
        if (passedGate) ++binsAboveGate;
        float boosted = avg * m_settings.intensityGain;
        float level = passedGate ? (1.0f - std::exp(-boosted)) : 0.0f;
        level = clamp01(level);
        m_targets[k] = level;
        if (level > peakPost) peakPost = level;
    }
    for (uint32_t k = m_settings.keyCount; k < sim::kAudioKeyCount; ++k) {
        m_targets[k] = 0.0f;
    }
    m_debugFft.valid = true;
    m_debugFft.bins = binsConsidered;
    m_debugFft.hits = binsAboveGate;
    m_debugFft.peakPre = peakPre;
    m_debugFft.peakPost = peakPost;
    m_debugFft.avgPre = (binsConsidered > 0) ? (sumPre / float(binsConsidered)) : 0.0f;
}

void AudioReactiveSystem::applySmoothing(const std::vector<float>& targets, double dtSeconds) {
    float alphaUp = smoothingAlpha(dtSeconds, m_settings.smoothingAttackSec);
    float alphaDown = smoothingAlpha(dtSeconds, m_settings.smoothingReleaseSec);
    float sum = 0.0f;
    for (uint32_t k = 0; k < sim::kAudioKeyCount; ++k) {
        float target = (k < targets.size()) ? targets[k] : 0.0f;
        float current = m_smoothed[k];
        float alpha = (target > current) ? alphaUp : alphaDown;
        current += (target - current) * alpha;
        current = clamp01(current);
        m_smoothed[k] = current;
        m_frame.keyIntensities[k] = current;
        if (k < m_settings.keyCount) sum += current;
    }
    m_frame.globalEnergy = (m_settings.keyCount > 0) ? (sum / float(m_settings.keyCount)) : 0.0f;
}

void AudioReactiveSystem::updateBeatLogic(double dtSeconds) {
    float emaAlpha = smoothingAlpha(dtSeconds, m_settings.globalEnergyEmaSeconds);
    m_energyEma += (m_frame.globalEnergy - m_energyEma) * emaAlpha;
    m_frame.globalEnergyEma = m_energyEma;

    m_timeSinceBeat += dtSeconds;
    const float beatGate = m_energyEma * m_settings.beatThreshold;
    bool triggerBeat = (m_frame.globalEnergy > beatGate) && (m_timeSinceBeat >= m_settings.beatHoldSeconds);
    if (triggerBeat) {
        m_timeSinceBeat = 0.0;
        m_frame.isBeat = 1;
        m_frame.beatStrength = m_frame.globalEnergy;
        if (m_settings.debugPrint)
            ++m_debugSummary.beats;
        if (m_settings.debugPrint) {
            DebugLog(m_settings, "beat triggered: energy=%.4f ema=%.4f gate=%.4f hold=%.2fs",
                m_frame.globalEnergy,
                m_energyEma,
                beatGate,
                m_settings.beatHoldSeconds);
        }
    }
    else {
        m_frame.isBeat = 0;
        m_frame.beatStrength = 0.0f;
    }
    m_frame.frameSeed = ++m_frameSeed;
}

void AudioReactiveSystem::resetFrameData() {
    m_frame = sim::AudioFrameData{};
    m_energyEma = 0.0f;
    m_timeSinceBeat = 0.0;
    m_frameSeed = 0;
}

} // namespace audio
