#include "simulator.h"
#include "precision_traits.cuh"
#include "../engine/core/console.h"

namespace sim {

    // 简单方差估计辅助（在线）
    struct OnlineVar {
        double mean = 0.0;
        double m2 = 0.0;
        uint64_t n = 0;
        __host__ inline void add(double x) {
            ++n;
            double delta = x - mean;
            mean += delta / double(n);
            double delta2 = x - mean;
            m2 += delta * delta2;
        }
        __host__ inline double variance() const {
            return (n > 1) ? (m2 / double(n - 1)) : 0.0;
        }
    };

    // 根据 SimStats 与控制台阈值决定是否调整只读半精开关
    bool Simulator::adaptivePrecisionCheck(const SimStats& stats) {
        const auto& cfg = console::Instance().sim.precision;
        if (!cfg.adaptivePrecision) return false;
        if (cfg.adaptCheckEveryN <= 0) return false;
        if ((m_frameIndex % (uint64_t)cfg.adaptCheckEveryN) != 0ull) return false;

        // 估算相对密度误差（与 target restDensity 比较）
        double rhoRel = stats.avgRhoRel; // 已是 avgRho / restDensity
        double densityError = fabs(rhoRel - 1.0); // 与 1 比较
        m_adaptDensityErrorHistory.add(densityError);

        // λ 方差（需要有 lambda 缓冲；若无则跳过）
        double lambdaVar = 0.0;
        if (m_bufs.d_lambda && stats.N > 0) {
            // 采样子集（限制高开销）
            uint32_t sample = std::min<uint32_t>(stats.N, 8192);
            std::vector<float> h_lambda(sample);
            cudaMemcpy(h_lambda.data(), m_bufs.d_lambda, sizeof(float) * sample, cudaMemcpyDeviceToHost);
            OnlineVar ov;
            for (uint32_t i = 0; i < sample; ++i) ov.add((double)h_lambda[i]);
            lambdaVar = ov.variance();
        }
        m_adaptLambdaVarHistory.add(lambdaVar);

        bool needUpgrade = (densityError > (double)cfg.densityErrorTolerance) ||
            (lambdaVar > (double)cfg.lambdaVarianceTolerance);

        // 冷却/回退策略：若连续多次低于 50% 阈值则尝试恢复 half 加载
        bool canDowngrade = (!needUpgrade) &&
            (densityError < 0.5 * (double)cfg.densityErrorTolerance) &&
            (lambdaVar < 0.5 * (double)cfg.lambdaVarianceTolerance);

        constexpr int kUpgradeHoldFrames = 2;
        constexpr int kDowngradeHoldFrames = 5;

        bool changed = false;
        if (needUpgrade) {
            if (m_adaptUpgradeHold < kUpgradeHoldFrames) {
                ++m_adaptUpgradeHold;
            }
            else {
                // 提升：禁用半精只读（回到 FP32 主缓冲加载）
                if (m_adaptHalfDisabled == false) {
                    m_adaptHalfDisabled = true;
                    changed = true;
                }
                m_adaptDowngradeHold = 0;
            }
        }
        else if (canDowngrade && m_adaptHalfDisabled) {
            if (m_adaptDowngradeHold < kDowngradeHoldFrames) {
                ++m_adaptDowngradeHold;
            }
            else {
                // 回退到半精只读
                m_adaptHalfDisabled = false;
                changed = true;
                m_adaptUpgradeHold = 0;
            }
        }
        else {
            // 中性状态
            m_adaptUpgradeHold = 0;
            m_adaptDowngradeHold = 0;
        }

        if (changed) {
            std::fprintf(stderr,
                "[AdaptivePrecision] Frame=%llu | densityErr=%.4g lambdaVar=%.4g | halfDisabled=%d\n",
                (unsigned long long)m_frameIndex, densityError, lambdaVar, m_adaptHalfDisabled ? 1 : 0);

            // 刷新设备常量视图（不改变分配，仅修改 useHalf* 标志逻辑映射）
            SimPrecision pr = m_params.precision;
            if (m_adaptHalfDisabled) {
                // 暂存用户原始配置
                pr._adaptive_pos_prev = pr.positionStore;
                pr._adaptive_vel_prev = pr.velocityStore;
                pr._adaptive_pos_pred_prev = pr.predictedPosStore;
                // 强制视图改为 FP32（保留缓冲不释放，快速切换）
                pr.positionStore = NumericType::FP32;
                pr.velocityStore = NumericType::FP32;
                pr.predictedPosStore = NumericType::FP32;
            }
            else {
                if (pr._adaptive_pos_prev != NumericType::InvalidSentinel) {
                    pr.positionStore = pr._adaptive_pos_prev;
                    pr.velocityStore = pr._adaptive_vel_prev;
                    pr.predictedPosStore = pr._adaptive_pos_pred_prev;
                }
            }
            m_params.precision = pr;
            UpdateDevicePrecisionView(m_bufs, m_params.precision);
        }
        return changed;
    }

} // namespace sim