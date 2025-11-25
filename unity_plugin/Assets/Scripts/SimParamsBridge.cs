using System;
using System.Runtime.InteropServices;
using UnityEngine;

[StructLayout(LayoutKind.Sequential)]
public struct Float3 { public float x, y, z; public Float3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; } }

[StructLayout(LayoutKind.Sequential)]
public struct PbfTuningInterop
{
    public int scorr_enable;
    public float scorr_k;
    public float scorr_n;
    public float scorr_dq_h;
    public float wq_min;
    public float scorr_min;

    public float grad_r_eps;
    public float lambda_denom_eps;

    public float compliance;
    public int xpbd_enable;

    public int enable_lambda_clamp;
    public float lambda_max_abs;
    public int enable_disp_clamp;
    public float disp_clamp_max_h;

    public int enable_relax;
    public float relax_omega;

    public int xsph_gate_enable;
    public int xsph_n_min;
    public int xsph_n_max;

    public int lambda_warm_start_enable;
    public float lambda_warm_start_decay;

    public int semi_implicit_integration_enable;
}

[StructLayout(LayoutKind.Sequential)]
public struct KernelCoeffsInterop
{
    public float h, inv_h, h2, poly6, spiky, visc;
}

[StructLayout(LayoutKind.Sequential)]
public struct Int3 { public int x, y, z; }

[StructLayout(LayoutKind.Sequential)]
public struct GridBoundsInterop
{
    public Float3 mins;
    public Float3 maxs;
    public float cellSize;
    public Int3 dim; // 可不填，由 native 侧计算
}

[StructLayout(LayoutKind.Sequential)]
public struct SimParamsInterop
{
    public uint numParticles;
    public uint maxParticles;
    public Float3 gravity;
    public float dt;
    public float cfl;
    public float restDensity;
    public int solverIters;
    public int maxNeighbors;
    public int sortEveryN;
    public float boundaryRestitution;
    public float particleMass;
    public PbfTuningInterop pbf;
    public float xsph_c;
    public KernelCoeffsInterop kernel;
    public GridBoundsInterop grid;
    public uint ghostParticleCount;
    public float maxSpeedClamp;
    public uint version;

    public static SimParamsInterop Default(uint N, float h = 2.0f)
    {
        var pbf = new PbfTuningInterop
        {
            scorr_enable = 1,
            scorr_k = 0.003f,
            scorr_n = 4.0f,
            scorr_dq_h = 0.3f,
            wq_min = 1e-12f,
            scorr_min = -0.25f,
            grad_r_eps = 1e-6f,
            lambda_denom_eps = 1e-4f,
            compliance = 0.0f,
            xpbd_enable = 0,
            enable_lambda_clamp = 1,
            lambda_max_abs = 50.0f,
            enable_disp_clamp = 1,
            disp_clamp_max_h = 0.05f,
            enable_relax = 1,
            relax_omega = 0.75f,
            xsph_gate_enable = 0,
            xsph_n_min = 0,
            xsph_n_max = 8,
            lambda_warm_start_enable = 0,
            lambda_warm_start_decay = 0.5f,
            semi_implicit_integration_enable = 1
        };
        var k = new KernelCoeffsInterop { h = h, inv_h = (h > 0 ? 1.0f / h : 0), h2 = h * h, poly6 = 0, spiky = 0, visc = 0 };
        var g = new GridBoundsInterop
        {
            mins = new Float3(0, 0, 0),
            maxs = new Float3(200, 200, 200),
            cellSize = 0.0f,
            dim = new Int3()
        };
        return new SimParamsInterop
        {
            version = 1,
            numParticles = N,
            maxParticles = N,
            gravity = new Float3(0, -9.8f, 0),
            dt = 0.0167f,
            cfl = 0.45f,
            restDensity = 1.0f,
            solverIters = 1,
            maxNeighbors = 64,
            sortEveryN = 4,
            boundaryRestitution = 0.0f,
            particleMass = 1.0f,
            pbf = pbf,
            xsph_c = 0.05f,
            kernel = k,
            grid = g,
            ghostParticleCount = 0,
            maxSpeedClamp = -1.0f
        };
    }
}

public static class NativeSimParamsAPI
{
    private const string Dll = "NativeSimPlugin";
    [DllImport(Dll)] public static extern void Sim_SetParams(ref SimParamsInterop p);
    [DllImport(Dll)] public static extern void Sim_GetParams(out SimParamsInterop p);
    [DllImport(Dll)] public static extern void Sim_UpdateDt(float dt);
    [DllImport(Dll)] public static extern void Sim_UpdateGravity(float gx, float gy, float gz);
    [DllImport(Dll)] public static extern void Sim_UpdateCounts(uint numParticles, uint maxParticles);
}