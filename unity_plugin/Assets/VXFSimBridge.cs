using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.VFX;

public class VFXSimBridge : MonoBehaviour
{
    private const string Dll = "NativeSimPlugin";

    // Native 接口（零拷贝主线）
    [DllImport(Dll)] private static extern bool Sim_InitCuda();
    [DllImport(Dll)] private static extern void Sim_ShutdownCuda();
    [DllImport(Dll)] private static extern void Sim_BindPingBuffer(int slot, System.IntPtr nativeResPtr, uint strideBytes, uint capacityElements);
    [DllImport(Dll)] private static extern bool Sim_InitSimulator();
    [DllImport(Dll)] private static extern void Sim_ShutdownSimulator();
    [DllImport(Dll)] private static extern void Sim_SetParams(ref SimParamsInterop p);
    [DllImport(Dll)] private static extern void Sim_UpdateDt(float dt);
    [DllImport(Dll)] private static extern void Sim_UpdateCounts(uint numParticles, uint maxParticles);
    [DllImport(Dll)] private static extern void Sim_SetParticleCount(uint count);
    [DllImport(Dll)] private static extern uint Sim_GetParticleCount();
    [DllImport(Dll)] private static extern int Sim_GetReadIndex();
    [DllImport(Dll)] private static extern System.IntPtr GetRenderEventFunc();
    [DllImport(Dll)] private static extern void Sim_Tick();
    [DllImport(Dll)] private static extern void Sim_SetTime(float t);
    [DllImport(Dll)] private static extern void Sim_EnableCudaWait(int enable);
    [DllImport(Dll)] private static extern bool Sim_SeedCubeMix(uint groupCount, uint edgeParticles, float spacing, int jitterEnable, float jitterAmp, uint jitterSeed);
    [DllImport(Dll)] private static extern bool Sim_BindSimulatorExternalPosPingPong();

    [Header("Buffer Layout")]
    [Tooltip("粒子缓冲元素 = float4(x,y,z,w)。w 可作为 groupId 或半径。")]
    public int strideBytes = 16;

    [Header("Capacity / Count")]
    public int initialCapacity = 262144;
    public int particleCount = 200000;

    [Header("Simulation")]
    public bool enableSimulation = true;
    [Tooltip("GPU-only 双向同步（CUDA 等待渲染完成 & 渲染等待 CUDA 完成）")]
    public bool enableGpuOnlyCrossQueueWait = true;

    [Header("VFX Graph Link")]
    public VisualEffect vfx;
    public string particleBufferProperty = "ParticleBuffer";
    public string particleCountProperty = "ParticleCount";

    [Header("Grouping / Palette")]
    [Tooltip("是否将 pos.w 解释为分组 ID（否则可在 Graph 内当半径使用）。")]
    public bool usePosWAsGroupId = true;
    public string paletteBufferProperty = "PaletteBuffer";
    public string groupCountProperty = "GroupCount";
    public string particlesPerGroupProperty = "ParticlesPerGroup";
    public int groupCount = 4;
    public int particlesPerGroup = 0;
    public Color[] palette = new Color[] { Color.red, Color.green, Color.blue, Color.white };

    [Header("Radius Override")]
    [Tooltip("如果 > 0 且未使用 pos.w 作为分组，可传递给 VFX Graph 的一个统一半径属性。")]
    public float overrideRadius = 0.05f;
    public string overrideRadiusProperty = "ParticleRadius";

    private GraphicsBuffer pingA;
    private GraphicsBuffer pingB;
    private GraphicsBuffer paletteBuffer;

    private int capacity;
    private int stride;

    // 属性 ID
    private static int s_propBuffer;
    private static int s_propCount;
    private static int s_propPalette;
    private static int s_propGroupCount;
    private static int s_propParticlesPerGroup;
    private static int s_propOverrideRadius;

    void Awake()
    {
        s_propBuffer = Shader.PropertyToID(particleBufferProperty);
        s_propCount = Shader.PropertyToID(particleCountProperty);
        s_propPalette = Shader.PropertyToID(paletteBufferProperty);
        s_propGroupCount = Shader.PropertyToID(groupCountProperty);
        s_propParticlesPerGroup = Shader.PropertyToID(particlesPerGroupProperty);
        s_propOverrideRadius = Shader.PropertyToID(overrideRadiusProperty);
    }

    void Start()
    {
        stride = Mathf.Max(16, strideBytes);
        capacity = Mathf.Max(1, initialCapacity);
        AllocateOrRecreateBuffers(capacity);

        // CUDA 初始化
        Sim_InitCuda();
        Sim_EnableCudaWait(enableGpuOnlyCrossQueueWait ? 1 : 0);

        // 绑定 ping-pong
        Sim_BindPingBuffer(0, pingA.GetNativeBufferPtr(), (uint)stride, (uint)capacity);
        Sim_BindPingBuffer(1, pingB.GetNativeBufferPtr(), (uint)stride, (uint)capacity);

        // 初始化参数（可在此直接调整 solverIters / h / gravity 等，不暴露 Inspector）
        var sp = SimParamsInterop.Default((uint)Mathf.Min(particleCount, capacity), h: 2.0f);
        sp.maxParticles = (uint)capacity;
        Sim_SetParams(ref sp);

        // 初始化模拟器
        Sim_InitSimulator();

        // 零拷贝绑定模拟器位置 ping-pong
        Sim_BindSimulatorExternalPosPingPong();

        // CubeMix 播种
        Sim_SetParticleCount((uint)Mathf.Min(particleCount, capacity));
        Sim_SeedCubeMix((uint)groupCount, 20, 2.0f, 1, 0.05f, 12345u);

        EnsureOrUpdatePaletteBuffer();
        PushPaletteToVfx();

        // 初始化 VFX 缓冲绑定
        if (vfx)
        {
            int readIdx = Sim_GetReadIndex();
            vfx.SetGraphicsBuffer(s_propBuffer, readIdx == 0 ? pingA : pingB);
            vfx.SetUInt(s_propCount, Sim_GetParticleCount());
            if (!usePosWAsGroupId && overrideRadius > 0f)
                vfx.SetFloat(s_propOverrideRadius, overrideRadius);
        }
    }

    void LateUpdate()
    {
        if (!enableSimulation)
            return;

        // 更新参数与数量
        uint targetCount = (uint)Mathf.Min(particleCount, capacity);
        Sim_UpdateDt(Time.deltaTime);
        Sim_UpdateCounts(targetCount, (uint)capacity);
        Sim_SetParticleCount(targetCount);

        // Tick + 渲染事件（单次）
        Sim_Tick();
        GL.IssuePluginEvent(GetRenderEventFunc(), 1);

        // 更新时间
        Sim_SetTime(Time.time);

        // 动态扩容
        if (particleCount > capacity)
        {
            int newCap = Mathf.NextPowerOfTwo(particleCount);
            AllocateOrRecreateBuffers(newCap);
            Sim_BindPingBuffer(0, pingA.GetNativeBufferPtr(), (uint)stride, (uint)capacity);
            Sim_BindPingBuffer(1, pingB.GetNativeBufferPtr(), (uint)stride, (uint)capacity);
            Sim_BindSimulatorExternalPosPingPong();
        }

        EnsureOrUpdatePaletteBuffer();
        PushPaletteToVfx();

        if (vfx)
        {
            int readIdx = Sim_GetReadIndex();
            vfx.SetGraphicsBuffer(s_propBuffer, readIdx == 0 ? pingA : pingB);
            vfx.SetUInt(s_propCount, Sim_GetParticleCount());
            if (!usePosWAsGroupId && overrideRadius > 0f)
                vfx.SetFloat(s_propOverrideRadius, overrideRadius);
        }
    }

    private void AllocateOrRecreateBuffers(int requiredCapacity)
    {
        pingA?.Dispose();
        pingB?.Dispose();
        capacity = requiredCapacity;
        pingA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, capacity, stride);
        pingB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, capacity, stride);
        if (vfx) vfx.SetGraphicsBuffer(s_propBuffer, pingA);
    }

    private void EnsureOrUpdatePaletteBuffer()
    {
        int desired = Mathf.Max(0, groupCount);
        if (desired == 0)
        {
            paletteBuffer?.Dispose();
            paletteBuffer = null;
            return;
        }
        if (paletteBuffer != null && paletteBuffer.count == desired)
        {
            Vector3[] data = new Vector3[desired];
            for (int i = 0; i < desired; ++i)
            {
                Color c = (palette != null && i < palette.Length) ? palette[i] :
                          (palette != null && palette.Length > 0 ? palette[palette.Length - 1] : Color.white);
                data[i] = new Vector3(c.r, c.g, c.b);
            }
            paletteBuffer.SetData(data);
            return;
        }
        paletteBuffer?.Dispose();
        paletteBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, desired, sizeof(float) * 3);
        Vector3[] init = new Vector3[desired];
        for (int i = 0; i < desired; ++i)
        {
            Color c = (palette != null && i < palette.Length) ? palette[i] :
                      (palette != null && palette.Length > 0 ? palette[palette.Length - 1] : Color.white);
            init[i] = new Vector3(c.r, c.g, c.b);
        }
        paletteBuffer.SetData(init);
    }

    private void PushPaletteToVfx()
    {
        if (!vfx) return;
        if (paletteBuffer != null) vfx.SetGraphicsBuffer(s_propPalette, paletteBuffer);
        vfx.SetUInt(s_propGroupCount, (uint)Mathf.Max(0, groupCount));
        vfx.SetUInt(s_propParticlesPerGroup, (uint)Mathf.Max(0, particlesPerGroup));
    }

    void OnDestroy()
    {
        try { Sim_ShutdownCuda(); } catch { }
        try { Sim_ShutdownSimulator(); } catch { }
        paletteBuffer?.Dispose();
        pingA?.Dispose();
        pingB?.Dispose();
    }
}