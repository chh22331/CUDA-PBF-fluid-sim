using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.VFX;

public class SimBridge : MonoBehaviour
{
    [DllImport("NativeSimPlugin")] static extern IntPtr GetRenderEventFunc();
    [DllImport("NativeSimPlugin")] static extern uint GetParticleCount();
    [DllImport("NativeSimPlugin")] static extern uint Gfx_GetParticleStride();      // 直接复用 renderer 导出（若需要可再通过插件转发）
    [DllImport("NativeSimPlugin")] static extern uint Gfx_GetParticleCapacity();
    [DllImport("NativeSimPlugin")] static extern void SetUnityTargetBuffer(IntPtr nativeResPtr, uint strideBytes, uint capacityElements);
    [DllImport("NativeSimPlugin")] static extern void SetCopyMode(int mode); // 0直接SRV,1拷贝
    [DllImport("NativeSimPlugin")] static extern bool Sim_InitCuda();
    [DllImport("NativeSimPlugin")] static extern void Sim_ShutdownCuda();

    public VisualEffect vfx;
    public string particleCountProperty = "ParticleCount";
    public string particleBufferProperty = "ParticleBuffer";

    GraphicsBuffer _gb;
    uint _stride;
    uint _capacity;

    void Start()
    {
        if (!Sim_InitCuda())
        {
            Debug.LogError("CUDA init failed.");
        }
        _stride = Gfx_GetParticleStride();
        if (_stride == 0) _stride = 48; // 回退：假设 float4*3 (可根据你的 Particle 结构调整)
        _capacity = Gfx_GetParticleCapacity();
        if (_capacity == 0) _capacity = 1024;

        _gb = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)_capacity, (int)_stride);
        vfx.SetGraphicsBuffer(particleBufferProperty, _gb);

        IntPtr nativePtr = _gb.GetNativeBufferPtr();
        SetUnityTargetBuffer(nativePtr, _stride, _capacity);
        SetCopyMode(1); // 初始使用 Copy 模式；后续切换为 0 测试无拷贝路径
    }

    void Update()
    {
        GL.IssuePluginEvent(GetRenderEventFunc(), 1);
        uint count = GetParticleCount();
        vfx.SetUInt(particleCountProperty, count);
    }

    void OnDestroy()
    {
        _gb?.Dispose();
        Sim_ShutdownCuda();
    }
}