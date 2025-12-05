# PBF-X / 鍩轰簬浣嶇疆鐨勬祦浣撴ā鎷?

PBF-X is a Windows-focused Position-Based Fluids sandbox that pairs a lightweight Direct3D 12 engine with CUDA-accelerated simulation kernels.  
PBF-X 鏄竴涓潰鍚?Windows 鐨勫熀浜庝綅缃殑娴佷綋娌欑洅锛岀粨鍚堜簡杞婚噺绾х殑 Direct3D 12 寮曟搸涓?CUDA 鍔犻€熺殑浠跨湡鍐呮牳銆? 
This is a just-for-fun project; everyone is welcome to explore or chat, but hands-on technical support might not always be available.  
鏈」鐩彧鏄釜浜虹殑鍏磋叮鍒涗綔锛屾杩庢墍鏈変汉鍓嶆潵浜ゆ祦锛屼笉杩囨棤娉曚繚璇佽兘澶熼殢鏃舵彁渚涙妧鏈敮鎸併€?

## Overview / 椤圭洰绠€浠?
- Core gameplay-independent systems (configuration, console, renderer, interop helpers) live under `engine/`.  
  鏍稿績鐨勯€氱敤绯荤粺锛堥厤缃€佹帶鍒跺彴銆佹覆鏌撱€佷簰鎿嶄綔宸ュ叿锛変綅浜?`engine/`銆?
- Simulation logic plus CUDA kernels are kept in `sim/`, covering graph construction, phase scheduling, and GPU kernels such as `pbf_lambda.cu`.  
  浠跨湡閫昏緫涓?CUDA 鍐呮牳闆嗕腑鍦?`sim/`锛屽寘鍚浘鏋勫缓銆侀樁娈佃皟搴︿互鍙?`pbf_lambda.cu` 绛?GPU 鍐呮牳銆?

## Project Layout / 仓库结构
- engine/core/, engine/gfx/, engine/interop/ — platform layer, configuration, D3D12 renderer, and bridge helpers.
  engine/core/、engine/gfx/、engine/interop/ — 平台层、配置、D3D12 渲染器与桥接工具。
- sim/ — CUDA + CPU simulation stages (graph builder, kernels, statistics).
  sim/ — CUDA 与 CPU 混合的仿真阶段（图构建、内核、统计）。
- uild/ — CMake/Ninja outputs: executables in uild/bin[/<config>], libraries in uild/lib[/<config>].
  uild/ —— CMake/Ninja 输出，可执行位于 uild/bin[/<config>]，静态库位于 uild/lib[/<config>]。
- x64/ — Visual Studio outputs per configuration (e.g., x64/Release/PBF-X.exe).
  x64/ —— Visual Studio 各配置输出目录（如 x64/Release/PBF-X.exe）。## Prerequisites / 鐜渚濊禆
- Windows 10/11 with a GPU that supports CUDA SM 75+.  
  杩愯 Windows 10/11锛孏PU 闇€鏀寔 CUDA SM 75 鍙婁互涓娿€?
- CMake 鈮?3.20 and Ninja (default generator).  
  CMake 3.20 鍙婁互涓婄増鏈拰 Ninja锛堥粯璁ょ敓鎴愬櫒锛夈€?
- Visual Studio 2022 (Desktop development with C++) or clang-cl + lld-link toolchain.  
  Visual Studio 2022锛堝惈 C++ 妗岄潰寮€鍙戠粍浠讹級鎴?clang-cl + lld-link 宸ュ叿閾俱€?
- NVIDIA CUDA Toolkit (matching your driver, 12.x recommended).  
  NVIDIA CUDA Toolkit锛堜笌椹卞姩鍖归厤锛屾帹鑽?12.x锛夈€?
- Latest Windows SDK + DirectX 12 runtime.  
  鏈€鏂?Windows SDK 涓?DirectX 12 杩愯搴撱€?

## Configure & Build / 配置与构建
### CMake + Ninja
1. Configure with CUDA enabled (default ON):
   cmake -S . -B build -DPBFX_ENABLE_CUDA=ON
   启用 CUDA（默认 ON）执行配置：cmake -S . -B build -DPBFX_ENABLE_CUDA=ON
2. Build the Release binary:
   cmake --build build --config Release
   编译 Release 版本：cmake --build build --config Release
   - 产物路径：可执行位于 uild/bin/Release/（或单配置生成器的 uild/bin/），库位于 uild/lib/Release/（或 uild/lib/）。
### Visual Studio Workflow / 浣跨敤 Visual Studio
1. Open `PBF-X.sln` or generate via CMake GUI with the Visual Studio generator.  
   鎵撳紑 `PBF-X.sln` 鎴栦娇鐢?CMake GUI 鐢熸垚 VS 瑙ｅ喅鏂规銆?
2. Ensure `PBFX_ENABLE_CUDA` stays checked so CUDA kernels compile; toggle only if you lack a CUDA-capable GPU.  
   璇蜂繚鎸?`PBFX_ENABLE_CUDA` 閫変腑锛屼互缂栬瘧 CUDA 鍐呮牳锛涗粎鍦ㄦ病鏈?CUDA GPU 鏃跺叧闂€?
3. Select the `Release` x64 configuration and build `pbf_app`.  
   閫夋嫨 x64 `Release` 閰嶇疆骞剁敓鎴?`pbf_app`銆?

### Build Options / 鏋勫缓閫夐」
- `PBFX_ENABLE_CUDA` (ON/OFF): Enables CUDA targets, injects CUDA include paths into `pbf_core`, and selects SM architectures 75/80/86/89/90.  
  `PBFX_ENABLE_CUDA`锛堝紑/鍏筹級锛氬惎鐢?CUDA 鐩爣銆佷负 `pbf_core` 娣诲姞 CUDA 澶磋矾寰勶紝骞跺浐瀹?SM 鏋舵瀯涓?75/80/86/89/90銆?
- Override CUDA architectures if needed:  
  `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="89;90"`  
  鍙嚜瀹氫箟 CUDA 鏋舵瀯锛歚cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="89;90"`

## Run / 运行
- Standalone executable (CMake/Ninja): .\build\bin\pbf_app.exe（多配置生成器为 .\build\bin\Release\pbf_app.exe）。
  Ninja 构建后直接运行：.\build\bin\pbf_app.exe（或 .\build\bin\Release\pbf_app.exe）。
- Visual Studio output: .\x64\Release\PBF-X.exe
  Visual Studio 构建产物：.\x64\Release\PBF-X.exe
- Launch with a GPU connected display for the D3D12 swap chain; use the in-app console to adjust runtime parameters.
  启动时需连接 GPU 显示输出以创建 D3D12 交换链；可使用应用内控制台调整运行时参数。
## Runtime Console / 杩愯鏃舵帶鍒跺彴
- Editable defaults live in `engine/core/console.h` under `console::RuntimeConsole`; tweaking fields like `sim.h_over_r = 2.5f`, `sim.solverIters`, or `renderer.particleRadiusPx` lets you script new presets.  
  `engine/core/console.h` 涓殑 `console::RuntimeConsole` 闆嗕腑浜嗘墍鏈夐粯璁ゅ弬鏁帮紝鍙皟鏁?`sim.h_over_r = 2.5f`銆乣sim.solverIters`銆乣renderer.particleRadiusPx` 绛夊瓧娈典互瀹氫箟鏂扮殑棰勮銆?
- Helper functions (`console::BuildSimParams`, `console::ApplyRendererRuntime`, etc.) propagate these values into `sim::SimParams`, `sim::DeviceParams`, and `gfx::RenderInitParams` at startup or during hot reload.  
  鍚姩鎴栫儹閲嶈浇鏃讹紝`console::BuildSimParams`銆乣console::ApplyRendererRuntime` 绛夎緟鍔╁嚱鏁颁細鎶婅繖浜涙暟鍊兼敞鍏?`sim::SimParams`銆乣sim::DeviceParams` 涓?`gfx::RenderInitParams`銆?
- Pairing these structs with your own UI/CLI makes it easy to expose sliders, config files, or scripting hooks without hunting through scattered globals.  
  閰嶅悎鑷畾涔?UI/CLI锛屽彲浠ユ柟渚垮湴鎻愪緵婊戞潌銆侀厤缃枃浠舵垨鑴氭湰鎺ュ彛锛岃€屾棤闇€鍦ㄩ浂鏁ｇ殑鍏ㄥ眬鍙橀噺闂存煡鎵俱€?

## Development Notes / 寮€鍙戞敞鎰忎簨椤?
- Language standards: C++17 and CUDA C++ for engine/sim code; use 4-space indentation, braces on the same line.  
  璇█瑙勮寖锛氬紩鎿?浠跨湡浠ｇ爜浣跨敤 C++17 涓?CUDA C++锛涚缉杩涗负 4 绌烘牸锛屽乏鎷彿涓庤鍙ュ悓涓€琛屻€?
- Naming: Types use PascalCase, functions camelCase, members prefixed with `m_`, globals with `g_`.  
  鍛藉悕锛氱被鍨?PascalCase锛屽嚱鏁?camelCase锛屾垚鍛樺姞 `m_` 鍓嶇紑锛屽叏灞€鍙橀噺鍔?`g_` 鍓嶇紑銆?
- Keep headers focused (`sim/`, `engine/core/`) and explicitly separate CPU/GPU data paths (see `device_buffers.cuh`).  
  澶存枃浠朵繚鎸佺簿绠€锛坄sim/`銆乣engine/core/`锛夛紝骞舵樉寮忓尯鍒?CPU/GPU 鏁版嵁璺緞锛堝弬鑰?`device_buffers.cuh`锛夈€?

## Validation / 楠岃瘉
- There is no automated test suite yet; validate by running `pbf_app` and exercising key scenarios (camera flight, UI toggles, simulation resets).  
  褰撳墠灏氭棤鑷姩鍖栨祴璇曪紱璇疯繍琛?`pbf_app`锛岃鐩栧叧閿満鏅紙鐩告満椋炶銆乁I 寮€鍏炽€佷豢鐪熼噸缃級銆?
- Compare HUD stats (particle count, FPS, solver iterations) before/after your change to ensure regressions are caught early.  
  淇敼鍓嶅悗瀵规瘮 HUD 缁熻锛堢矑瀛愭暟閲忋€丗PS銆佹眰瑙ｈ凯浠ｏ級浠ュ敖鏃╁彂鐜板洖褰掋€?
- If you add tests, register them via `add_test` so `ctest` can execute them in CI later.  
  濡傞渶鏂板娴嬭瘯锛岃閫氳繃 `add_test` 娉ㄥ唽锛屼究浜庢湭鏉ヤ娇鐢?`ctest` 杩愯銆?

## Performance Snapshot / 鎬ц兘蹇収
- On an RTX 4060 Laptop GPU, Release builds running two solver iterations with a smoothing ratio `h_over_r = 2.5` (smoothing radius 鈮?2.5脳 particle radius) sustain roughly 1,000,000 particles at ~60 FPS.  
  鍦?RTX 4060 Laptop GPU 涓婏紝Release 鏋勫缓銆佷袱娆℃眰瑙ｈ凯浠ｅ苟璁剧疆 `h_over_r = 2.5`锛堝钩婊戝崐寰勭害绛変簬绮掑瓙鍗婂緞鐨?2.5 鍊嶏級鏃讹紝澶х害鍙互缁存寔 1,000,000 绮掑瓙骞朵繚鎸?60 FPS 宸﹀彸銆?

## Troubleshooting / 甯歌闂
- Missing CUDA runtime headers usually indicates `PBFX_ENABLE_CUDA` was disabled or the CUDA Toolkit path is not visible to CMake. Verify `find_package(CUDAToolkit REQUIRED)` succeeds during configuration.  
  鎵句笉鍒?CUDA 澶存枃浠堕€氬父鏄?`PBFX_ENABLE_CUDA` 琚叧闂垨 CUDA Toolkit 璺緞涓嶅彲瑙侊紱璇风‘淇濋厤缃樁娈?`find_package(CUDAToolkit REQUIRED)` 鎴愬姛銆?
- Linker errors with clang-cl on Windows can be resolved by keeping the provided `/INCREMENTAL:NO` flags and ensuring lld-link is selected.  
  鍦?Windows 涓婁娇鐢?clang-cl 鍑虹幇閾炬帴閿欒鏃讹紝璇蜂繚鐣?`/INCREMENTAL:NO` 绛夋彁渚涚殑鏍囧織骞剁‘璁ら€夋嫨浜?lld-link銆?
- D3D12 initialization failures often stem from running over Remote Desktop without a hardware adapter鈥攍aunch locally or use a GPU-capable remote solution.  
  D3D12 鍒濆鍖栧け璐ュ鍗婂洜涓鸿繙绋嬫闈㈢己灏戠‖浠堕€傞厤鍣ㄢ€斺€旇鍦ㄦ湰鍦版垨鍏峰 GPU 鏀寔鐨勮繙绋嬬幆澧冭繍琛屻€?



