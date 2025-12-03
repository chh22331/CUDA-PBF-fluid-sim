from pathlib import Path
path = Path(r'engine/gfx/d3d12_shaders/points_speed.hlsl')
print('size', path.stat().st_size)
print(path.read_text(encoding='utf-8', errors='ignore')[:400])
