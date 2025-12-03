import re

with open("renderer.cpp", "r", encoding="utf-8") as f:
    content = f.read()

# 压缩连续空行成一行
fixed = re.sub(r'\n{2,}', '\n', content)

with open("renderer_fixed.txt", "w", encoding="utf-8") as f:
    f.write(fixed)
