import pathlib
amp = chr(38)
path = pathlib.Path('app_main.cpp')
data = path.read_text(encoding='mbcs')
