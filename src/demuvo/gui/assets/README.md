# Fonts

`Roboto-Regular.ttf` and `Roboto-Bold.ttf` are Roboto, by Google, under the Apache
License 2.0 (`LICENSE.txt`). They are the fallback for a pip install: the program
reads the system's Roboto first (the `fonts-roboto-unhinted` package on Debian and
Ubuntu), and a Debian package drops these copies and depends on that one.

The files came from the `assets/fonts/Roboto` folder of the `imgui-bundle` 1.93.0
wheel from PyPI, which carries the Apache 2.0 licence file for them. Nothing else
from that wheel is used: its Font Awesome, Inconsolata, GUST and Playbox files are
under licences that are not on the project's allowed list.

# Icon

`icon_32.png`, `icon_64.png`, `icon_128.png` and `icon_180.png` are the demuvo logo
tile, made for this project (`brand/demuvo/` in the repository) and under the
project's Apache License 2.0. The window shows them as its icon. They are saved as
PNG files that `imgio.read_png` can read.
