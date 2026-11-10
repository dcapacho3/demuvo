# Fonts

`Roboto-Regular.ttf` and `Roboto-Bold.ttf` are Roboto, by Google, under the Apache
License 2.0 (`LICENSE.txt`). They are the fallback for a pip install: the program
reads the system's Roboto first (the `fonts-roboto-unhinted` package on Debian and
Ubuntu), and a Debian package drops these copies and depends on that one.

The files came from the `assets/fonts/Roboto` folder of the `imgui-bundle` 1.93.0
wheel from PyPI, which carries the Apache 2.0 licence file for them. Nothing else
from that wheel is used: its Font Awesome, Inconsolata, GUST and Playbox files are
under licences that are not on the project's allowed list.
