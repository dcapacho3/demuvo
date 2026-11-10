"""Everything that draws or opens a window.

A world and a run need none of it, so nothing outside this package imports it, and
`import demuvo` never loads it. It is written without a picture or window toolkit:
pictures are drawn into numpy arrays, and the window comes from the system's GLFW
and OpenGL libraries.
"""
