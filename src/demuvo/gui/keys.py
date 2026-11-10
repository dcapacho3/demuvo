"""Key codes and modifier bits, the numbers GLFW uses.

The toolkit and its tests need no window library: they say `KEY["enter"]`, and
the window passes on what GLFW reports.
"""

KEY = {
    "space": 32,
    "escape": 256,
    "enter": 257,
    "tab": 258,
    "backspace": 259,
    "delete": 261,
    "right": 262,
    "left": 263,
    "down": 264,
    "up": 265,
    "kp_enter": 335,
}
KEY.update({chr(code).lower(): code for code in range(ord("A"), ord("Z") + 1)})
KEY.update({str(digit): ord(str(digit)) for digit in range(10)})
# Modifier bits.
SHIFT, CTRL = 1, 2
