"""Subprocess scenario for the dual-context teardown regression test.

Pairs X11Window (capture) with WindowedRenderer in the natural `with` order,
runs a few frames, then either exits cleanly or raises mid-loop to exercise teardown during exception unwind.
A regression reappears as a SIGSEGV,
which the parent test sees as a negative return code and distinguishes from a clean exit (0) or a normal traceback (1).

Usage:
    python _dual_context_teardown.py {clean|error}
"""
import sys
import time
from typing import Any

from Xlib import X
from Xlib import display as xdisplay

from frame2tensor.capture import X11Window
from frame2tensor.render import WindowedRenderer


def make_target_window() -> tuple[Any, Any]:
    """Map a small visible window for the compositor to redirect, mirroring the x11 test fixture."""
    d      = xdisplay.Display()
    screen = d.screen()
    target = screen.root.create_window(
        x=0, y=0, width=128, height=96, border_width=0,
        depth=screen.root_depth,
        window_class=X.InputOutput,
        visual=X.CopyFromParent,
        background_pixel=screen.white_pixel,
        event_mask=0,
    )
    target.map()
    d.sync()
    time.sleep(0.3)  # let the compositor redirect and render the window before capture
    return d, target


def main(mode: str) -> None:
    d, target = make_target_window()
    try:
        with X11Window(target.id) as src, WindowedRenderer(src.width, src.height) as win:
            for i in range(3):
                buf = src.capture()
                win.write(buf)
                win.draw()
                win.swap()
                if mode == "error" and i == 1:
                    # Mimic a model forward() blowing up mid-loop: the exception must reach
                    # the top level as a traceback, not vanish into a teardown segfault.
                    raise RuntimeError("simulated model failure")
    finally:
        target.unmap()
        target.destroy()
        d.sync()
        d.close()


if __name__ == "__main__":
    main(sys.argv[1])
