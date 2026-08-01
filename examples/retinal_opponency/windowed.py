"""Live retinal color opponency on a captured X11 window.

Captures a window every frame, runs a spatial DoG filter, and renders the result to a second window.

Tab cycles views; 1-4 select directly; Esc or Q quits.
"""
import argparse
import time

import glfw
import torch
from _filters import MODES, RetinalOpponency, render_mode

from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.render import WindowedRenderer


class FpsReporter:
    """Once per second, rewrite the terminal status line with the measured frame rate and the active view."""

    def __init__(self):
        self.frame_count = 0
        self.clock       = time.perf_counter()

    def tick(self, mode):
        self.frame_count += 1

        now = time.perf_counter()
        if now - self.clock >= 1.0:
            print(f"\r{self.frame_count / (now - self.clock):.0f} fps  [{mode}]   ", end="", flush=True)
            self.clock       = now
            self.frame_count = 0


def limit_framerate(t0, fps):
    """Limit framerate to `fps`.

    Args:
        t0 : Frame start.
        fps: Frames per second. 0 for vsync, negative to disable.
    """
    if fps <= 0:
        return

    budget = 1.0 / fps - (time.perf_counter() - t0)
    if budget > 0:
        time.sleep(budget)


def poll_input(handle, mode, key_state):
    """Poll the keyboard and return the (possibly changed) mode and whether to quit.

    Fires on key press.
    Call after swap(), which pumps GLFW's event queue.
    """
    def pressed(key):
        down = glfw.get_key(window=handle, key=key) == glfw.PRESS
        edge = down and not key_state.get(key, False)
        key_state[key] = down

        return edge

    if pressed(glfw.KEY_TAB):
        mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
    for offset, name in enumerate(MODES):
        if pressed(glfw.KEY_1 + offset):
            mode = name

    quit_requested = pressed(glfw.KEY_ESCAPE) | pressed(glfw.KEY_Q)

    return mode, quit_requested


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "window_id", nargs="?", type=lambda s: int(s, 0),
        help="X11 window XID (decimal or 0x hex); defaults to the active window",
    )
    p.add_argument("--mode", choices=MODES, default=MODES[0], help=f"Initial view (default: {MODES[0]})")
    p.add_argument(
        "--fps", type=int, default=60,
        help="Frame limiter; 0 for vsync, negative to disable (default: 60)",
    )

    return p.parse_args()


def main():
    args      = parse_args()
    window_id = args.window_id if args.window_id is not None else get_active_window()
    vsync     = args.fps == 0
    mode      = args.mode

    with (
        X11Window(window_id) as src,
        WindowedRenderer(src.width, src.height, title="Retinal Opponency - frame2tensor", vsync=vsync) as win,
    ):
        print(f"Retinal opponency on 0x{window_id:x}  {src.width}x{src.height}")
        print("Keys: Tab cycles views, 1-4 select, Esc or Q quits")

        model         = RetinalOpponency().to("cuda")
        win.make_current()
        handle        = glfw.get_current_context()
        key_state     = {}
        seen_revision = src.revision
        fps_report    = FpsReporter()

        try:
            while not win.is_closing:
                t0  = time.perf_counter()
                buf = src.capture()

                win.make_current()
                if src.revision != seen_revision:
                    seen_revision = src.revision
                    win.resize(src.width, src.height)

                frame    = torch.as_tensor(buf, device="cuda").float() / 255.0
                opponent = model(frame)
                win.write(render_mode(opponent, mode))
                win.draw()
                win.swap()

                mode, quit_requested = poll_input(handle, mode, key_state)
                if quit_requested:
                    break

                limit_framerate(t0, args.fps)
                fps_report.tick(mode)
            print()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
