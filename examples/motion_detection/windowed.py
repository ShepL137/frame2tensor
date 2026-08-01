"""Live motion detection on a captured X11 window.

Captures a window every frame, runs a small PyTorch frame-differencing filter on it, and renders to a second window.

Runs until the display window is closed.
"""
import argparse
import time

import torch
from _detectors import MODELS

from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.render import WindowedRenderer


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


class FpsReporter:
    """Once per second, rewrite the terminal status line with the measured frame rate."""

    def __init__(self):
        self.frame_count = 0
        self.clock       = time.perf_counter()

    def tick(self):
        self.frame_count += 1

        now = time.perf_counter()
        if now - self.clock >= 1.0:
            print(f"\r{self.frame_count / (now - self.clock):.0f} fps   ", end="", flush=True)
            self.clock       = now
            self.frame_count = 0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "window_id", nargs="?", type=lambda s: int(s, 0),
        help="X11 window XID (decimal or 0x hex); defaults to the active window",
    )
    p.add_argument(
        "--model", choices=MODELS, default="ema",
        help="Detector: ema differences against a moving-average background,"
             " adaptive z-scores each pixel (default: ema)",
    )
    p.add_argument(
        "--fps", type=int, default=30,
        help="Frame limiter; match the source's real update rate. 0 for vsync, negative to disable (default: 30)",
    )

    return p.parse_args()


def main():
    args      = parse_args()
    window_id = args.window_id if args.window_id is not None else get_active_window()
    vsync     = args.fps == 0

    with (
        X11Window(window_id) as src,
        WindowedRenderer(src.width, src.height, title="Motion Detection - frame2tensor", vsync=vsync) as win,
    ):
        print(f"Detecting motion ({args.model}) on 0x{window_id:x}  {src.width}x{src.height}")

        model         = MODELS[args.model](src.height, src.width)
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
                    model = MODELS[args.model](src.height, src.width)

                frame = torch.as_tensor(buf, device="cuda").float() / 255.0
                win.write(model(frame))
                win.draw()
                win.swap()

                limit_framerate(t0, args.fps)
                fps_report.tick()
            print()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
