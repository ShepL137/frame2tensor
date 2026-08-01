"""Capture frames from an X11 window as GPU buffers.

Usage:
    uv run python scripts/capture_x11_window.py [WINDOW_ID]

A WindowedRenderer displays the live capture.
"""
import argparse

from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.render import WindowedRenderer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "window_id", nargs="?", type=lambda s: int(s, 0),
        help="X11 window XID (decimal or 0x hex)",
    )
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    window_id = args.window_id if args.window_id is not None else get_active_window()

    with X11Window(window_id) as win:
        print(f"Capturing 0x{window_id:x}  {win.width}x{win.height} RGBA uint8")

        renderer      = WindowedRenderer(win.width, win.height, title="X11 Capture - frame2tensor")
        seen_revision = win.revision

        while not renderer.is_closing:
            buf = win.capture()

            if win.revision != seen_revision:
                seen_revision = win.revision
                renderer.resize(win.width, win.height)

            renderer.write(buf)
            renderer.draw()
            renderer.swap()

        renderer.close()


if __name__ == "__main__":
    main()
