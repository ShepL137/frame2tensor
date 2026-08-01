"""Record a raw X11 window to a video file.

Proves the capture to output pipeline end to end:
X11Window.capture() -> CUDABuffer -> VideoWriter -> ffmpeg -> mp4.

X11Window frames are top-down (row 0 is the visual top), so no vertical flip is needed.

Usage:
    uv run python scripts/record_x11_window.py [WINDOW_ID] [--frames N] [--fps F] [--output FILE]

WINDOW_ID can be decimal or 0x-prefixed hex.
If omitted, the script uses the active window (see `frame2tensor.capture.get_active_window`).
Requires ffmpeg on PATH.
"""
import argparse
import signal
import sys
import time

from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.output import VideoWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "window_id", nargs="?", type=lambda s: int(s, 0),
        help="X11 window XID (decimal or 0x hex)",
    )
    p.add_argument("--frames", type=int, default=None, help="Stop after N frames (default: record until Ctrl-C)")
    p.add_argument("--fps", type=int, default=60, help="Target capture and output frame rate (default: 60)")
    p.add_argument("--output", default="capture.mp4", help="Output video path (default: capture.mp4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    window_id = args.window_id if args.window_id is not None else get_active_window()

    with X11Window(window_id) as win:
        width, height = win.width, win.height
        limit = f"{args.frames} frames" if args.frames is not None else "until Ctrl-C"
        print(f"Recording 0x{window_id:x}  {width}x{height} -> {args.output}  ({limit} @ {args.fps} fps)")

        seen_revision  = win.revision
        frame_interval = 1.0 / args.fps
        next_frame     = time.perf_counter()
        recorded       = 0
        stop           = False

        def _on_sigint(_sig: int, _frame: object) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _on_sigint)

        with VideoWriter(args.output, width=width, height=height, fps=args.fps) as rec:
            while not stop and (args.frames is None or recorded < args.frames):
                buf = win.capture()
                if win.revision != seen_revision:
                    print("Source window resized; stopping (fixed-geometry recording).", file=sys.stderr)
                    break
                rec.write(buf)
                recorded += 1

                next_frame += frame_interval
                delay       = next_frame - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

        if stop:
            print("\nStopping.", file=sys.stderr)
        print(f"Wrote {recorded} frames to {args.output}")


if __name__ == "__main__":
    main()
