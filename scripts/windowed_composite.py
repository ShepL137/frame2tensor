"""Live demo: composite a captured X11 window with an animated GL overlay.

Captures an X11 window every frame and writes it into a WindowedRenderer's FBO via write(),
then draws an animated translucent marker on top via a real GL draw call (shader, VAO, alpha blending)
before the frame is blitted to screen.
Demonstrates the write()/draw() compositing seam end to end:
a window-capture background with a GPU-drawn overlay, shown live.

With --output,
the same composited frame is also recorded to a video file via WindowedRenderer.capture() and VideoWriter,
so live preview and recording run from the same draw call.

Usage:
    uv run python scripts/windowed_composite.py [WINDOW_ID] [--output FILE] [--fps N]

WINDOW_ID can be decimal or 0x-prefixed hex.
If omitted, the script uses the active window (see `frame2tensor.capture.get_active_window`).
Runs until the window is closed.
Recording stops with a warning if the source resizes, since VideoWriter can't change dimensions mid-stream.
Requires ffmpeg on PATH if --output is given.
"""
import argparse
import math
import struct
import time

import moderngl as mgl

from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.output import VideoWriter
from frame2tensor.render import WindowedRenderer

_MARKER_VERTEX_SHADER = """
#version 330
in vec2 in_position;
uniform vec2 u_offset;
void main() {
    gl_Position = vec4(in_position + u_offset, 0.0, 1.0);
}
"""

_MARKER_FRAGMENT_SHADER = """
#version 330
out vec4 frag_color;
void main() {
    frag_color = vec4(1.0, 0.1, 0.8, 0.75);  // translucent magenta: obvious against most content
}
"""

_MARKER_QUAD = [
    -0.12, -0.08,  0.12, -0.08,  0.12,  0.08,
    -0.12, -0.08,  0.12,  0.08, -0.12,  0.08,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "window_id", nargs="?", type=lambda s: int(s, 0),
        help="X11 window XID (decimal or 0x hex)",
    )
    p.add_argument("--output", default=None, help="Also record the composited output to this video file")
    p.add_argument("--fps", type=int, default=60, help="Nominal recording frame rate (default: 60)")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    window_id = args.window_id if args.window_id is not None else get_active_window()

    with X11Window(window_id) as win:
        print(f"Compositing overlay onto 0x{window_id:x}  {win.width}x{win.height} RGBA uint8")

        renderer = WindowedRenderer(win.width, win.height, title="Windowed Composite - frame2tensor")
        ctx      = renderer.ctx

        marker_program = ctx.program(vertex_shader=_MARKER_VERTEX_SHADER, fragment_shader=_MARKER_FRAGMENT_SHADER)
        marker_vbo     = ctx.buffer(data=struct.pack(f"{len(_MARKER_QUAD)}f", *_MARKER_QUAD))
        marker_vao     = ctx.vertex_array(marker_program, [(marker_vbo, "2f", "in_position")])

        seen_revision = win.revision
        start         = time.perf_counter()

        recorder: VideoWriter | None = None
        if args.output is not None:
            recorder = VideoWriter(args.output, width=win.width, height=win.height, fps=args.fps)
            print(f"Recording composited output to {args.output} ({win.width}x{win.height} @ {args.fps} nominal fps)")

        def draw_overlay(ctx: mgl.Context, _fbo: mgl.Framebuffer) -> None:
            t        = time.perf_counter() - start
            offset_x = math.sin(t) * 0.6
            offset_y = math.cos(t * 1.3) * 0.6
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            marker_program["u_offset"] = (offset_x, offset_y)
            marker_vao.render(mgl.TRIANGLES)
            ctx.disable(mgl.BLEND)

        times  : list[float] = []
        frames : int         = 0
        while not renderer.is_closing:
            win.make_current()
            t0  = time.perf_counter()
            buf = win.capture()

            renderer.make_current()
            if win.revision != seen_revision:
                seen_revision = win.revision
                renderer.resize(win.width, win.height)
                if recorder is not None:
                    print("Source resized; VideoWriter can't change dimensions mid-stream, stopping recording.")
                    recorder.close()
                    recorder = None

            renderer.write(buf)
            renderer.draw(draw_overlay)
            if recorder is not None:
                recorder.write(renderer.capture())
            renderer.swap()
            times.append(time.perf_counter() - t0)

            frames += 1
            if frames % 60 == 0:
                avg = sum(times[-60:]) / 60 * 1000
                print(f"  [{frames:6d} frames]  {avg:.2f} ms/frame  ({1000/avg:.0f} fps)")

        renderer.close()
        if recorder is not None:
            recorder.close()
            print(f"Wrote {frames} composited frames to {args.output}")
        if times:
            avg = sum(times) / len(times) * 1000
            print(f"  {frames} frames total  avg {avg:.2f} ms  ({1000/avg:.0f} fps)")


if __name__ == "__main__":
    main()
