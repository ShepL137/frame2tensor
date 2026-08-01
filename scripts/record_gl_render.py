"""Record synthetic EGL-rendered content to a video file.

Proves the generated-content recording path end to end:
EGLCanvas.capture() -> CUDABuffer -> VideoWriter -> ffmpeg -> mp4.

The scene is intentionally vertically asymmetric (red top, blue bottom, bouncing marker)
so an orientation bug is visually obvious in the output.

There is no live source to keep up with here, so frames are rendered and encoded as fast as the pipeline allows.

Usage:
    uv run python scripts/record_gl_render.py [--duration SECONDS] [--fps F] [--output FILE]

Requires ffmpeg on PATH.
"""
import argparse
import math
import struct
import time

import moderngl as mgl

from frame2tensor.output import VideoWriter
from frame2tensor.render import EGLCanvas

_GRADIENT_VERTEX_SHADER = """
#version 330
in vec2 in_position;
out float v_y;
void main() {
    v_y         = in_position.y;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_GRADIENT_FRAGMENT_SHADER = """
#version 330
in float v_y;
out vec4 frag_color;
void main() {
    float t           = (v_y + 1.0) * 0.5;      // 0 at bottom, 1 at top
    vec3 top_color    = vec3(0.9, 0.15, 0.15);  // red
    vec3 bottom_color = vec3(0.15, 0.25, 0.9);  // blue
    frag_color        = vec4(mix(bottom_color, top_color, t), 1.0);
}
"""

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
    frag_color = vec4(1.0, 1.0, 0.2, 1.0);  // yellow
}
"""

_FULLSCREEN_QUAD = [
    -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,
]

_MARKER_QUAD = [
    -0.08, -0.05,  0.08, -0.05,  0.08,  0.05,
    -0.08, -0.05,  0.08,  0.05, -0.08,  0.05,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--width", type=int, default=1280, help="Frame width in pixels (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Frame height in pixels (default: 720)")
    p.add_argument("--duration", type=float, default=10.0, help="Length of generated video in seconds (default: 10)")
    p.add_argument("--fps", type=int, default=60, help="Output frame rate; drives the animation clock (default: 60)")
    p.add_argument("--output", default="capture.mp4", help="Output video path (default: capture.mp4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_frames = round(args.duration * args.fps)

    with EGLCanvas(args.width, args.height) as canvas:
        ctx              = canvas.ctx

        gradient_program = ctx.program(vertex_shader=_GRADIENT_VERTEX_SHADER, fragment_shader=_GRADIENT_FRAGMENT_SHADER)
        marker_program   = ctx.program(vertex_shader=_MARKER_VERTEX_SHADER, fragment_shader=_MARKER_FRAGMENT_SHADER)

        gradient_vbo     = ctx.buffer(data=struct.pack(f"{len(_FULLSCREEN_QUAD)}f", *_FULLSCREEN_QUAD))
        gradient_vao     = ctx.vertex_array(gradient_program, [(gradient_vbo, "2f", "in_position")])

        marker_vbo       = ctx.buffer(data=struct.pack(f"{len(_MARKER_QUAD)}f", *_MARKER_QUAD))
        marker_vao       = ctx.vertex_array(marker_program, [(marker_vbo, "2f", "in_position")])

        current_frame = 0

        def draw_frame(ctx: mgl.Context, _fbo: mgl.Framebuffer) -> None:
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            gradient_vao.render(mgl.TRIANGLES)
            t        = current_frame / args.fps
            offset_y = math.sin(t) * 0.7
            marker_program["u_offset"] = (0.0, offset_y)
            marker_vao.render(mgl.TRIANGLES)

        print(
            f"Recording synthetic {args.width}x{args.height} -> {args.output}  "
            f"({total_frames} frames, {args.duration:.1f}s @ {args.fps} fps)",
        )

        start = time.perf_counter()
        with VideoWriter(args.output, width=args.width, height=args.height, fps=args.fps) as rec:
            for _current_frame in range(total_frames):
                canvas.draw(draw_frame)
                rec.write(canvas.capture())
        elapsed = time.perf_counter() - start

        ms_per_frame = elapsed / total_frames * 1000 if total_frames > 0 else math.inf
        rtf          = elapsed / args.duration if args.duration > 0 else math.inf
        speedup      = args.duration / elapsed if elapsed > 0 else math.inf
        print(
            f"Wrote {total_frames} frames to {args.output} in {elapsed:.2f}s  "
            f"({ms_per_frame:.2f} ms/frame, RTF {rtf:.3f}, {speedup:.1f}x realtime)",
        )


if __name__ == "__main__":
    main()
