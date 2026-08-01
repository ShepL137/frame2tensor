"""Benchmark the frame2tensor pipeline across all routes and resolutions.

Routes:
  egl_render     EGLCanvas.draw() + capture()
  egl_composite  write() + draw() + capture()
  egl_record     draw() + capture() + VideoWriter.write()
  x11_capture    X11Window.capture() (requires --x11 XID)

Every route runs by default; --skip drops the ones you do not want.

Each route reports per-stage and full-pipeline timings (mean, p50, p95, p99, fps),
aggregated over --frames frames after --warmup discard frames.

Results are written to data/ as JSON and summarised to stdout.

Usage:
    uv run python benchmarks/benchmark_pipeline.py
    uv run python benchmarks/benchmark_pipeline.py --skip egl_record
    uv run python benchmarks/benchmark_pipeline.py --x11 0x... --tensor-op
    uv run python benchmarks/benchmark_pipeline.py --resolutions 1920x1080,3840x2160
"""
import argparse
import datetime
import importlib
import platform
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import NamedTuple, NotRequired, TypedDict, cast

import cuda.bindings.runtime as cudart
import moderngl as mgl
import orjson
from OpenGL import GL

from frame2tensor.capture import get_active_window
from frame2tensor.exceptions import WindowNotFoundError
from frame2tensor.output import VideoWriter
from frame2tensor.render import EGLCanvas
from frame2tensor.types import CUDABuffer

# Typing pedantry

def _try_import_torch() -> bool:
    try:
        importlib.import_module("torch")
    except ImportError:
        return False
    return True


def _try_import_capture() -> bool:
    try:
        importlib.import_module("frame2tensor.capture")
    except Exception:
        return False
    return True


_TORCH_OK = _try_import_torch()
_X11_OK   = _try_import_capture()

EGL_ROUTES = ("egl_render", "egl_composite", "egl_record")
ALL_ROUTES = ("x11_capture", *EGL_ROUTES)


# -----------------------------------------------------------------------------
# Scene shaders (gradient: red top, blue bottom; exercises every fragment)
# -----------------------------------------------------------------------------

_GRAD_VERT = """
#version 330
in vec2 in_position;
out float v_y;
void main() {
    v_y         = in_position.y;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_GRAD_FRAG = """
#version 330
in float v_y;
out vec4 frag_color;
void main() {
    float t    = (v_y + 1.0) * 0.5;
    frag_color = vec4(t, 0.15, 1.0 - t, 1.0);
}
"""

_QUAD = [-1.0, -1.0,  1.0, -1.0,  1.0,  1.0,
         -1.0, -1.0,  1.0,  1.0, -1.0,  1.0]


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Timing:
    """Aggregate frame timings for one stage at one resolution.

    `fps` is derived from the mean, so it is a sustained-throughput figure that carries the tail.
    A handful of slow frames routinely puts the mean above p95, which is why p99 is reported:
    without it the gap between p95 and the mean has no visible cause.
    """
    mean_ms: float
    p50_ms : float
    p95_ms : float
    p99_ms : float
    fps    : float


# Stage name to its timings, in the order the route measured them.
StageTimings = dict[str, Timing]

# One stage's per-frame work, timed as a unit.
StageBody = Callable[[], object]


class _RouteResultJSON(TypedDict):
    """Wire shape of one `results` entry in the benchmark's JSON payload."""

    route     : str
    resolution: list[int]
    stages    : dict[str, dict[str, float]]
    xid       : NotRequired[int | None]


@dataclass(frozen=True, slots=True)
class RouteResult:
    """One route's timings at one input size, as written to the JSON payload.

    Every route records the `[width, height]` it captured or rendered.
    The x11 route additionally records the XID it captured from.
    """
    route     : str
    resolution: list[int]
    stages    : StageTimings
    xid       : int | None = None

    @classmethod
    def from_json(cls, entry: object) -> "RouteResult":
        """Rebuild a result read back from a subprocess's JSON slice."""
        data = cast("_RouteResultJSON", entry)
        return cls(
            route      = data["route"],
            resolution = data["resolution"],
            xid        = data.get("xid"),
            stages     = {stage: Timing(**fields) for stage, fields in data["stages"].items()},
        )


def _percentile(sorted_ms: list[float], fraction: float) -> float:
    return sorted_ms[min(int(len(sorted_ms) * fraction), len(sorted_ms) - 1)]


def _stats(times_s: list[float]) -> Timing:
    """Reduce per-frame durations to mean, median, p95, p99, and the implied frame rate."""
    ms   = sorted(t * 1000.0 for t in times_s)
    n    = len(ms)
    mean = sum(ms) / n

    return Timing(
        mean_ms = round(mean, 3),
        p50_ms  = round(ms[n // 2], 3),
        p95_ms  = round(_percentile(ms, 0.95), 3),
        p99_ms  = round(_percentile(ms, 0.99), 3),
        fps     = round(1000.0 / mean, 1) if mean > 0.0 else float("inf"),
    )


def _time_stage(work: StageBody, warmup: int, frames: int) -> Timing:
    """Run a stage to steady state, then time it with a CUDA sync on every frame."""
    for _ in range(warmup):
        work()
    cudart.cudaDeviceSynchronize()

    times: list[float] = []
    for _ in range(frames):
        t0 = perf_counter()
        work()
        cudart.cudaDeviceSynchronize()
        times.append(perf_counter() - t0)

    return _stats(times)


# -----------------------------------------------------------------------------
# Scene setup helpers
# -----------------------------------------------------------------------------

def _build_scene(ctx: mgl.Context) -> tuple[mgl.Program, mgl.VertexArray, mgl.Buffer]:
    program = ctx.program(vertex_shader=_GRAD_VERT, fragment_shader=_GRAD_FRAG)
    vbo     = ctx.buffer(data=struct.pack(f"{len(_QUAD)}f", *_QUAD))
    vao     = ctx.vertex_array(program, [(vbo, "2f", "in_position")])

    return program, vao, vbo


def _release_scene(program: mgl.Program, vao: mgl.VertexArray, vbo: mgl.Buffer) -> None:
    vao.release()
    vbo.release()
    program.release()


def _tensor_reduce(buffer: CUDABuffer) -> None:
    """Stand in for a model's first touch of the frame: wrap, widen, and reduce on the GPU."""
    import torch

    torch.as_tensor(buffer, device="cuda").float().mean().item()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

def bench_egl_render(
    width    : int,
    height   : int,
    warmup   : int,
    frames   : int,
    tensor_op: bool,
) -> StageTimings:
    stages: StageTimings = {}

    with EGLCanvas(width, height) as canvas:
        program, vao, vbo = _build_scene(canvas.ctx)

        def draw(ctx: mgl.Context, _fbo: mgl.Framebuffer) -> None:
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(mgl.TRIANGLES)

        def render_only() -> None:
            canvas.draw(draw)
            GL.glFinish()

        def full() -> None:
            canvas.draw(draw)
            canvas.capture()

        stages["render_only"] = _time_stage(render_only, warmup, frames)

        # readback_only: capture() on an already-rendered FBO
        canvas.draw(draw)
        stages["readback_only"] = _time_stage(canvas.capture, warmup, frames)
        stages["full"]          = _time_stage(full, warmup, frames)

        if tensor_op and _TORCH_OK:
            def full_with_tensor_op() -> None:
                canvas.draw(draw)
                _tensor_reduce(canvas.capture())

            stages["full_with_tensor_op"] = _time_stage(full_with_tensor_op, warmup, frames)

        _release_scene(program, vao, vbo)

    return stages


def bench_egl_composite(
    width    : int,
    height   : int,
    warmup   : int,
    frames   : int,
    tensor_op: bool,
) -> StageTimings:
    stages: StageTimings = {}
    source = CUDABuffer(shape=(height, width, 4), typestr="|u1")

    try:
        with EGLCanvas(width, height) as canvas:
            def write_only() -> None:
                canvas.write(source)

            def draw_only() -> None:
                canvas.draw()
                GL.glFinish()

            def full() -> None:
                canvas.write(source)
                canvas.draw()
                canvas.capture()

            # write_only: CUDA device-to-device copy into the source texture
            stages["write_only"] = _time_stage(write_only, warmup, frames)

            # draw_only: composite blit (V-flip shader)
            canvas.write(source)
            stages["draw_only"] = _time_stage(draw_only, warmup, frames)

            canvas.draw()
            stages["readback_only"] = _time_stage(canvas.capture, warmup, frames)
            stages["full"]          = _time_stage(full, warmup, frames)

            if tensor_op and _TORCH_OK:
                def full_with_tensor_op() -> None:
                    canvas.write(source)
                    canvas.draw()
                    _tensor_reduce(canvas.capture())

                stages["full_with_tensor_op"] = _time_stage(full_with_tensor_op, warmup, frames)

    finally:
        source.close()

    return stages


def bench_egl_record(
    width   : int,
    height  : int,
    warmup  : int,
    frames  : int,
    tmp_path: Path,
) -> StageTimings:
    stages: StageTimings = {}

    with EGLCanvas(width, height) as canvas:
        program, vao, vbo = _build_scene(canvas.ctx)

        def draw(ctx: mgl.Context, _fbo: mgl.Framebuffer) -> None:
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(mgl.TRIANGLES)

        # encode_only: feed a pre-captured buffer to the writer, so only the encode leg is timed
        canvas.draw(draw)
        pre_buffer = canvas.capture()
        with VideoWriter(str(tmp_path), width=width, height=height, fps=60) as recorder:
            stages["encode_only"] = _time_stage(lambda: recorder.write(pre_buffer), warmup, frames)

        with VideoWriter(str(tmp_path), width=width, height=height, fps=60) as recorder:
            def full() -> None:
                canvas.draw(draw)
                recorder.write(canvas.capture())

            stages["full"] = _time_stage(full, warmup, frames)

        _release_scene(program, vao, vbo)

    tmp_path.unlink(missing_ok=True)

    return stages


def bench_x11(
    xid      : int,
    warmup   : int,
    frames   : int,
    tensor_op: bool,
) -> RouteResult:
    """Time window capture, recording the source's size alongside it.

    Returns a whole `RouteResult` rather than bare timings.
    """
    from frame2tensor.capture import X11Window

    stages: StageTimings = {}

    with X11Window(xid) as win:
        stages["capture"] = _time_stage(win.capture, warmup, frames)

        if tensor_op and _TORCH_OK:
            stages["capture_with_tensor_op"] = _time_stage(
                lambda: _tensor_reduce(win.capture()), warmup, frames,
            )

        return RouteResult(route="x11_capture", resolution=[win.width, win.height], xid=xid, stages=stages)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

class _ColumnWidths(NamedTuple):
    """Print-column widths for the summary table, one field per header label."""
    route     : int
    resolution: int
    stage     : int
    mean_ms   : int
    p50_ms    : int
    p95_ms    : int
    p99_ms    : int
    fps       : int


_COL   = _ColumnWidths(route=20, resolution=14, stage=24, mean_ms=9, p50_ms=9, p95_ms=9, p99_ms=9, fps=8)
_WIDTH = sum(_COL) + 4 * 2  # the four two-space gaps between the timing columns


def _row(route: str, resolution: str, stage: str, timing: Timing) -> None:
    print(
        f"  {route:<{_COL.route}}{resolution:<{_COL.resolution}}{stage:<{_COL.stage}}"
        f"{timing.mean_ms:>{_COL.mean_ms-2}.2f}ms  "
        f"{timing.p50_ms:>{_COL.p50_ms-2}.2f}ms  "
        f"{timing.p95_ms:>{_COL.p95_ms-2}.2f}ms  "
        f"{timing.p99_ms:>{_COL.p99_ms-2}.2f}ms  "
        f"{timing.fps:>{_COL.fps}.0f}",
    )


def _header() -> None:
    print(
        f"\n  {'route':<{_COL.route}}{'resolution':<{_COL.resolution}}{'stage':<{_COL.stage}}"
        f"{'mean':>{_COL.mean_ms}}  {'p50':>{_COL.p50_ms}}  {'p95':>{_COL.p95_ms}}  {'p99':>{_COL.p99_ms}}  "
        f"{'fps':>{_COL.fps}}",
    )
    print("  " + "-" * _WIDTH)


def _active_window() -> int | None:
    try:
        return get_active_window()
    except WindowNotFoundError:
        return None


def _gpu_name() -> str:
    err, props = cudart.cudaGetDeviceProperties(0)
    if err != 0:
        return "unknown"

    return props.name.decode() if isinstance(props.name, (bytes, bytearray)) else str(props.name)


def _payload(timestamp: str, args: argparse.Namespace, gpu: str, results: list[RouteResult]) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "system"   : {"gpu": gpu, "platform": platform.platform(), "python": sys.version},
        "config"   : {"warmup": args.warmup, "frames": args.frames, "tensor_op": args.tensor_op},
        "results"  : results,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resolutions", default="1280x720,1920x1080,2560x1440,3840x2160",
        help="Comma-separated WxH resolutions for EGL routes (default: 720p, 1080p, 1440p, 4K)",
    )
    parser.add_argument("--warmup", type=int, default=100, help="Warmup frames to discard (default: 100)")
    parser.add_argument("--frames", type=int, default=500, help="Measurement frames (default: 500)")
    parser.add_argument(
        "--x11", nargs="?", const="auto", default="auto", metavar="XID",
        help="X11 window XID (decimal or 0x hex); defaults to the active window",
    )
    parser.add_argument(
        "--skip", default="", metavar="ROUTE[,ROUTE...]",
        help=f"Routes to leave out; every route runs otherwise. Choices: {', '.join(ALL_ROUTES)}",
    )
    parser.add_argument("--tensor-op", action="store_true", help="Add torch.as_tensor + .float().mean() stage")
    parser.add_argument("--output", help="Output JSON path (default: data/benchmark_<timestamp>.json)")
    parser.add_argument("--quiet", action="store_true", help="Write the JSON without printing the summary table")

    return parser.parse_args()


def _parse_resolutions(spec: str) -> list[tuple[int, int]]:
    resolutions: list[tuple[int, int]] = []
    for part in spec.split(","):
        width, height = part.strip().split("x")
        resolutions.append((int(width), int(height)))

    return resolutions


def _parse_skip(spec: str) -> set[str]:
    skipped = {name.strip() for name in spec.split(",") if name.strip()}
    unknown = skipped - set(ALL_ROUTES)
    if unknown:
        raise SystemExit(f"unknown route(s) to skip: {', '.join(sorted(unknown))}; choose from {', '.join(ALL_ROUTES)}")

    return skipped


# A route's benchmark, already bound to everything but its resolution.
ResolutionBench = Callable[[int, int], StageTimings]


def _run_egl_route(
    route      : str,
    bench      : ResolutionBench,
    resolutions: list[tuple[int, int]],
    quiet      : bool,
) -> list[RouteResult]:
    """Run one EGL route at every resolution, printing each stage as it lands."""
    results: list[RouteResult] = []
    for width, height in resolutions:
        stages = bench(width, height)
        if not quiet:
            for stage, timing in stages.items():
                _row(route, f"{width}x{height}", stage, timing)
        results.append(RouteResult(route=route, resolution=[width, height], stages=stages))
    if not quiet:
        print(" " + " ." * (_WIDTH // 2))

    return results


def _run_x11_subprocess(xid: int, args: argparse.Namespace, data_dir: Path) -> list[RouteResult]:
    """Re-run this script with only the x11 route, and read its timings back.

    X11 capture (GLX) and the EGL routes each bind CUDA to their own GL context,
    and CUDA permits one GL context association per device, so the two cannot share a process.
    The child is given the complement of this route as its `--skip` list, which leaves it nothing to do but x11 capture,
    and `--quiet` so only this process prints the table.
    """
    tmp = data_dir / "_bench_x11_tmp.json"
    cmd = [
        sys.executable, __file__,
        f"--x11={xid}", f"--skip={','.join(EGL_ROUTES)}", "--quiet",
        f"--warmup={args.warmup}", f"--frames={args.frames}",
        f"--output={tmp}",
    ]
    if args.tensor_op:
        cmd.append("--tensor-op")

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
        payload  = orjson.loads(tmp.read_bytes())
        results  = [RouteResult.from_json(entry) for entry in payload["results"]]
        _print_x11(results)
        return results
    except Exception as exc:
        print(f"  x11_capture  failed: {exc}")
        return []
    finally:
        tmp.unlink(missing_ok=True)


def _print_x11(results: list[RouteResult]) -> None:
    for result in results:
        width, height = result.resolution
        for stage, timing in result.stages.items():
            _row(result.route, f"{width}x{height}", stage, timing)
    print(" " + " ." * (_WIDTH // 2))


def _run_x11(args: argparse.Namespace, skipped: set[str], data_dir: Path, quiet: bool) -> list[RouteResult]:
    """Benchmark the x11 route, in-process when no EGL route needs the device's GL association."""
    if not _X11_OK:
        if not quiet:
            print("\n  x11_capture  skipped (X11 capture unavailable)")
        return []

    xid = _active_window() if args.x11 == "auto" else int(args.x11, 0)
    if xid is None:
        if not quiet:
            print("\n  x11_capture  skipped (no active window)")
        return []

    if not set(EGL_ROUTES) <= skipped:
        return _run_x11_subprocess(xid, args, data_dir)

    results = [bench_x11(xid, args.warmup, args.frames, args.tensor_op)]
    if not quiet:
        _print_x11(results)

    return results


def main() -> None:
    args      = parse_args()
    skipped   = _parse_skip(args.skip)
    data_dir  = Path("data")
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat()
    out_path  = Path(args.output) if args.output else data_dir / f"benchmark_{timestamp[:19].replace(':', '-')}.json"
    gpu       = _gpu_name()

    resolutions = _parse_resolutions(args.resolutions)

    if args.tensor_op and not _TORCH_OK:
        print("warning: --tensor-op requested but torch is not available; skipping tensor stages", file=sys.stderr)

    if not args.quiet:
        print("\nframe2tensor pipeline benchmark")
        print(f"  gpu:    {gpu}")
        print(f"  warmup: {args.warmup}  frames: {args.frames}")
        if skipped:
            print(f"  skipped: {', '.join(sorted(skipped))}")
        _header()

    results: list[RouteResult] = []

    if "x11_capture" not in skipped:
        results += _run_x11(args, skipped, data_dir, args.quiet)

    record_path = data_dir / "_bench_record_tmp.mp4"
    egl_benches: dict[str, ResolutionBench] = {
        "egl_render"   : lambda w, h: bench_egl_render(w, h, args.warmup, args.frames, args.tensor_op),
        "egl_composite": lambda w, h: bench_egl_composite(w, h, args.warmup, args.frames, args.tensor_op),
        "egl_record"   : lambda w, h: bench_egl_record(w, h, args.warmup, args.frames, record_path),
    }
    for route, bench in egl_benches.items():
        if route not in skipped:
            results += _run_egl_route(route, bench, resolutions, args.quiet)

    out_path.write_bytes(orjson.dumps(_payload(timestamp, args, gpu, results), option=orjson.OPT_INDENT_2))
    if not args.quiet:
        print()
        print(" " + "-" * _WIDTH)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
