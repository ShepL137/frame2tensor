"""Live per-layer RNN activation heatmaps.

Each hidden layer of a small PyTorch RNN is rendered as a tile in a heatmap grid.

Runs until the window is closed.
"""
import argparse
import math
import struct
import time
from pathlib import Path

import moderngl as mgl
import torch
from torch import nn

from frame2tensor import CUDAWritableTexture
from frame2tensor.render import WindowedRenderer

LAYER_SIZE  = 32    # side of each layer's square tile, so 32x32 = 1024 units per layer
TILE_GRID   = 2     # tiles per side; the model gets one layer per tile
TILE_MARGIN = 0.05  # gap between tiles, in NDC (normalized device coordinate) units
INPUT_SIZE  = 64
WINDOW_SIZE = 720

# The input units sit on a circle, and a bump of activity sweeps around it.
RING_WIDTH = 0.4    # angular half-width of the bump, in radians
RING_SPEED = 2.0    # revolutions per second

SHADER_DIR = Path(__file__).parent / "shaders"


# -----------------------------------------------------------------------------
# Model

class StackedRNN(nn.Module):
    """Stack of RNN cells driven by a bump of input sweeping around a ring."""

    def __init__(self, layer_size, num_layers, input_size=INPUT_SIZE):
        super().__init__()
        self.layer_size = layer_size
        self.input_size = input_size

        # Each layer is displayed as a square tile of units.
        self.hidden_size = layer_size**2

        self.layers = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else self.hidden_size, self.hidden_size)
            for i in range(num_layers)
        ])
        self.hidden = [
            torch.zeros(1, self.hidden_size, device="cuda")
            for _ in range(num_layers)
        ]

    def forward(self, x):
        activations = []
        signal      = x

        for i, cell in enumerate(self.layers):
            self.hidden[i] = cell(signal, self.hidden[i])
            signal         = torch.relu(self.hidden[i])
            activations.append(signal.view(self.layer_size, self.layer_size))

        return activations


def ring_positions(count):
    """Angular position of each input unit, spread evenly around one full turn."""
    return torch.linspace(0.0, math.tau, count + 1, device="cuda")[:-1]


def ring_input(positions, phase, width):
    """A Gaussian bump of activity on the ring, centered at `phase`."""
    # Wrap the angular distance into [-pi, pi) so the bump stays continuous across the seam at 0.
    offset = torch.remainder(positions - phase + math.pi, math.tau) - math.pi

    return torch.exp(-(offset**2) / (2.0 * width**2)).unsqueeze(0)


def normalize(activations):
    """Rescale one layer against its own range, into the 0..1 the heatmap gradient expects.

    Zero-centered weights + ReLU cuts activation sums roughly in half per layer.
    Since their magnitudes differ by depth, each layer is scaled separately.
    Tiles therefore show relative structure within a layer and not absolute magnitudes.
    """
    low, high = activations.aminmax()

    return (activations - low) / (high - low).clamp(min=1e-8)


# -----------------------------------------------------------------------------
# GL resources

def build_textures(ctx, count, layer_size):
    """Create a single-channel float texture per layer, each with a CUDA handle to write through.

    Returns the GL textures (used for rendering) and the CUDA handles (used for writing), in matching order.
    """
    textures      = []
    cuda_textures = []

    for _ in range(count):
        texture = ctx.texture(size=(layer_size, layer_size), components=1, dtype="f4")
        # Linear filtering smooths the small grid of units across a much larger on-screen tile.
        texture.filter = (mgl.LINEAR, mgl.LINEAR)

        textures.append(texture)
        cuda_textures.append(CUDAWritableTexture(texture=texture, width=layer_size, height=layer_size))

    return textures, cuda_textures


def build_program(ctx):
    """Compile the heatmap shaders and point their sampler at texture slot 0."""
    program = ctx.program(
        vertex_shader   = (SHADER_DIR / "heatmap.vert").read_text(),
        fragment_shader = (SHADER_DIR / "heatmap.frag").read_text(),
    )
    program["u_texture"] = 0

    return program


def make_quad(x0, y0, x1, y1):
    """Two triangles covering one rectangle; each vertex carries a position then a texture coordinate."""
    return [
        x0, y0,  0.0, 0.0,
        x1, y0,  1.0, 0.0,
        x1, y1,  1.0, 1.0,
        x0, y0,  0.0, 0.0,
        x1, y1,  1.0, 1.0,
        x0, y1,  0.0, 1.0,
    ]


def build_tiles(ctx, program, grid, margin):
    """Lay out a grid of quads, ordered left to right and top to bottom to match the layer order.

    Normalized device coordinates span -1..1 on both axes,
    so one cell is 2/grid across before the margin is taken out of each side.
    Returns the vertex arrays to render and the buffers behind them, which are released together.
    """
    vaos = []
    vbos = []
    cell = 2.0 / grid

    for row in range(grid):
        for column in range(grid):
            x0 = -1.0 + column * cell + margin
            x1 = -1.0 + (column + 1) * cell - margin
            # NDC y (normalized device coordinates) grows upward while rows count downward,
            # so row 0 starts at the top edge.
            y1 =  1.0 - row * cell - margin
            y0 =  1.0 - (row + 1) * cell + margin

            vertices = make_quad(x0, y0, x1, y1)
            vbo      = ctx.buffer(data=struct.pack(f"{len(vertices)}f", *vertices))

            vbos.append(vbo)
            vaos.append(ctx.vertex_array(program, [(vbo, "2f 2f", "in_position", "in_texcoord")]))

    return vaos, vbos


def cleanup(cuda_textures, gl_resources):
    """Release the interop handles, then the GL objects behind them.

    Each CUDAWritableTexture holds a CUDA registration on its texture, so the registration has to go first.
    """
    for cuda_texture in cuda_textures:
        cuda_texture.close()

    for resource in gl_resources:
        resource.release()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(
        description     = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Frame limiter; 0 for vsync, negative to disable (default: 60)",
    )
    parser.add_argument(
        "--layer-size", type=int, default=LAYER_SIZE,
        help=f"Side of each layer's square tile, so N*N units per layer (default: {LAYER_SIZE})",
    )
    parser.add_argument(
        "--ring-speed", type=float, default=RING_SPEED,
        help=f"Revolutions per second of the input bump; 0 holds it still (default: {RING_SPEED})",
    )

    return parser.parse_args()


def main():
    args       = parse_args()
    vsync      = args.fps == 0
    num_layers = TILE_GRID**2

    with WindowedRenderer(
        width=WINDOW_SIZE, height=WINDOW_SIZE, title="Neural Activity - frame2tensor", vsync=vsync,
    ) as window:
        model = StackedRNN(layer_size=args.layer_size, num_layers=num_layers).cuda()
        model.eval()

        program                 = build_program(window.ctx)
        textures, cuda_textures = build_textures(window.ctx, num_layers, args.layer_size)
        vaos, vbos              = build_tiles(window.ctx, program, TILE_GRID, TILE_MARGIN)

        def draw_tiles(ctx, _fbo):
            ctx.clear(0.05, 0.05, 0.1, 1.0)
            for texture, vao in zip(textures, vaos, strict=True):
                texture.use(location=0)
                vao.render(mgl.TRIANGLES)

        print(f"Visualizing {num_layers} layers of {args.layer_size}x{args.layer_size} units")

        positions  = ring_positions(model.input_size)
        fps_report = FpsReporter()
        start      = time.perf_counter()

        try:
            while not window.is_closing:
                t0 = time.perf_counter()

                # Phase comes from elapsed time, so the bump sweeps at the same rate whatever the frame rate.
                phase = math.tau * args.ring_speed * (t0 - start)

                with torch.no_grad():
                    activations = model(ring_input(positions, phase, RING_WIDTH))

                tiles = [normalize(activation) for activation in activations]

                for cuda_texture, tile in zip(cuda_textures, tiles, strict=True):
                    cuda_texture.write(tile)

                window.draw(draw_tiles)
                window.swap()

                limit_framerate(t0, args.fps)
                fps_report.tick()
            print()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cleanup(cuda_textures, (*textures, *vaos, *vbos, program))


if __name__ == "__main__":
    main()
