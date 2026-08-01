# frame2tensor

A simple Python toolkit for real-time Computer Vision pipelines on Linux systems.
Using the CUDA runtime API, it registers GL textures with CUDA: enabling on-device interoperation between CUDA, GL, and ML frameworks.

Designed for: game playing agents/bots, neural visualization, synthetic data generation, and anything else where frames and tensors must remain on the GPU.

>Requires: X11 & a CUDA-capable GPU; Python 3.11+

## Features

- Capture a window or a custom draw function into a tensor.
- Render tensors and GL draw functions to a window or disk.
- Write tensors into a GL texture, e.g. for visualization.

## Dependencies

`frame2tensor` depends on `cuda-python`, which must match your installed NVIDIA driver. If you see `cudaErrorInsufficientDriver`, your driver is too old for the installed `cuda-python` version.
Check with `nvidia-smi`, then consult the [CUDA toolkit compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) to find the compatible `cuda-python` range and pin it in your dependencies.

>NOTE: this project was developed on a 3060 with a 525–550 series driver. I have no idea how to support other hardware at the moment. Keep this in mind if you have any issues.

`xwininfo` is used by `window_finder.pick_window`.

`ffmpeg` is used by `VideoWriter.write`.

## Install

Install from GitHub:

```
uv add git+https://github.com/ShepL137/frame2tensor.git
```
or
```
pip install git+https://github.com/ShepL137/frame2tensor.git
```

## Quick Start

### Window capture

Capture a live window and feed it to a PyTorch model.

```python
import torch
from frame2tensor.capture import X11Window, pick_window

with X11Window(pick_window()) as win:
    buf    = win.capture()                          # CUDABuffer, 4 channel uint8
    tensor = torch.as_tensor(buf, device="cuda")    # zero-copy wrap
    tensor = tensor.float() / 255.0                 # cast + copy to float (we're working to eliminate this)
    model(tensor)                                   # feed it your CV pipeline
```

`CUDABuffer` exposes `__cuda_array_interface__`, so any compatible framework can wrap it zero-copy.

`capture()` returns a borrowed reference, the same object on every call, mutated in place; `tensor` above is a zero-copy wrap of it, so clone it before exiting the context manager if you still need the old frame.
>WARNING: on a source-window resize, `capture()` frees the old buffer and allocates a new one internally, so a zero-copy wrap that _wasn't_ cloned is now pointing to memory that's been freed.

`X11Window` takes an X11 XID (decimal or `0x`-prefixed hex):
- `pick_window()` selects one interactively by click (requires `xwininfo`)
- `get_active_window()` returns the currently focused window
- or pass an XID directly.

See [docs/quickstart.md](docs/quickstart.md) for more.

## Examples

See [`examples/`](examples/) for complete, self-contained demos:

- [`motion_detection`](examples/motion_detection/): live frame-differencing motion filters over a captured window (including windowed and click-through AR overlay versions).
- [`retinal_opponency`](examples/retinal_opponency/): a center-surround color-opponency filter over a captured window, with four views switchable at runtime.
- [`rnn_activity`](examples/rnn_activity/): an RNN's per-layer activations rendered live as heatmap tiles.

## Motivation

I wanted to make a bot with biologically motivated real-time vision to play games, exploring such things as world models; for, games and other applications supply a vast range of complex dynamics and challenges.

I wasn't sure of anything besides mss for the question of frame capture. Perhaps similar projects as this exist, but I didn't manage to discover them. Regardless, I should hope that this one might help to boost the signal for such solutions.

I only wanted to get frames as tensors without the CPU. I do hope that this might spare others the trouble; that others may, as I had wished to, care not for the tooling, but for the model architecture instead.

And, I shall endeavor to prove this project by the others I have planned.

## Future

The roadmap gist:

- better type support (rendering & direct f32 capture)
- better API
- JAX support (via `dlpack`)
- desktop capture
- performance optimizations (especially for recording)
- various fixes
- more examples
- better docs
