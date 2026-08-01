# Quick Start

## Window capture

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

## Headless capture (GL → tensor)
Render synthetic content off-screen and pull the frame back as a tensor.

```python
import torch
from frame2tensor.render import EGLCanvas

with EGLCanvas(width, height) as canvas:
    while rendering:
        canvas.draw(my_draw)
        tensor = torch.as_tensor(canvas.capture(), device="cuda")
```

The FBO `my_draw` renders into (and what `capture()` reads back) is always RGBA uint8; other formats aren't configurable yet.

## Windowed rendering (GL → tensor)
Same as headless capture, but rendered to a visible window instead of off-screen.

```python
import torch
from frame2tensor.render import WindowedRenderer

with WindowedRenderer(width, height) as win:
    while not win.is_closing:
        win.draw(my_draw)
        tensor = torch.as_tensor(win.capture(), device="cuda")
        win.swap()
```

Call `capture()` after `draw()` and before `swap()`.

## Compositing a CUDA buffer into a window
Blend a captured window with your own overlay content drawn on top. See [`motion_detection`](examples/motion_detection/).

```python
from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.render import WindowedRenderer

def my_overlay(ctx, fbo):
    ...  # your own ModernGL drawing code, e.g. a translucent highlight

with X11Window(get_active_window()) as src, WindowedRenderer(src.width, src.height) as win:
    while not win.is_closing:
        buf = src.capture()
        win.write(buf)          # copy into source texture
        win.draw(my_overlay)    # blend source + overlay, blit to screen
        win.swap()
```

`write()` accepts any `__cuda_array_interface__` object.
`draw()` composites it into the FBO and calls `on_draw(ctx, fbo)` for any overlay layer before the blit.

## Writing tensors to GL (CUDA → texture)
E.g., neural visualizations. See [`rnn_activity`](examples/rnn_activity/)

```python
import moderngl as mgl
import torch

from frame2tensor import CUDAWritableTexture

ctx      = mgl.create_context(standalone=True)                                              # or reuse an existing ctx
gl_tex   = ctx.texture(size=(WIDTH, HEIGHT), components=1, dtype="f4")                      # define a float32 texture
cuda_tex = CUDAWritableTexture(texture=gl_tex, width=WIDTH, height=HEIGHT, components=1)    # register gl_tex for interop

while rendering:
    activation = torch.rand(WIDTH, HEIGHT, device="cuda")
    cuda_tex.write(activation)
```

Inputs to `write()` must be contiguous.

## Recording to video
Write captured frames to disk as a video file via an `ffmpeg` subprocess. Requires `ffmpeg` on `PATH`.

```python
from frame2tensor.capture import X11Window, get_active_window
from frame2tensor.output import VideoWriter

with X11Window(get_active_window()) as win, VideoWriter("out.mp4", win.width, win.height, fps=60) as rec:
    for _frame in range(300):    # 5 seconds @ 60 fps
        rec.write(win.capture())
```

All frame sources (`EGLCanvas`, `WindowedRenderer`, `X11Window`) return top-down buffers (row zero is top of screen) via `.capture()`.

>NOTE: Recording is currently a major bottleneck. `VideoWriter` remains very limited.
