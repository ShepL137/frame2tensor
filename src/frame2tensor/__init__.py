"""Low-latency GL-CUDA tensor interop and screen capture for real-time vision pipelines.

The package root holds the core interop layer: buffer and texture types plus the exception hierarchy.
The toolkit lives in subpackages: :mod:`~frame2tensor.capture` (X11 window capture),
:mod:`~frame2tensor.render` (headless and windowed render targets),
and :mod:`~frame2tensor.output` (frame sinks).
"""

from importlib.metadata import PackageNotFoundError, version

from .exceptions import (
    CaptureSourceLostError,
    CUDAError,
    Frame2TensorError,
    GLContextError,
    InvalidTensorError,
    OutputError,
    WindowNotFoundError,
)
from .interop import CUDAWritableTexture, GLCUDATexture
from .types import CUDABuffer, SupportsCUDAArray

__all__ = [
    "CUDABuffer",
    "CUDAError",
    "CUDAWritableTexture",
    "CaptureSourceLostError",
    "Frame2TensorError",
    "GLCUDATexture",
    "GLContextError",
    "InvalidTensorError",
    "OutputError",
    "SupportsCUDAArray",
    "WindowNotFoundError",
]

try:
    __version__ = version(distribution_name="frame2tensor")
except PackageNotFoundError:
    __version__ = "unknown"
