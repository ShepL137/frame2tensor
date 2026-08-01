"""Output path: write raw frames to a file descriptor.

``FileDescriptorWriter`` copies a device buffer to host memory and writes its raw bytes to a file descriptor.
The descriptor may be anything: an ffmpeg subprocess's stdin, a named pipe, a plain file, or a socket.
The writer does not interpret the bytes or own the descriptor.
"""

import ctypes
import os
from typing import Any

from frame2tensor._cuda import memcpy_device_to_host
from frame2tensor.exceptions import InvalidTensorError, OutputError
from frame2tensor.types import _ITEMSIZES, SupportsCUDAArray  # pyright: ignore[reportPrivateUsage]


class FileDescriptorWriter:
    """Write raw frames from device memory to a file descriptor.

    Each ``write()`` copies a device buffer (anything exposing ``__cuda_array_interface__``)
    into a host staging buffer and writes the raw bytes to the descriptor.

    Geometry is fixed at construction so a source whose shape drifts, such as a resized capture buffer,
    is rejected before it desyncs the consumer's stream.

    All standard frame sources (
    :class:`~frame2tensor.capture.X11Window`,
    :class:`~frame2tensor.render.WindowedRenderer`,
    :class:`~frame2tensor.render.EGLCanvas`
    ) return top-down buffers (row 0 is the visual top).
    ``FileDescriptorWriter`` does not reorder rows.
    """

    def __init__(
        self,
        fd        : int,
        width     : int,
        height    : int,
        components: int = 4,
        typestr   : str = "|u1",
    ) -> None:
        """Prepare a writer for fixed-geometry frames to a descriptor.

        Args:
            fd        : File descriptor to write to. Owned by the caller; not closed.
            width     : Frame width in pixels.
            height    : Frame height in pixels.
            components: Number of color components per pixel.
            typestr   : Numpy-style type string.

        Raises:
            ValueError: If typestr is not supported.
        """
        if typestr not in _ITEMSIZES:
            raise ValueError(f"Unsupported typestr: {typestr!r}")

        self._fd         : int  = fd
        self.width       : int  = width
        self.height      : int  = height
        self.components  : int  = components
        self.typestr     : str  = typestr
        self._itemsize   : int  = _ITEMSIZES[typestr]
        self._frame_bytes: int  = height * width * components * self._itemsize

        self._staging    : ctypes.Array[ctypes.c_char] = (ctypes.c_char * self._frame_bytes)()
        self._staging_ptr: int                          = ctypes.addressof(self._staging)

    # -----------------------------------------------------------------------------

    def write(self, source: SupportsCUDAArray) -> None:
        """Copy a device buffer to host and write its raw bytes to the descriptor.

        Re-validates the source's shape, dtype, and contiguity on every call, copies device-to-host,
        then writes the full frame to the descriptor, looping over partial writes.

        Args:
            source: An object exposing ``__cuda_array_interface__``.

        Raises:
            InvalidTensorError: If the source does not match the fixed geometry.
            OutputError       : If the descriptor write fails.
        """
        src_ptr: int = self._validate(source)
        memcpy_device_to_host(dst=self._staging_ptr, src=src_ptr, count=self._frame_bytes)

        view : memoryview = memoryview(self._staging)
        total: int        = 0
        while total < self._frame_bytes:
            try:
                written: int = os.write(self._fd, view[total:])
            except OSError as err:
                raise OutputError(f"write to fd {self._fd} failed: {err}") from err
            if written == 0:
                raise OutputError(f"fd {self._fd} accepted no bytes")
            total += written

    # -----------------------------------------------------------------------------

    def _validate(self, source: SupportsCUDAArray) -> int:
        """Validate a source against the fixed geometry and return its device pointer.

        Args:
            source: Any object exposing ``__cuda_array_interface__``.

        Returns:
            The source's device pointer.

        Raises:
            InvalidTensorError: If the object lacks the interface, has the wrong shape or dtype, or is not C-contiguous.
        """
        interface: Any | None = getattr(source, "__cuda_array_interface__", None)
        if interface is None:
            raise InvalidTensorError("object does not expose __cuda_array_interface__")

        shape = interface["shape"]
        expected: tuple[int, int] | tuple[int, int, int]
        expected = (self.height, self.width) if self.components == 1 else (self.height, self.width, self.components)
        if shape != expected:
            raise InvalidTensorError(f"expected shape {expected}, got {shape}")

        typestr = interface["typestr"]
        if typestr != self.typestr:
            raise InvalidTensorError(f"expected dtype {self.typestr!r}, got {typestr!r}")

        strides = interface.get("strides")
        if strides is not None:
            expected_strides: list[int] = []
            stride          : int       = self._itemsize
            for dim in reversed(shape):
                expected_strides.append(stride)
                stride *= dim
            expected_strides.reverse()
            if tuple(strides) != tuple(expected_strides):
                raise InvalidTensorError("source must be contiguous")

        return int(interface["data"][0])
