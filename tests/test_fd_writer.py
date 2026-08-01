"""Tests for frame2tensor.output.fd_writer.FileDescriptorWriter.

Validation and lifecycle tests run without a GPU (the source is rejected before any copy).
The round-trip and broken-pipe tests seed real device memory.
"""

import os

import pytest
import torch

from frame2tensor.exceptions import InvalidTensorError, OutputError
from frame2tensor.output import FileDescriptorWriter
from frame2tensor.types import CUDAArrayInterface


class _FakeArray:
    """Minimal object exposing a chosen `__cuda_array_interface__`."""

    def __init__(self, interface: CUDAArrayInterface) -> None:
        self._interface: CUDAArrayInterface = interface

    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterface:
        return self._interface


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_unsupported_typestr_raises(self):
        with pytest.raises(ValueError, match="Unsupported typestr"):
            FileDescriptorWriter(fd=1, width=4, height=4, typestr="<i8")


# -----------------------------------------------------------------------------
# Validation (no GPU; rejection precedes any copy)
# -----------------------------------------------------------------------------


class TestValidation:
    def test_missing_interface_raises(self):
        writer = FileDescriptorWriter(fd=1, width=4, height=4)
        with pytest.raises(InvalidTensorError, match="__cuda_array_interface__"):
            writer.write(object())  # pyright: ignore[reportArgumentType]

    def test_wrong_shape_raises(self):
        writer = FileDescriptorWriter(fd=1, width=4, height=4)
        source = _FakeArray({"shape": (8, 8, 4), "typestr": "|u1", "data": (0xDEAD, False), "version": 3})
        with pytest.raises(InvalidTensorError, match="expected shape"):
            writer.write(source)

    def test_wrong_dtype_raises(self):
        writer = FileDescriptorWriter(fd=1, width=4, height=4)
        source = _FakeArray({"shape": (4, 4, 4), "typestr": "<f4", "data": (0xDEAD, False), "version": 3})
        with pytest.raises(InvalidTensorError, match="expected dtype"):
            writer.write(source)

    def test_non_contiguous_raises(self):
        writer = FileDescriptorWriter(fd=1, width=4, height=4)
        # Strides that do not match a C-contiguous (4, 4, 4) uint8 layout.
        source = _FakeArray(
            {"shape": (4, 4, 4), "typestr": "|u1", "data": (0xDEAD, False), "strides": (4, 16, 1), "version": 3},
        )
        with pytest.raises(InvalidTensorError, match="contiguous"):
            writer.write(source)


# -----------------------------------------------------------------------------
# Round-trip through a real pipe (needs CUDA)
# -----------------------------------------------------------------------------


class TestRoundTrip:
    def test_frame_bytes_match_through_pipe(self):
        height, width, components = 4, 4, 4
        count = height * width * components
        data  = torch.arange(count, dtype=torch.uint8, device="cuda").reshape(height, width, components)

        read_fd, write_fd = os.pipe()
        try:
            FileDescriptorWriter(write_fd, width=width, height=height).write(data)
            out = os.read(read_fd, count)
        finally:
            os.close(read_fd)
            os.close(write_fd)

        assert out == data.cpu().numpy().tobytes()

    def test_broken_pipe_raises_output_error(self):
        data = torch.zeros((4, 4, 4), dtype=torch.uint8, device="cuda")

        read_fd, write_fd = os.pipe()
        os.close(read_fd)  # reader gone: the write should fail with a broken pipe
        try:
            with pytest.raises(OutputError):
                FileDescriptorWriter(write_fd, width=4, height=4).write(data)
        finally:
            os.close(write_fd)
