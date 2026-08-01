"""Tests for CUDABuffer."""

from typing import Any

import pytest
import torch
from torch._tensor import Tensor

from frame2tensor.types import CUDABuffer


class TestCUDABufferAllocation:
    """Memory allocation and lifecycle."""

    def test_allocates_device_memory(self) -> None:
        """Buffer holds a nonzero device pointer after construction."""
        with CUDABuffer(shape=(32, 32, 4), typestr="|u1") as buf:
            assert buf.ptr != 0

    def test_properties(self) -> None:
        """Shape, typestr, and itemsize reflect constructor args."""
        with CUDABuffer(shape=(64, 64, 4), typestr="|u1") as buf:
            assert buf.shape    == (64, 64, 4)
            assert buf.typestr  == "|u1"
            assert buf.itemsize == 1

    def test_float32_itemsize(self) -> None:
        with CUDABuffer(shape=(32, 32), typestr="<f4") as buf:
            assert buf.itemsize == 4

    def test_unsupported_typestr(self) -> None:
        with pytest.raises(ValueError, match="Unsupported typestr"):
            CUDABuffer(shape=(32, 32), typestr="<f8")

    def test_close_idempotent(self) -> None:
        buf: CUDABuffer = CUDABuffer(shape=(32, 32), typestr="|u1")
        buf.close()
        buf.close()

    def test_context_manager(self) -> None:
        with CUDABuffer(shape=(32, 32), typestr="|u1") as buf:
            assert buf.ptr != 0
        assert buf._closed


class TestCUDABufferInterface:
    """__cuda_array_interface__ exposes correct metadata."""

    def test_interface_fields(self) -> None:
        with CUDABuffer(shape=(64, 32, 4), typestr="|u1") as buf:
            interface = buf.__cuda_array_interface__
            assert interface["shape"]   == (64, 32, 4)
            assert interface["typestr"] == "|u1"
            assert interface["version"] == 3
            pointer, read_only = interface["data"]
            assert pointer   == buf.ptr
            assert read_only is False


class TestCUDABufferTorchInterop:
    """PyTorch consumes CUDABuffer zero-copy."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_as_tensor_shape_and_dtype(self) -> None:
        with CUDABuffer(shape=(32, 64, 4), typestr="|u1") as buf:
            t: Tensor = torch.as_tensor(data=buf, device="cuda")
            assert t.shape == (32, 64, 4)
            assert t.dtype == torch.uint8

    def test_as_tensor_float32(self) -> None:
        with CUDABuffer(shape=(32, 32), typestr="<f4") as buf:
            t: Tensor = torch.as_tensor(data=buf, device="cuda")
            assert t.shape == (32, 32)
            assert t.dtype == torch.float32

    def test_zero_copy(self) -> None:
        """torch.as_tensor wraps the same device memory, no copy."""
        with CUDABuffer(shape=(32, 32, 4), typestr="|u1") as buf:
            t: Tensor = torch.as_tensor(data=buf, device="cuda")
            assert t.data_ptr() == buf.ptr


class TestCUDABufferCuPyInterop:
    """CuPy consumes CUDABuffer zero-copy."""

    @pytest.fixture
    def cupy(self) -> Any:
        return pytest.importorskip(modname="cupy")

    def test_asarray_shape_and_dtype(self, cupy: Any) -> None:
        with CUDABuffer(shape=(32, 64, 4), typestr="|u1") as buf:
            arr: Any = cupy.asarray(buf)
            assert arr.shape == (32, 64, 4)
            assert arr.dtype == cupy.uint8

    def test_asarray_float32(self, cupy: Any) -> None:
        with CUDABuffer(shape=(32, 32), typestr="<f4") as buf:
            arr: Any = cupy.asarray(buf)
            assert arr.shape == (32, 32)
            assert arr.dtype == cupy.float32

    def test_zero_copy(self, cupy: Any) -> None:
        """cupy.asarray wraps the same device memory, no copy."""
        with CUDABuffer(shape=(32, 32, 4), typestr="|u1") as buf:
            arr: Any = cupy.asarray(buf)
            assert arr.data.ptr == buf.ptr
