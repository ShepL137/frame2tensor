"""Tests for frame2tensor._cuda wrappers.

All cudart calls are mocked. We verify that each wrapper forwards the correct positional arguments
and raises `CUDAError` on non-zero return codes.
"""

from unittest.mock import patch

import cuda.bindings.runtime as cudart
import pytest

from frame2tensor._cuda import (
    _check,
    get_mapped_array,
    map_resources,
    memcpy_2d_from_array,
    memcpy_2d_to_array,
    memcpy_device_to_host,
    register_gl_image,
    unmap_resources,
    unregister_resource,
)
from frame2tensor.exceptions import CUDAError

# Patch target; the cudart module as imported inside _cuda.py
_CUDART = "frame2tensor._cuda.cudart"


# -----------------------------------------------------------------------------
# _check
# -----------------------------------------------------------------------------


class TestCheck:
    def test_success_is_silent(self):
        _check(0, "op")

    def test_nonzero_raises(self):
        with pytest.raises(CUDAError) as exc_info:
            _check(35, "some_operation")
        assert exc_info.value.error_code == 35
        assert exc_info.value.operation == "some_operation"


# -----------------------------------------------------------------------------
# Wrapper success paths
# -----------------------------------------------------------------------------


class TestRegisterGLImage:
    @patch(f"{_CUDART}.cudaGraphicsGLRegisterImage")
    def test_forwards_args_and_returns_resource(self, mock_register):
        sentinel = object()
        mock_register.return_value = (0, sentinel)

        result = register_gl_image(glo=42, target=0x0DE1, flags=1)

        mock_register.assert_called_once_with(42, 0x0DE1, 1)
        assert result is sentinel


class TestUnregisterResource:
    @patch(f"{_CUDART}.cudaGraphicsUnregisterResource")
    def test_forwards_resource(self, mock_unregister):
        sentinel = object()
        mock_unregister.return_value = (0,)

        unregister_resource(sentinel)

        mock_unregister.assert_called_once_with(sentinel)


class TestMapResources:
    @patch(f"{_CUDART}.cudaGraphicsMapResources")
    def test_forwards_args_default_stream(self, mock_map):
        sentinel = object()
        mock_map.return_value = (0,)

        map_resources(sentinel)

        mock_map.assert_called_once_with(1, sentinel, 0)

    @patch(f"{_CUDART}.cudaGraphicsMapResources")
    def test_forwards_explicit_stream(self, mock_map):
        sentinel = object()
        mock_map.return_value = (0,)

        map_resources(sentinel, stream=7)

        mock_map.assert_called_once_with(1, sentinel, 7)


class TestUnmapResources:
    @patch(f"{_CUDART}.cudaGraphicsUnmapResources")
    def test_forwards_args_default_stream(self, mock_unmap):
        sentinel = object()
        mock_unmap.return_value = (0,)

        unmap_resources(sentinel)

        mock_unmap.assert_called_once_with(1, sentinel, 0)


class TestGetMappedArray:
    @patch(f"{_CUDART}.cudaGraphicsSubResourceGetMappedArray")
    def test_forwards_args_and_returns_array(self, mock_get):
        resource = object()
        cuda_array = object()
        mock_get.return_value = (0, cuda_array)

        result = get_mapped_array(resource)

        mock_get.assert_called_once_with(resource, 0, 0)
        assert result is cuda_array

    @patch(f"{_CUDART}.cudaGraphicsSubResourceGetMappedArray")
    def test_custom_array_index_and_mip(self, mock_get):
        resource = object()
        mock_get.return_value = (0, object())

        get_mapped_array(resource, array_index=2, mip_level=3)

        mock_get.assert_called_once_with(resource, 2, 3)


class TestMemcpy2DFromArray:
    @patch(f"{_CUDART}.cudaMemcpy2DFromArray")
    def test_forwards_args(self, mock_copy):
        mock_copy.return_value = (0,)
        src = object()

        memcpy_2d_from_array(
            dst_ptr=0xDEAD,
            dst_pitch=128,
            src_array=src,
            width_bytes=128,
            height=32,
        )

        args = mock_copy.call_args[0]
        assert args[0] == 0xDEAD    # dst
        assert args[1] == 128       # dpitch
        assert args[2] is src       # src
        assert args[3] == 0         # wOffset
        assert args[4] == 0         # hOffset
        assert args[5] == 128       # width
        assert args[6] == 32        # height
        # args[7] is cudaMemcpyDeviceToDevice (existence is sufficient)  # noqa: ERA001


class TestMemcpy2DToArray:
    @patch(f"{_CUDART}.cudaMemcpy2DToArray")
    def test_forwards_args(self, mock_copy):
        mock_copy.return_value = (0,)
        dst = object()

        memcpy_2d_to_array(
            dst_array=dst,
            src_ptr=0xBEEF,
            src_pitch=256,
            width_bytes=256,
            height=64,
        )

        args = mock_copy.call_args[0]
        assert args[0] is dst       # dst
        assert args[1] == 0         # wOffset
        assert args[2] == 0         # hOffset
        assert args[3] == 0xBEEF    # src
        assert args[4] == 256       # spitch
        assert args[5] == 256       # width
        assert args[6] == 64        # height


class TestMemcpyDeviceToHost:
    @patch(f"{_CUDART}.cudaMemcpy")
    def test_forwards_args_with_device_to_host_kind(self, mock_copy):
        mock_copy.return_value = (0,)

        memcpy_device_to_host(dst=0xCAFE, src=0xBEEF, count=4096)

        args = mock_copy.call_args[0]
        assert args[0] == 0xCAFE    # dst (host)
        assert args[1] == 0xBEEF    # src (device)
        assert args[2] == 4096      # count
        assert args[3] == cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


# -----------------------------------------------------------------------------
# Wrapper failure paths
# -----------------------------------------------------------------------------

# Each entry: (wrapper_fn, args, cudart_function_name, mock_return_on_failure)
_FAILURE_CASES = [
    (register_gl_image,     {"glo": 1, "target": 0x0DE1, "flags": 0}, "cudaGraphicsGLRegisterImage", ( 35, None)),
    (unregister_resource,   {"resource": object()}, "cudaGraphicsUnregisterResource",                (400,     )),
    (map_resources,         {"resource": object()}, "cudaGraphicsMapResources",                      ( 35,     )),
    (unmap_resources,       {"resource": object()}, "cudaGraphicsUnmapResources",                    ( 35,     )),
    (get_mapped_array,      {"resource": object()}, "cudaGraphicsSubResourceGetMappedArray",         (400, None)),
    (memcpy_2d_from_array,  {"dst_ptr": 0, "dst_pitch": 4, "src_array": object(), "width_bytes": 4, "height": 1},
        "cudaMemcpy2DFromArray", (35,)),
    (memcpy_2d_to_array,    {"dst_array": object(), "src_ptr": 0, "src_pitch": 4, "width_bytes": 4, "height": 1},
        "cudaMemcpy2DToArray",   (35,)),
    (memcpy_device_to_host, {"dst": 0, "src": 0, "count": 4}, "cudaMemcpy", (35,)),
]


@pytest.mark.parametrize(
    ("wrapper_fn", "kwargs", "cudart_fn_name", "mock_return"),
    _FAILURE_CASES,
    ids=[c[2] for c in _FAILURE_CASES],
)
def test_wrapper_raises_on_failure(wrapper_fn, kwargs, cudart_fn_name, mock_return):
    with patch(f"{_CUDART}.{cudart_fn_name}", return_value=mock_return):
        with pytest.raises(CUDAError) as exc_info:
            wrapper_fn(**kwargs)
        assert exc_info.value.error_code == mock_return[0]
