"""Tests for frame2tensor.exceptions."""

import pytest

from frame2tensor.exceptions import (
    CUDAError,
    Frame2TensorError,
    GLContextError,
    InvalidTensorError,
    OutputError,
    WindowNotFoundError,
)


class TestHierarchy:
    """All exceptions funnel into Frame2TensorError → RuntimeError."""

    @pytest.mark.parametrize(
        "exc_class",
        [CUDAError, GLContextError, InvalidTensorError, OutputError, WindowNotFoundError],
    )
    def test_subclasses_frame2tensor_error(self, exc_class):
        assert issubclass(exc_class, Frame2TensorError)

    def test_frame2tensor_error_is_runtime_error(self):
        assert issubclass(Frame2TensorError, RuntimeError)

    @pytest.mark.parametrize(
        "exc_class",
        [CUDAError, GLContextError, InvalidTensorError, OutputError, WindowNotFoundError],
    )
    def test_catch_all(self, exc_class):
        """A bare `except Frame2TensorError` catches every subclass."""
        err = exc_class("op", 1) if exc_class is CUDAError else exc_class("something went wrong")

        with pytest.raises(Frame2TensorError):
            raise err


class TestCUDAError:
    """CUDAError stores structured attributes for programmatic use."""

    def test_attributes(self):
        err = CUDAError("register_gl_image", 35)
        assert err.operation == "register_gl_image"
        assert err.error_code == 35

    def test_str_is_nonempty(self):
        err = CUDAError("register_gl_image", 35)
        assert str(err)

    @pytest.mark.parametrize("code", [1, 2, 3, 35, 100, 101, 208, 400, 700])
    def test_known_error_codes(self, code):
        err = CUDAError("op", code)
        assert err.error_code == code
        assert str(err)

    def test_unknown_error_code(self):
        err = CUDAError("op", 99999)
        assert err.error_code == 99999
        assert str(err)


class TestGLContextError:
    def test_message_roundtrip(self):
        msg = "EGL context creation failed"
        err = GLContextError(msg)
        assert str(err) == msg


class TestInvalidTensorError:
    def test_message_roundtrip(self):
        msg = "expected shape (32, 32), got (64, 64)"
        err = InvalidTensorError(msg)
        assert str(err) == msg


class TestOutputError:
    def test_message_roundtrip(self):
        msg = "pipe write failed: broken pipe"
        err = OutputError(msg)
        assert str(err) == msg


class TestWindowNotFoundError:
    def test_message_roundtrip(self):
        msg = "no window matches class 'firefox'"
        err = WindowNotFoundError(msg)
        assert str(err) == msg
