"""Integration tests for the GL↔CUDA interop layer."""

import pytest
import torch
from torch._tensor import Tensor

from frame2tensor.exceptions import CUDAError, Frame2TensorError, InvalidTensorError
from frame2tensor.interop import CUDAWritableTexture, GLCUDATexture
from frame2tensor.render import EGLCanvas
from frame2tensor.types import CUDAArrayInterface, CUDABuffer


class _BadPointerTensor:
    """Fake tensor with a valid interface but invalid device pointer."""

    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterface:
        return {
            "shape"  : (32, 32),
            "typestr": "<f4",
            "data"   : (0xDEAD, False),
            "version": 3,
        }


class TestGLCUDATextureRead:
    """GL → tensor read path."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_solid_color_readback(self, gl_ctx, texture_factory) -> None:
        """Render solid red, read back as buffer, verify every pixel."""
        tex, fbo = texture_factory(64, 64, 4)

        fbo.use()
        gl_ctx.clear(1.0, 0.0, 0.0, 1.0)  # solid red

        with GLCUDATexture(texture=tex, width=64, height=64, components=4) as cuda_tex:
            buf   : CUDABuffer = cuda_tex.to_buffer()
            tensor: Tensor     = torch.as_tensor(data=buf, device="cuda")

            assert tensor.shape == (64, 64, 4)
            assert tensor.dtype == torch.uint8
            assert (tensor[:, :, 0] == 255).all(), "Red channel"
            assert (tensor[:, :, 1] ==   0).all(), "Green channel"
            assert (tensor[:, :, 2] ==   0).all(), "Blue channel"
            assert (tensor[:, :, 3] == 255).all(), "Alpha channel"

    def test_two_frames_not_stale(self, gl_ctx, texture_factory) -> None:
        """Render two different colors, verify each read reflects the change."""
        tex, fbo = texture_factory(64, 64, 4)

        with GLCUDATexture(texture=tex, width=64, height=64, components=4) as cuda_tex:
            fbo.use()
            gl_ctx.clear(1.0, 0.0, 0.0, 1.0)
            buf: CUDABuffer = cuda_tex.to_buffer()
            red: Tensor     = torch.as_tensor(data=buf, device="cuda").clone()

            fbo.use()
            gl_ctx.clear(0.0, 0.0, 1.0, 1.0)
            buf = cuda_tex.to_buffer()
            blue: Tensor = torch.as_tensor(data=buf, device="cuda")

            assert (red [:, :, 0] == 255).all()
            assert (blue[:, :, 2] == 255).all()
            assert (blue[:, :, 0] ==   0).all(), "Red channel should be zero after switching to blue"

    def test_borrowed_semantics(self, gl_ctx, texture_factory) -> None:
        """Two consecutive to_buffer() calls return the same underlying memory."""
        tex, fbo = texture_factory(32, 32, 4)

        fbo.use()
        gl_ctx.clear(0.0, 1.0, 0.0, 1.0)

        with GLCUDATexture(texture=tex, width=32, height=32, components=4) as cuda_tex:
            b1: CUDABuffer = cuda_tex.to_buffer()
            b2: CUDABuffer = cuda_tex.to_buffer()

            assert b1.ptr == b2.ptr


class TestEGLCanvas:
    """Headless capture end-to-end."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_capture_solid_color(self) -> None:
        """Create, render solid green, capture, verify output."""
        with EGLCanvas(width=64, height=64) as cap:
            cap.draw(on_draw=lambda ctx, _fbo: ctx.clear(red=0.0, green=1.0, blue=0.0, alpha=1.0))
            buf   : CUDABuffer = cap.capture()
            tensor: Tensor     = torch.as_tensor(data=buf, device="cuda")

            assert tensor.shape == (64, 64, 4)
            assert tensor.dtype == torch.uint8
            assert (tensor[:, :, 0] ==   0).all(), "Red channel"
            assert (tensor[:, :, 1] == 255).all(), "Green channel"
            assert (tensor[:, :, 2] ==   0).all(), "Blue channel"
            assert (tensor[:, :, 3] == 255).all(), "Alpha channel"


class TestResourceLifecycle:
    """Context managers and close idempotency."""

    def test_gl_cuda_texture_context_manager(self, gl_ctx, texture_factory) -> None:
        """Resource is released after exiting the context manager."""
        tex, fbo = texture_factory(32, 32, 4)

        with GLCUDATexture(texture=tex, width=32, height=32, components=4) as cuda_tex:
            fbo.use()
            gl_ctx.clear(1.0, 0.0, 0.0, 1.0)
            cuda_tex.to_buffer()

        assert cuda_tex._closed
        assert cuda_tex._resource is None

    def test_gl_cuda_texture_double_close(self, gl_ctx, texture_factory) -> None:
        """Calling close() twice does not raise."""
        tex, _fbo = texture_factory(32, 32, 4)

        cuda_tex: GLCUDATexture = GLCUDATexture(texture=tex, width=32, height=32, components=4)
        cuda_tex.close()
        cuda_tex.close()

    def test_frame_capture_context_manager(self) -> None:
        """EGLCanvas cleans up via context manager."""
        with EGLCanvas(width=32, height=32) as cap:
            cap.draw(on_draw=lambda ctx, _fbo: ctx.clear(red=0.0, green=0.0, blue=0.0, alpha=1.0))
            cap.capture()

        assert cap._closed

    def test_frame_capture_double_close(self) -> None:
        """EGLCanvas double close does not raise."""
        cap: EGLCanvas = EGLCanvas(width=32, height=32)
        cap.close()
        cap.close()

    def test_to_buffer_after_close_raises(self, gl_ctx, texture_factory) -> None:
        """to_buffer() on a closed GLCUDATexture raises Frame2TensorError."""
        tex, _ = texture_factory(32, 32, 4)
        cuda_tex: GLCUDATexture = GLCUDATexture(texture=tex, width=32, height=32, components=4)
        cuda_tex.close()

        with pytest.raises(Frame2TensorError):
            cuda_tex.to_buffer()


class TestMapUnmapSafety:
    """Failed operations must not leave resources mapped."""

    def test_failed_write_does_not_block_close(self, gl_ctx, texture_factory) -> None:
        """A write that fails mid-operation should still unmap, allowing clean close.

        Uses a fake tensor with an invalid device pointer, which causes
        the CUDA memcpy to fail after the resource is already mapped.
        """
        tex, _ = texture_factory(32, 32, 1, dtype='f4')

        writer    : CUDAWritableTexture = CUDAWritableTexture(texture=tex, width=32, height=32)
        bad_tensor: _BadPointerTensor   = _BadPointerTensor()

        with pytest.raises(CUDAError):
            writer.write(tensor=bad_tensor)

        # Resource should still be usable, close should not raise
        writer.close()

    def test_failed_write_allows_subsequent_write(self, gl_ctx, texture_factory) -> None:
        """After a failed write, a correct write should succeed.

        Uses a fake tensor with an invalid pointer to trigger the initial
        failure, then verifies a valid CUDA tensor can be written.
        """
        tex, _ = texture_factory(32, 32, 1, dtype='f4')

        writer     : CUDAWritableTexture = CUDAWritableTexture(texture=tex, width=32, height=32)
        bad_tensor : _BadPointerTensor   = _BadPointerTensor()
        good_tensor: Tensor              = torch.zeros(32, 32, dtype=torch.float32, device="cuda")

        with pytest.raises(CUDAError):
            writer.write(tensor=bad_tensor)

        # Should succeed; resource was properly unmapped after failure
        writer.write(tensor=good_tensor)
        writer.close()


class TestCUDAWritableTextureWrite:
    """Write path via __cuda_array_interface__."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_write_torch(self, gl_ctx, texture_factory) -> None:
        """Write a torch tensor, no errors."""
        tex, _ = texture_factory(32, 32, 1, dtype='f4')
        tensor: Tensor = torch.zeros(32, 32, dtype=torch.float32, device="cuda")

        with CUDAWritableTexture(texture=tex, width=32, height=32) as writer:
            writer.write(tensor)

    def test_write_wrong_shape(self, gl_ctx, texture_factory) -> None:
        tex, _ = texture_factory(32, 32, 1, dtype='f4')
        tensor: Tensor = torch.zeros(64, 64, dtype=torch.float32, device="cuda")

        writer: CUDAWritableTexture = CUDAWritableTexture(texture=tex, width=32, height=32)
        with pytest.raises(InvalidTensorError, match="expected shape"):
            writer.write(tensor)
        writer.close()

    def test_write_no_interface(self, gl_ctx, texture_factory) -> None:
        tex, _ = texture_factory(32, 32, 1, dtype='f4')

        writer: CUDAWritableTexture = CUDAWritableTexture(texture=tex, width=32, height=32)
        with pytest.raises(InvalidTensorError, match="__cuda_array_interface__"):
            writer.write(tensor="not a tensor")  # pyright: ignore[reportArgumentType]
        writer.close()

    def test_write_unsupported_dtype(self, gl_ctx, texture_factory) -> None:
        tex, _ = texture_factory(32, 32, 1, dtype='f4')
        tensor: Tensor = torch.zeros(32, 32, dtype=torch.float64, device="cuda")

        writer: CUDAWritableTexture = CUDAWritableTexture(texture=tex, width=32, height=32)
        with pytest.raises(InvalidTensorError, match="unsupported dtype"):
            writer.write(tensor)
        writer.close()

    def test_write_rejects_changed_shape(self, gl_ctx, texture_factory) -> None:
        """A source whose dimensions changed between writes is rejected, not copied out of bounds."""
        tex, _ = texture_factory(32, 32, 1, dtype='f4')
        initial: Tensor = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
        resized: Tensor = torch.zeros(16, 16, dtype=torch.float32, device="cuda")

        with CUDAWritableTexture(texture=tex, width=32, height=32) as writer:
            writer.write(initial)
            with pytest.raises(InvalidTensorError, match="expected shape"):
                writer.write(resized)


class TestWriteReadRoundtrip:
    """Tensor → GL → tensor round-trip."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_roundtrip_float32(self, gl_ctx, texture_factory) -> None:
        """Write a known pattern, read it back, verify values match."""
        tex, _ = texture_factory(32, 32, 1, dtype='f4')

        pattern: Tensor = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
        pattern[:16, :] = 1.0

        with CUDAWritableTexture(texture=tex, width=32, height=32, components=1) as writer:
            writer.write(tensor=pattern)

        with GLCUDATexture(texture=tex, width=32, height=32, components=1, typestr="<f4") as reader:
            buf   : CUDABuffer = reader.to_buffer()
            result: Tensor     = torch.as_tensor(data=buf, device="cuda")

            assert result.shape == (32, 32, 1)
            assert result.dtype == torch.float32
            assert (result[:16, :, 0] == 1.0).all(), "Top half should be 1.0"
            assert (result[16:, :, 0] == 0.0).all(), "Bottom half should be 0.0"

    def test_roundtrip_uint8_rgba(self, gl_ctx, texture_factory) -> None:
        """Write and read back a 4-component uint8 pattern."""
        tex, _ = texture_factory(32, 32, 4)

        # Red pixels: (255, 0, 0, 255)
        pattern: Tensor = torch.zeros(32, 32, 4, dtype=torch.uint8, device="cuda")
        pattern[:, :, 0] = 255
        pattern[:, :, 3] = 255

        with CUDAWritableTexture(texture=tex, width=32, height=32, components=4) as writer:
            writer.write(tensor=pattern)

        with GLCUDATexture(texture=tex, width=32, height=32, typestr="|u1") as reader:
            buf   : CUDABuffer = reader.to_buffer()
            result: Tensor     = torch.as_tensor(data=buf, device="cuda")

            assert (result[:, :, 0] == 255).all(), "Red channel"
            assert (result[:, :, 1] ==   0).all(), "Green channel"
            assert (result[:, :, 2] ==   0).all(), "Blue channel"
            assert (result[:, :, 3] == 255).all(), "Alpha channel"

    def test_write_non_contiguous(self, gl_ctx, texture_factory) -> None:
        """Non-contiguous tensors are rejected by write()."""
        tex   , _                   = texture_factory(32, 32, 1, dtype='f4')
        tensor: Tensor              = torch.zeros(64, 32, dtype=torch.float32, device="cuda")
        sliced: Tensor              = tensor[::2, :]    # non-contiguous: shape (32, 32) but stride (64, 1)
        writer: CUDAWritableTexture = CUDAWritableTexture(texture=tex, width=32, height=32)

        with pytest.raises(InvalidTensorError, match="contiguous"):
            writer.write(tensor=sliced)
        writer.close()
