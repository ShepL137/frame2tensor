"""Integration tests for the write() / draw() / capture() seam on a headless render target.

`CUDAWritableTexture.write()` is covered in `test_interop.py`; this covers `RenderTarget.write()`
above it, which chooses the source texture format and composites through the flip shader.
"""

import pytest
import torch
from torch._tensor import Tensor

from frame2tensor.exceptions import InvalidTensorError
from frame2tensor.render import EGLCanvas
from frame2tensor.render.render_target import _MGL_DTYPE
from frame2tensor.types import _ITEMSIZES, CUDAArrayInterface, CUDABuffer

_SIZE = 32

# Peak value per dtype: normalized float sources saturate at 1.0, uint8 at 255.
_PEAK: dict[str, float] = {"|u1": 255.0, "<f2": 1.0, "<f4": 1.0}


class _UnsupportedDtypeSource:
    """A valid CUDA array interface carrying a dtype the render path cannot map.

    The pointer is never dereferenced: `write()` rejects the dtype before any copy is issued.
    """

    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterface:
        return {
            "shape"  : (_SIZE, _SIZE, 4),
            "typestr": "<i4",
            "data"   : (0xDEAD, False),
            "version": 3,
        }


def _magenta_buffer(typestr: str, width: int = _SIZE, height: int = _SIZE) -> CUDABuffer:
    """Allocate an RGBA source filled with opaque magenta, so a channel swizzle cannot pass."""
    buffer: CUDABuffer = CUDABuffer(shape=(height, width, 4), typestr=typestr)
    peak  : float      = _PEAK[typestr]
    tensor: Tensor     = torch.as_tensor(data=buffer, device="cuda")
    tensor[..., 0]     = peak
    tensor[..., 1]     = 0
    tensor[..., 2]     = peak
    tensor[..., 3]     = peak

    return buffer


def _assert_magenta(captured: CUDABuffer) -> None:
    """Assert a captured frame is the opaque magenta written by `_magenta_buffer`."""
    frame: Tensor = torch.as_tensor(data=captured, device="cuda")
    assert (frame[:, :, 0] == 255).all(), "Red channel"
    assert (frame[:, :, 1] ==   0).all(), "Green channel"
    assert (frame[:, :, 2] == 255).all(), "Blue channel"
    assert (frame[:, :, 3] == 255).all(), "Alpha channel"


class TestDtypeSupport:
    """Every dtype the buffer layer accepts must survive the render path."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_dtype_tables_agree(self) -> None:
        """`_ITEMSIZES` and `_MGL_DTYPE` must list the same dtypes.

        These drifted once: `<f2` was accepted by `CUDABuffer` and every validation path,
        then raised `ValueError` the first time it reached a render target.
        """
        assert set(_ITEMSIZES) == set(_MGL_DTYPE)

    @pytest.mark.parametrize("typestr", ["|u1", "<f2", "<f4"])
    def test_supported_dtype_roundtrip(self, typestr: str) -> None:
        """A source of each supported dtype composites and reads back as RGBA8."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            buffer: CUDABuffer = _magenta_buffer(typestr)
            canvas.write(buffer)
            canvas.draw()

            captured: CUDABuffer = canvas.capture()
            assert captured.shape  == (_SIZE, _SIZE, 4)
            assert captured.typestr == "|u1"
            _assert_magenta(captured)

            buffer.close()

    def test_single_channel_source(self) -> None:
        """A single-channel source lands in red, with green and blue defaulted by the sampler."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            buffer: CUDABuffer = CUDABuffer(shape=(_SIZE, _SIZE), typestr="<f4")
            torch.as_tensor(data=buffer, device="cuda").fill_(1.0)

            canvas.write(buffer)
            canvas.draw()

            frame: Tensor = torch.as_tensor(data=canvas.capture(), device="cuda")
            assert (frame[:, :, 0] == 255).all(), "Red channel"
            assert (frame[:, :, 1] ==   0).all(), "Green channel"

            buffer.close()

    def test_unsupported_dtype_raises(self) -> None:
        """A dtype outside `_MGL_DTYPE` is rejected before any copy is issued."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas, pytest.raises(ValueError, match="unsupported dtype"):
            canvas.write(_UnsupportedDtypeSource())


class TestComposite:
    """Background persistence and orientation through the flip shader."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_background_persists_across_draws(self) -> None:
        """A written source is recomposited on every draw(), not consumed by the first."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            buffer: CUDABuffer = _magenta_buffer("|u1")
            canvas.write(buffer)

            canvas.draw()
            canvas.draw()
            _assert_magenta(canvas.capture())

            buffer.close()

    def test_orientation_is_preserved(self) -> None:
        """A top-down source captures back top-down: the two flips in the seam must cancel."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            buffer: CUDABuffer = CUDABuffer(shape=(_SIZE, _SIZE, 4), typestr="|u1")
            source: Tensor     = torch.as_tensor(data=buffer, device="cuda")
            source[:, :, :]                = 0
            source[: _SIZE // 2, :, 1]     = 255  # green band across the top half
            source[:, :, 3]                = 255

            canvas.write(buffer)
            canvas.draw()

            frame: Tensor = torch.as_tensor(data=canvas.capture(), device="cuda")
            assert (frame[: _SIZE // 2, :, 1] == 255).all(), "top half should stay on top"
            assert (frame[_SIZE // 2 :, :, 1] ==   0).all(), "bottom half should stay on the bottom"

            buffer.close()

    def test_on_draw_runs_over_the_background(self) -> None:
        """The `on_draw` callback paints after the source blit, so it wins where they overlap."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            buffer: CUDABuffer = _magenta_buffer("|u1")
            canvas.write(buffer)
            canvas.draw(on_draw=lambda ctx, _fbo: ctx.clear(red=0.0, green=1.0, blue=0.0, alpha=1.0))

            frame: Tensor = torch.as_tensor(data=canvas.capture(), device="cuda")
            assert (frame[:, :, 1] == 255).all(), "callback output should cover the background"
            assert (frame[:, :, 0] ==   0).all(), "background magenta should be gone"

            buffer.close()


class TestSourceRebuild:
    """The source texture is rebuilt when its geometry or component count changes."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self) -> None:
        pytest.importorskip(modname="torch")

    def test_shape_mismatch_raises(self) -> None:
        """A source whose dimensions differ from the target's is rejected."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            buffer: CUDABuffer = _magenta_buffer("|u1", width=_SIZE // 2, height=_SIZE // 2)
            with pytest.raises(InvalidTensorError):
                canvas.write(buffer)

            buffer.close()

    def test_component_count_change_rebuilds_source(self) -> None:
        """Switching between RGBA and single-channel sources rebuilds the texture rather than failing."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            rgba: CUDABuffer = _magenta_buffer("<f4")
            canvas.write(rgba)
            canvas.draw()
            _assert_magenta(canvas.capture())

            single: CUDABuffer = CUDABuffer(shape=(_SIZE, _SIZE), typestr="<f4")
            torch.as_tensor(data=single, device="cuda").fill_(1.0)
            canvas.write(single)
            canvas.draw()

            frame: Tensor = torch.as_tensor(data=canvas.capture(), device="cuda")
            assert (frame[:, :, 0] == 255).all(), "Red channel"
            assert (frame[:, :, 2] ==   0).all(), "Blue channel should be gone with the RGBA source"

            rgba.close()
            single.close()

    def test_write_after_resize(self) -> None:
        """resize() releases the source and readback; both rebuild at the new size on next use."""
        with EGLCanvas(width=_SIZE, height=_SIZE) as canvas:
            canvas.resize(width=_SIZE * 2, height=_SIZE * 2)

            buffer: CUDABuffer = _magenta_buffer("|u1", width=_SIZE * 2, height=_SIZE * 2)
            canvas.write(buffer)
            canvas.draw()

            captured: CUDABuffer = canvas.capture()
            assert captured.shape == (_SIZE * 2, _SIZE * 2, 4)
            _assert_magenta(captured)

            buffer.close()
