"""Headless frame capture via EGL."""
from typing import Self

import moderngl as mgl

from frame2tensor.exceptions import GLContextError

from .render_target import RenderTarget


class EGLCanvas(RenderTarget):
    """Headless EGL context that renders and captures frames as CUDA buffers.

    Render into it with :meth:`~frame2tensor.render.RenderTarget.draw`,
    then call :meth:`~frame2tensor.render.RenderTarget.capture` for a top-down :class:`~frame2tensor.types.CUDABuffer`.
    Call :meth:`~frame2tensor.render.RenderTarget.write` beforehand to composite an existing CUDA buffer
    as a background layer under the draw.
    See :class:`~frame2tensor.render.RenderTarget` for the top-down convention.

    Usage::

        with EGLCanvas(1920, 1080) as canvas:
            canvas.draw(my_draw)
            buf    = canvas.capture()
            tensor = torch.as_tensor(buf, device="cuda")   # (1080, 1920, 4) uint8, top-down (row 0 = top)
    """

    def __init__(self, width: int, height: int) -> None:
        """Create a headless EGL context with a CUDA-registered framebuffer.

        Args:
            width : Texture width in texels.
            height: Texture height in texels.

        Raises:
            GLContextError: If EGL context creation fails.
        """
        self._closed: bool = False
        try:
            ctx = mgl.create_context(standalone=True, backend="egl")  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise GLContextError("EGL context creation failed.") from e
        super().__init__(ctx, width, height)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    def close(self) -> None:
        """Release ModernGL context and GPU resources."""
        if self._closed:
            return
        self._closed = True
        self._close_render_target()
        self.ctx.release()
