"""Windowed rendering with optional CUDA extraction via GLFW."""
from collections.abc import Callable
from typing import Any, Self

import glfw
import moderngl as mgl
from moderngl_window.context.base.window import BaseWindow

from frame2tensor.exceptions import GLContextError
from frame2tensor.types import SupportsCUDAArray

from .render_target import RenderTarget


class WindowedRenderer(RenderTarget):
    """GLFW window with an offscreen FBO for optional CUDA extraction.

    Renders to an offscreen framebuffer, blits it to the window,
    and extracts the frame as a top-down :class:`~frame2tensor.types.CUDABuffer` for recording or model inference.
    Call write() beforehand to composite an existing CUDA buffer, such as a captured window, as the background layer.
    See :class:`~frame2tensor.render.RenderTarget` for the top-down convention.

    Usage::

        with WindowedRenderer(1280, 720) as win:
            while not win.is_closing:
                win.draw(my_draw)
                buf = win.capture()   # optional; top-down (row 0 = top) buffer of current frame
                win.swap()
    """

    def __init__(self, width: int, height: int, title: str = "render - frame2tensor", vsync: bool = True) -> None:
        """Create a GLFW window with an offscreen FBO for CUDA extraction.

        Args:
            width : Window width in pixels.
            height: Window height in pixels.
            title : Window title.
            vsync : If True, ``swap()`` blocks until the display's next vblank,
                    capping the render loop at the monitor's refresh rate.
                    Set False to decouple rendering from the display and limit frames manually.

        Raises:
            GLContextError: If GLFW window creation fails.
        """
        import moderngl_window as mglw
        from moderngl_window.conf import settings

        self._closed: bool              = False
        self._window: BaseWindow | None = None

        settings.WINDOW["class"]      = "moderngl_window.context.glfw.Window"
        settings.WINDOW["gl_version"] = (3, 3)
        settings.WINDOW["width"]      = width
        settings.WINDOW["height"]     = height
        settings.WINDOW["title"]      = title
        settings.WINDOW["vsync"]      = vsync

        try:
            self._window = mglw.create_window_from_settings()
        except Exception as e:
            raise GLContextError("GLFW window creation failed.") from e

        self._glfw_handle: Any = glfw.get_current_context()
        super().__init__(self._window.ctx, width, height)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    @property
    def is_closing(self) -> bool:
        """Whether the window has received a close request."""
        return self._window.is_closing  # pyright: ignore[reportOptionalMemberAccess]

    # -----------------------------------------------------------------------------

    def make_current(self) -> None:
        """Make this window's GL context current on the calling thread."""
        glfw.make_context_current(self._glfw_handle)

    def write(self, buffer: SupportsCUDAArray) -> None:
        """Store a CUDA buffer as the background for subsequent ``draw()`` calls."""
        self.make_current()
        super().write(buffer)

    def draw(self, on_draw: Callable[[mgl.Context, mgl.Framebuffer], None] | None = None) -> None:
        """Composite the written source (if any) into the FBO, call ``on_draw``, then blit to screen.

        After this call the FBO is in GL-native (bottom-up) orientation.
        Inside ``on_draw``, NDC (normalized device coordinates, -1 to 1) has Y=+1 at the visual top.
        The screen blit is a non-flipped copy because GL-native FBO and window surface share the same row convention.

        Call :meth:`~frame2tensor.render.RenderTarget.capture` before ``swap()`` if a GPU buffer of the frame is needed.

        Args:
            on_draw: Optional callback after the source blit.
        """
        self.make_current()
        super().draw(on_draw)
        self.ctx.copy_framebuffer(dst=self.ctx.screen, src=self.fbo)

    def swap(self) -> None:
        """Swap the front and back buffers, presenting the frame."""
        # moderngl_window's swap_buffers() also polls GLFW's event queue,
        # so key state and is_closing only reflect the latest input after this call returns.
        self._window.swap_buffers()  # pyright: ignore[reportOptionalMemberAccess]

    def close(self) -> None:
        """Release window and GPU resources."""
        if self._closed:
            return

        self._closed = True
        self.make_current()
        self._close_render_target()
        if self._window is not None:
            # Destroy only this window, not the whole GLFW library.
            # moderngl_window's destroy() is a bare glfw.terminate(),
            # which would invalidate every other GLFW context still open in the process and segfault its cleanup.
            # GLFW itself is reclaimed at process exit.
            glfw.destroy_window(self._glfw_handle)
            self._window = None
