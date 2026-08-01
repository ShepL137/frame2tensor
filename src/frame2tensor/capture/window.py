"""X11 window capture via XComposite and GLX."""
from typing import Self

from frame2tensor.exceptions import CaptureSourceLostError, GLContextError
from frame2tensor.types import CUDABuffer

from .glx_context import GLXCaptureContext
from .xcomposite import XCompositeContext


class X11Window:
    """Captures frames from an X11 window.

    Returns frames as :class:`~frame2tensor.types.CUDABuffer` objects.
    Uses XComposite to redirect the window to an offscreen pixmap, binds it as a GL texture,
    and downloads frames via a pixel buffer object on the GPU copy engine (one on-device copy per frame).

    All GL and CUDA operations must happen on the thread that created this object.

    Usage::

        with X11Window(window_id) as win:
            while True:
                buf    = win.capture()
                tensor = torch.as_tensor(buf, device="cuda")  # (H, W, 4) uint8
                result = model(tensor.clone())  # clone before next capture()
    """

    def __init__(self, window_id: int, display: str | None = None) -> None:
        """Redirect the window and set up the capture pipeline.

        Args:
            window_id: X11 window XID to capture.
            display  : X11 display string. None uses $DISPLAY.

        Raises:
            GLContextError: If X11, XComposite, or GLX setup fails.
            CUDAError     : If CUDA PBO registration fails.
        """
        self._closed   : bool              = False
        self._revision : int               = 0
        self._window_id: int               = window_id
        self._xcomp    : XCompositeContext = XCompositeContext(window_id, display)
        try:
            self._glx: GLXCaptureContext = GLXCaptureContext(
                pixmap_id = self._xcomp.pixmap_id,
                visual_id = self._xcomp.visual_id,
                width     = self._xcomp.width,
                height    = self._xcomp.height,
            )
        except Exception:
            self._xcomp.close()
            raise
        self._buffer: CUDABuffer = CUDABuffer(
            shape   = (self._xcomp.height, self._xcomp.width, 4),
            typestr = "|u1",
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._xcomp.width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._xcomp.height

    @property
    def revision(self) -> int:
        """Monotonic counter that increments whenever the frame size changes.

        Poll this between frames to detect a source resize
        and rebuild any size-dependent consumers (textures, writable textures)::

            buf = win.capture()
            if win.revision != seen:
                seen = win.revision
                # rebuild consumers to win.width x win.height
        """
        return self._revision

    # -----------------------------------------------------------------------------

    def make_current(self) -> None:
        """Make the capture GL context current on the calling thread.

        ``capture()`` calls this automatically, so manual calls are only needed for other GL operations between frames.
        """
        self._glx.make_current()

    def capture(self) -> CUDABuffer:
        """Capture the current frame and return a GPU buffer.

        Drains pending structure events before each capture.
        If the source window moved or was resized, the GLX pixmap is recreated transparently.
        On resize, the internal :class:`~frame2tensor.types.CUDABuffer` is also reallocated to the new dimensions
        and ``revision`` is bumped so consumers can rebuild size-dependent resources.

        While the source is minimized or otherwise unmapped, the last frame is held and returned unchanged;
        capture resumes automatically when the window is mapped again.

        Returns:
            A reference to the internal :class:`~frame2tensor.types.CUDABuffer`.

        Raises:
            CaptureSourceLostError: If the source window has been destroyed.
            GLContextError        : If the GL pipeline fails.
            CUDAError             : If CUDA operations fail.
        """
        self._glx.make_current()
        pixmap_changed, size_changed = self._xcomp.poll_configure()
        if self._xcomp.destroyed:
            raise CaptureSourceLostError(f"source window 0x{self._window_id:x} was destroyed")
        if not self._xcomp.mapped:
            return self._buffer  # minimized or unmapped: hold the last frame until remap
        if pixmap_changed:
            try:
                self._glx.update_pixmap(
                    pixmap_id = self._xcomp.pixmap_id,
                    width     = self._xcomp.width,
                    height    = self._xcomp.height,
                    resize    = size_changed,
                )
            except GLContextError:
                # update_pixmap builds the new pixmap before destroying the old,
                # so the previous one is still bound on failure.
                # return the last good frame and retry on the next capture() rather than crashing.
                return self._buffer
            if size_changed:
                self._buffer.close()
                self._buffer = CUDABuffer(
                    shape   = (self._xcomp.height, self._xcomp.width, 4),
                    typestr = "|u1",
                )
                self._revision += 1
        self._glx.download(self._buffer.ptr)
        return self._buffer

    def close(self) -> None:
        """Release all GL, CUDA, and X11 resources."""
        if self._closed:
            return
        self._closed = True
        self._buffer.close()
        self._glx.close()
        self._xcomp.close()
