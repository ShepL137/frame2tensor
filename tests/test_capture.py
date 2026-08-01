"""Integration tests for X11 window capture.

These tests require a live X11 display with a compositor running. Run with:
    uv run pytest -m x11
"""
import time
from collections.abc import Iterator

import pytest

from frame2tensor.capture import X11Window
from frame2tensor.capture.window import X11Window as X11WindowDirect
from frame2tensor.types import CUDABuffer


@pytest.fixture(scope="module")
def x11_window_id() -> Iterator[int]:
    """Create a minimal visible X11 window for capture tests.

    The window is mapped and given a white background so that the compositor
    has content to composite before capture() is called.
    """
    from Xlib import X
    from Xlib import display as xdisplay

    d      = xdisplay.Display()
    screen = d.screen()
    root   = screen.root

    win = root.create_window(
        0, 0, 320, 240, 0,
        screen.root_depth,
        X.InputOutput,
        X.CopyFromParent,
        background_pixel=screen.white_pixel,
        event_mask=0,
    )
    win.map()
    d.sync()
    # Give the compositor a moment to redirect and render the window.
    time.sleep(0.3)

    yield win.id

    win.unmap()
    win.destroy()
    d.sync()
    d.close()


@pytest.mark.x11
class TestX11Window:
    def test_construction(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            assert win.width  == 320
            assert win.height == 240

    def test_capture_returns_cuda_buffer(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            buf = win.capture()
            assert isinstance(buf, CUDABuffer)

    def test_capture_shape(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            buf = win.capture()
            assert buf.shape   == (240, 320, 4)
            assert buf.typestr == "|u1"

    def test_capture_has_device_pointer(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            buf = win.capture()
            assert buf.ptr != 0

    def test_capture_cuda_array_interface(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            buf = win.capture()
            cai = buf.__cuda_array_interface__
            assert cai["shape"]   == (240, 320, 4)
            assert cai["typestr"] == "|u1"
            assert cai["version"] == 3

    def test_capture_multiple_frames(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            for _ in range(5):
                buf = win.capture()
                assert buf.ptr != 0

    def test_capture_borrowed_semantics_same_object(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            buf1 = win.capture()
            buf2 = win.capture()
            assert buf1 is buf2

    def test_context_manager(self, x11_window_id: int) -> None:
        with X11Window(x11_window_id) as win:
            assert not win._closed
        assert win._closed

    def test_close_is_idempotent(self, x11_window_id: int) -> None:
        win = X11Window(x11_window_id)
        win.close()
        win.close()  # must not raise

    def test_top_level_export(self, x11_window_id: int) -> None:
        assert X11Window is X11WindowDirect

    def test_capture_survives_source_resize(self, x11_window_id: int) -> None:
        """Resizing the source window must not abort; capture keeps producing frames at the new size."""
        from Xlib import display as xdisplay

        d   = xdisplay.Display()
        win = d.create_resource_object("window", x11_window_id)
        try:
            with X11Window(x11_window_id) as cap:
                cap.capture()

                win.configure(width=400, height=300)
                d.sync()
                time.sleep(0.3)  # let the compositor recomposite at the new size

                buf = cap.capture()
                for _ in range(3):
                    buf = cap.capture()  # must not crash across the resize

                assert isinstance(buf, CUDABuffer)
                assert (cap.width, cap.height) == (400, 300)
                assert buf.shape == (300, 400, 4)
        finally:
            # Restore the module-scoped fixture window for any later test.
            win.configure(width=320, height=240)
            d.sync()
            d.close()

    def test_revision_increments_on_resize(self, x11_window_id: int) -> None:
        """Revision bumps on a size change so consumers can rebuild size-dependent resources."""
        from Xlib import display as xdisplay

        d   = xdisplay.Display()
        win = d.create_resource_object("window", x11_window_id)
        try:
            with X11Window(x11_window_id) as cap:
                cap.capture()
                assert cap.revision == 0

                win.configure(width=400, height=300)
                d.sync()
                time.sleep(0.3)
                for _ in range(3):
                    cap.capture()

                assert cap.revision >= 1
                assert (cap.width, cap.height) == (400, 300)
        finally:
            win.configure(width=320, height=240)
            d.sync()
            d.close()

    def test_unmapped_source_holds_last_frame(self, x11_window_id: int) -> None:
        """While the source is unmapped (minimized), capture holds the last frame without crashing."""
        from Xlib import display as xdisplay

        d   = xdisplay.Display()
        win = d.create_resource_object("window", x11_window_id)
        try:
            with X11Window(x11_window_id) as cap:
                first = cap.capture()

                win.unmap()
                d.sync()
                time.sleep(0.3)

                held = cap.capture()  # must not read the freed pixmap or crash
                assert isinstance(held, CUDABuffer)
                assert held is first  # borrowed buffer held unchanged

                win.map()
                d.sync()
                time.sleep(0.3)
                assert isinstance(cap.capture(), CUDABuffer)  # resumes after remap
        finally:
            win.map()  # leave the module-scoped fixture window mapped
            d.sync()
            d.close()

    def test_destroyed_source_raises(self) -> None:
        """Destroying the source window raises CaptureSourceLostError on the next capture."""
        from Xlib import X
        from Xlib import display as xdisplay

        from frame2tensor.exceptions import CaptureSourceLostError

        d      = xdisplay.Display()
        screen = d.screen()
        win    = screen.root.create_window(
            0, 0, 160, 120, 0,
            screen.root_depth,
            X.InputOutput,
            X.CopyFromParent,
            background_pixel=screen.white_pixel,
            event_mask=0,
        )
        win.map()
        d.sync()
        time.sleep(0.3)

        cap = X11Window(win.id)
        try:
            cap.capture()
            win.destroy()
            d.sync()
            time.sleep(0.3)
            with pytest.raises(CaptureSourceLostError):
                cap.capture()
        finally:
            cap.close()
            d.close()

    def test_move_without_resize_keeps_capturing(self, x11_window_id: int) -> None:
        """A pure move keeps the same pixmap: no refresh, no revision bump, frames keep flowing."""
        from Xlib import display as xdisplay

        d   = xdisplay.Display()
        win = d.create_resource_object("window", x11_window_id)
        try:
            with X11Window(x11_window_id) as cap:
                cap.capture()
                rev_before = cap.revision

                for pos in (40, 80, 120, 0):  # several moves at the original size
                    win.configure(x=pos, y=pos)
                    d.sync()
                    time.sleep(0.1)
                    assert isinstance(cap.capture(), CUDABuffer)  # never freezes or crashes

                assert cap.revision == rev_before  # pure moves do not bump revision
                assert (cap.width, cap.height) == (320, 240)
        finally:
            win.configure(x=0, y=0)
            d.sync()
            d.close()
