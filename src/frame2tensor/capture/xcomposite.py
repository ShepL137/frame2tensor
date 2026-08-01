"""XComposite window redirect and pixmap management."""
import contextlib
from typing import Any, NamedTuple, Self

from frame2tensor.exceptions import GLContextError


class ConfigureResult(NamedTuple):
    """Result of an ``XCompositeContext.poll_configure()`` poll."""

    pixmap_changed: bool
    """True when the GLX pixmap was refreshed; the caller must recreate the GLXPixmap."""
    size_changed: bool
    """True when width or height also changed; PBO and :class:`~frame2tensor.types.CUDABuffer` must be reallocated."""


class XCompositeContext:
    """X11 display connection with XComposite redirect for one window.

    Redirects the target window to an offscreen pixmap via XComposite and names that pixmap for use as a GLXPixmap.

    All attributes are read-only after construction.
    """

    def __init__(self, window_id: int, display_name: str | None = None) -> None:
        """Open an X11 connection and redirect a window to an offscreen pixmap.

        Args:
            window_id   : X11 window XID to capture.
            display_name: X11 display string. None uses $DISPLAY.

        Raises:
            GLContextError: If XComposite is unavailable or pixmap setup fails.
        """
        from Xlib import display as xdisplay
        from Xlib.ext import composite

        self._closed         : bool = False
        self._display        : Any  = None
        self._window         : Any  = None
        self._pixmap_id      : int  = 0
        self._visual_id      : int  = 0
        self._width          : int  = 0
        self._height         : int  = 0
        self._refresh_pending: bool = False
        self._mapped         : bool = True
        self._destroyed      : bool = False

        try:
            d = xdisplay.Display(display=display_name)
        except Exception as e:
            raise GLContextError("X11 display connection failed.") from e

        if d.query_extension(name="Composite") is None:
            d.close()
            raise GLContextError("XComposite extension not available on this display.")

        self._display = d

        window = d.create_resource_object(type="window", id=window_id)
        self._window = window
        composite.redirect_window(window, update=composite.RedirectAutomatic)  # pyright: ignore[reportArgumentType]

        # python-xlib 0.33 may raise BadRRCrtcError (missing sequence_number attr)
        # for unrelated RandR events; suppress to avoid spurious failures.
        with contextlib.suppress(Exception):
            d.sync()

        try:
            geom = window.get_geometry()
        except Exception as e:
            raise GLContextError("Cannot query window geometry; is window_id valid?") from e

        self._width  = geom.width
        self._height = geom.height

        attrs = window.get_attributes()
        self._visual_id = attrs.visual

        # Receive ConfigureNotify so we can detect resize/move and refresh the pixmap.
        from Xlib import X
        window.change_attributes(event_mask=X.StructureNotifyMask)

        pixmap = composite.name_window_pixmap(window)
        with contextlib.suppress(Exception):
            d.sync()

        try:
            _ = d.create_resource_object(type="pixmap", id=pixmap.id).get_geometry()
        except Exception as e:
            raise GLContextError("XCompositeNameWindowPixmap failed.") from e

        self._pixmap_id = pixmap.id

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    @property
    def pixmap_id(self) -> int:
        """X11 pixmap XID for use with glXCreatePixmap."""
        return self._pixmap_id

    @property
    def visual_id(self) -> int:
        """Visual ID of the captured window, for FBConfig matching."""
        return self._visual_id

    @property
    def width(self) -> int:
        """Window width in pixels at construction time."""
        return self._width

    @property
    def height(self) -> int:
        """Window height in pixels at construction time."""
        return self._height

    @property
    def mapped(self) -> bool:
        """Whether the source window is currently mapped (viewable).

        False while the window is minimized or otherwise unmapped,
        during which the compositor has no valid backing pixmap and capture should hold the last frame.
        """
        return self._mapped

    @property
    def destroyed(self) -> bool:
        """Whether the source window has been destroyed (DestroyNotify seen)."""
        return self._destroyed

    # -----------------------------------------------------------------------------

    def poll_configure(self) -> ConfigureResult:
        """Drain pending structure events and refresh the pixmap if needed.

        Must be called before each frame capture, so a resize or move doesn't leave the GLX pixmap
        pointing at a dead X resource.

        A size change or a remap (MapNotify) marks the pixmap stale and pending refresh;
        a pure move does not, and nothing refreshes while ``destroyed`` or not ``mapped``.

        Raises:
            GLContextError: If called after ``close()``.

        Returns:
            ``ConfigureResult``. Both fields are False when nothing changed or no pending refresh; retried next call.
        """
        from Xlib import X
        from Xlib.ext import composite

        if self._display is None or self._window is None:
            raise GLContextError("poll_configure called on a closed XCompositeContext.")

        while self._display.pending_events() > 0:
            with contextlib.suppress(Exception):
                event = self._display.next_event()
                match event.type:
                    case X.ConfigureNotify:
                        # Per XComposite, the named pixmap stays valid across moves; only a size change invalidates it.
                        if event.width != self._width or event.height != self._height:
                            self._refresh_pending = True
                    case X.MapNotify:
                        # The compositor allocates a fresh pixmap on remap, so treat it as an invalidation too.
                        self._mapped          = True
                        self._refresh_pending = True
                    case X.UnmapNotify:
                        self._mapped          = False
                    case X.DestroyNotify:
                        self._destroyed       = True
                    case _:
                        pass

        if self._destroyed or not self._mapped or not self._refresh_pending:
            return ConfigureResult(pixmap_changed=False, size_changed=False)

        pixmap = composite.name_window_pixmap(self._window)
        with contextlib.suppress(Exception):
            self._display.sync()

        geom = None
        with contextlib.suppress(Exception):
            # If a newly named pixmap is not committed, the server raises BadDrawable here.
            # Leave the refresh pending so that this retries next call.
            geom = self._display.create_resource_object(type="pixmap", id=pixmap.id).get_geometry()
        if geom is None:
            return ConfigureResult(pixmap_changed=False, size_changed=False)

        self._refresh_pending = False
        self._pixmap_id       = pixmap.id

        size_changed = geom.width != self._width or geom.height != self._height
        self._width  = geom.width
        self._height = geom.height

        return ConfigureResult(pixmap_changed=True, size_changed=size_changed)

    def close(self) -> None:
        """Unredirect the window and close the X11 connection."""
        if self._closed:
            return
        self._closed = True

        if self._display is not None and self._window is not None:
            from Xlib.ext import composite
            with contextlib.suppress(Exception):
                composite.unredirect_window(self._window, update=composite.RedirectAutomatic)  # pyright: ignore[reportArgumentType]
            with contextlib.suppress(Exception):
                self._display.close()

        self._display = None
        self._window  = None
