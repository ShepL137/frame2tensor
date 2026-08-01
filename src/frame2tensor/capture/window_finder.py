"""X11 window resolution: active window, or interactive pick.

Pure X11 protocol querying, independent of the XComposite/GLX/CUDA capture pipeline.
Each function resolves to a ``window_id: int``, the sole input :class:`~frame2tensor.capture.X11Window` needs.
"""
from frame2tensor.exceptions import WindowNotFoundError


def get_active_window(display: str | None = None) -> int:
    """Read the window manager's ``_NET_ACTIVE_WINDOW`` root property.

    Args:
        display: X11 display string. None uses $DISPLAY.

    Returns:
        X11 window XID of the active window.

    Raises:
        WindowNotFoundError: If nothing is focused, or the window manager does not support ``_NET_ACTIVE_WINDOW``.
    """
    from Xlib import X
    from Xlib import display as xdisplay

    try:
        d = xdisplay.Display(display=display)
    except Exception as e:
        raise WindowNotFoundError("X11 display connection failed.") from e

    try:
        root = d.screen().root
        prop = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
        if prop is None or not prop.value or prop.value[0] == 0:
            raise WindowNotFoundError(
                "No active window: nothing is focused, or the window manager does not support _NET_ACTIVE_WINDOW."
            )
        return int(prop.value[0])
    finally:
        d.close()


def pick_window() -> int:
    """Resolve a window by interactive click-to-select. Requires ``xwininfo`` on PATH.

    Blocks until the user clicks a window.

    Returns:
        X11 window XID of the clicked window.

    Raises:
        WindowNotFoundError: If selection is cancelled or ``xwininfo``'s output cannot be parsed.
    """
    import re
    import subprocess

    try:
        out = subprocess.check_output(["xwininfo", "-int"], text=True)  # noqa: S607
    except subprocess.CalledProcessError as e:
        raise WindowNotFoundError("Window selection cancelled.") from e

    match = re.search(r"Window id: (\d+)", out)
    if match is None:
        raise WindowNotFoundError("Could not parse xwininfo output.")

    return int(match.group(1))
