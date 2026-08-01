"""Window capture via XComposite."""
from .window import X11Window
from .window_finder import get_active_window, pick_window

__all__ = ["X11Window", "get_active_window", "pick_window"]
