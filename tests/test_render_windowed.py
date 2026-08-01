"""Windowed renderer tests.

These require a display server. Run with:
    uv run pytest -m windowed
"""
import subprocess
import sys
from pathlib import Path

import pytest

from frame2tensor.render import WindowedRenderer

_TEARDOWN_SCENARIO = Path(__file__).parent / "_dual_context_teardown.py"


@pytest.mark.windowed
class TestWindowedRendererResize:
    """WindowedRenderer.resize rebuilds the offscreen texture and framebuffer."""

    def test_resize_updates_dimensions(self) -> None:
        renderer = WindowedRenderer(64, 48, title="resize-test")
        try:
            assert (renderer.width, renderer.height) == (64, 48)
            renderer.resize(128, 96)
            assert (renderer.width, renderer.height) == (128, 96)
            assert renderer.fbo.size == (128, 96)
        finally:
            renderer.close()

    def test_resize_noop_when_unchanged(self) -> None:
        renderer = WindowedRenderer(64, 48, title="resize-noop")
        try:
            fbo_before = renderer.fbo
            renderer.resize(64, 48)
            assert renderer.fbo is fbo_before, "unchanged size should not rebuild the FBO"
        finally:
            renderer.close()

    def test_resize_rebuilds_capture(self) -> None:
        """capture() reflects the new size after resize (the lazy readback is rebuilt)."""
        renderer = WindowedRenderer(64, 48, title="resize-buffer")
        try:
            renderer.draw()
            assert renderer.capture().shape == (48, 64, 4)

            renderer.resize(80, 60)
            renderer.draw()
            assert renderer.capture().shape == (60, 80, 4)
        finally:
            renderer.close()


@pytest.mark.windowed
@pytest.mark.x11
class TestDualContextTeardown:
    """Pairing WindowedRenderer with X11Window must tear down without segfaulting.

    Regression for the teardown crash where moderngl_window's window destroy globally
    terminated GLFW and neither close() made its own GL context current.
    The pairing runs in a subprocess
    so a re-regression (SIGSEGV) shows up as a negative return code instead of taking down the whole test session.
    """

    def _run(self, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, str(_TEARDOWN_SCENARIO), mode],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_clean_teardown_exits_zero(self) -> None:
        result = self._run("clean")
        assert result.returncode == 0, f"returncode={result.returncode}\n{result.stderr}"

    def test_exception_unwind_reports_traceback(self) -> None:
        result = self._run("error")
        # A clean unwind exits 1 with the traceback; a teardown segfault would be negative (SIGSEGV).
        assert result.returncode == 1, f"returncode={result.returncode}\n{result.stderr}"
        assert "RuntimeError" in result.stderr
        assert "simulated model failure" in result.stderr
