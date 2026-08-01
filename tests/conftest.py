"""Shared fixtures for GPU integration tests."""

from collections.abc import Callable, Generator
from typing import Any

import moderngl as mgl
import pytest
from moderngl import Context


@pytest.fixture(scope="session")
def gl_ctx() -> Generator[Context]:
    """Headless EGL context, shared across the entire test session."""
    ctx: Context = mgl.create_context(standalone=True, backend="egl")  # pyright: ignore[reportArgumentType]
    yield ctx
    ctx.release()


@pytest.fixture
def texture_factory(gl_ctx) -> Generator[Callable[..., tuple[Any, Any]]]:
    """Factory that creates a texture + framebuffer pair for a single test.

    Returns a callable: `make(width, height, components) -> (texture, fbo)`
    """
    created: list[Any] = []

    def make(width=64, height=64, components=4, dtype='f1') -> tuple[Any, Any]:
        tex: Any = gl_ctx.texture((width, height), components, dtype=dtype)
        tex.filter = (mgl.NEAREST, mgl.NEAREST)
        fbo: Any = gl_ctx.framebuffer(color_attachments=[tex])
        created.append((tex, fbo))
        return tex, fbo

    yield make

    for tex, fbo in created:
        fbo.release()
        tex.release()
