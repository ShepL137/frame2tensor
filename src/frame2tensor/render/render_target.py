"""Shared render-target base."""
import struct
from collections.abc import Callable
from typing import NamedTuple

import moderngl as mgl
from OpenGL import GL

from frame2tensor.interop import CUDAWritableTexture, GLCUDATexture
from frame2tensor.types import CUDABuffer, SupportsCUDAArray

_BLIT_VERT = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_BLIT_FRAG = """
#version 330
uniform sampler2D source;
out vec4 out_color;
void main() {
    vec2 uv   = gl_FragCoord.xy / vec2(textureSize(source, 0));
    uv.y      = 1.0 - uv.y;
    out_color = texture(source, uv);
}
"""

# Full-screen quad as a triangle strip: BL, BR, TL, TR
_QUAD_VERTS: list[float]    = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
_MGL_DTYPE : dict[str, str] = {"|u1": "f1", "<f2": "f2", "<f4": "f4"}


class _Readback(NamedTuple):
    """Y-flipped blit target and its CUDA reader, always created and released together."""
    fbo      : mgl.Framebuffer
    cuda_read: GLCUDATexture


class RenderTarget:
    """Shared FBO, source texture, and CUDA interop, base for the library's render targets.

    GL's own framebuffer storage is bottom-up,
    but everything else the library touches (window capture, an image tensor, a video file) is top-down.
    ``draw()`` flips a written source to match GL's orientation as it composites it into the FBO,
    and ``capture()`` flips the FBO back, so a top-down buffer in always produces a top-down buffer out.

    The FBO is fixed at RGBA8 (4-component uint8); ``capture()`` returns and ``write()`` accepts
    only this format until broader pixel-format support is implemented.

    Subclasses (:class:`~frame2tensor.render.EGLCanvas`, :class:`~frame2tensor.render.WindowedRenderer`)
    create their GL context then call ``super().__init__()``,
    and call ``_close_render_target()`` from their own ``close()`` before releasing the context.
    """
    def __init__(self, ctx: mgl.Context, width: int, height: int) -> None:
        """Build the FBO from the subclass's GL context. Other resources are created lazily on first use.

        Called by subclasses after they create their own GL context.

        Args:
            ctx   : The subclass's ModernGL context.
            width : Framebuffer width in texels.
            height: Framebuffer height in texels.
        """
        self.ctx         : mgl.Context                = ctx
        self.width       : int                        = width
        self.height      : int                        = height

        self._fbo_texture: mgl.Texture                = ctx.texture(size=(width, height), components=4)
        self._fbo_texture.filter                      = (mgl.NEAREST, mgl.NEAREST)
        self.fbo         : mgl.Framebuffer            = ctx.framebuffer(color_attachments=[self._fbo_texture])

        self._source_tex : mgl.Texture         | None = None
        self._cuda_write : CUDAWritableTexture | None = None

        self._readback   : _Readback           | None = None
        self._read_tex   : mgl.Texture         | None = None

        self._bg_prog    : mgl.Program         | None = None
        self._bg_vao     : mgl.VertexArray     | None = None
        self._bg_vbo     : mgl.Buffer          | None = None

    # -----------------------------------------------------------------------------

    def write(self, buffer: SupportsCUDAArray) -> None:
        """Store a top-down CUDA buffer as the background for subsequent ``draw()`` calls.

        The buffer is recomposited into the FBO every ``draw()``
        until the next ``write()``, ``resize()``, or ``close()``.
        Calling ``draw()`` repeatedly without writing again just redraws the same background.
        The buffer's shape and dtype determine the source texture format on first call.
        Subsequent calls reuse the source texture; rebuilt automatically after ``resize()`` or component count changes.

        Args:
            buffer: Any object exposing ``__cuda_array_interface__``.
                    Shape ``(H, W, C)`` or ``(H, W)`` matching this target's current dimensions.

        Raises:
            ValueError        : If buffer's dtype is not uint8 or float32.
            InvalidTensorError: If shape or contiguity do not match.
            CUDAError         : If the device-to-device copy fails.
        """
        interface  = buffer.__cuda_array_interface__
        shape      = interface["shape"]
        typestr    = interface["typestr"]
        components = shape[2] if len(shape) == 3 else 1
        mgl_dtype  = _MGL_DTYPE.get(typestr)
        if mgl_dtype is None:
            raise ValueError(
                f"unsupported dtype for write(): {typestr!r}  supported: {sorted(_MGL_DTYPE)}"
            )
        self._ensure_source(components, mgl_dtype).write(buffer)

    def draw(self, on_draw: Callable[[mgl.Context, mgl.Framebuffer], None] | None = None) -> None:
        """Composite the written source (if any) into the FBO, then call ``on_draw``.

        If ``write()`` has been called (on this call or any earlier one),
        its source texture is recomposited first via a full-screen quad shader that applies a vertical flip.
        ``on_draw`` then runs with NDC (normalized device coordinates, -1 to 1) Y=+1 as the visual top.

        After this call the FBO is in GL-native (bottom-up) orientation.
        Call ``capture()`` to extract a GPU buffer of the frame.
        On a :class:`~frame2tensor.render.WindowedRenderer`, do this before
        :meth:`~frame2tensor.render.WindowedRenderer.swap`.

        Args:
            on_draw: Optional callback after the source blit.
        """
        self.fbo.use()
        if self._source_tex is not None:
            self._source_tex.use(0)
            self._ensure_bg_shader().render(mgl.TRIANGLE_STRIP, vertices=4)
        if on_draw is not None:
            on_draw(self.ctx, self.fbo)

    def capture(self) -> CUDABuffer:
        """Return the current FBO as a top-down CUDABuffer.

        Applies a Y-flip via ``glBlitFramebuffer`` to an internal read texture so that row 0 is the visual top.

        Call after ``draw()``. On a :class:`~frame2tensor.render.WindowedRenderer`,
        call before :meth:`~frame2tensor.render.WindowedRenderer.swap`.

        Returns:
            A reference to the internal :class:`~frame2tensor.types.CUDABuffer`.

        Raises:
            CUDAError: If the GL-CUDA copy fails.
        """
        readback = self._ensure_readback()
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo.glo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, readback.fbo.glo)
        GL.glBlitFramebuffer(
            0, self.height, self.width, 0,
            0, 0, self.width, self.height,
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST,
        )
        return readback.cuda_read.to_buffer()

    def resize(self, width: int, height: int) -> None:
        """Resize the FBO, releasing the source texture and readback.

        The source texture rebuilds on the next ``write()``,
        and the readback on the next ``capture()``; both, at the new dimensions.
        The background shader program is size-independent and is reused.

        Args:
            width : New framebuffer width in pixels.
            height: New framebuffer height in pixels.
        """
        if width == self.width and height == self.height:
            return

        self.width  = width
        self.height = height
        self._release_source()
        self._release_readback()
        self.fbo.release()
        self._fbo_texture.release()
        self._fbo_texture        = self.ctx.texture(size=(width, height), components=4)
        self._fbo_texture.filter = (mgl.NEAREST, mgl.NEAREST)
        self.fbo                 = self.ctx.framebuffer(color_attachments=[self._fbo_texture])

    # -----------------------------------------------------------------------------

    def _close_render_target(self) -> None:
        """Release the source texture, readback, background shader, and FBO."""
        self._release_source()
        self._release_readback()

        if self._bg_vbo is not None:
            self._bg_vbo.release()
            self._bg_vbo = None

        if self._bg_vao is not None:
            self._bg_vao.release()
            self._bg_vao = None

        if self._bg_prog is not None:
            self._bg_prog.release()
            self._bg_prog = None

        self.fbo.release()
        self._fbo_texture.release()

    def _ensure_source(self, components: int, mgl_dtype: str) -> CUDAWritableTexture:
        """Return the CUDA-writable source texture, (re)creating it if absent or if components changed."""
        cuda_write = self._cuda_write

        if cuda_write is None or self._source_tex is None or self._source_tex.components != components:
            self._release_source()
            source_tex        = self.ctx.texture(size=(self.width, self.height), components=components, dtype=mgl_dtype)
            source_tex.filter = (mgl.NEAREST, mgl.NEAREST)
            cuda_write        = CUDAWritableTexture(
                texture    = source_tex,
                width      = self.width,
                height     = self.height,
                components = components,
            )
            self._source_tex  = source_tex
            self._cuda_write  = cuda_write

        return cuda_write

    def _ensure_bg_shader(self) -> mgl.VertexArray:
        """Return the background-blit shader, compiling it lazily on first use."""
        vao = self._bg_vao
        if vao is None:
            prog           = self.ctx.program(vertex_shader=_BLIT_VERT, fragment_shader=_BLIT_FRAG)
            prog["source"] = 0
            vbo            = self.ctx.buffer(data=struct.pack("8f", *_QUAD_VERTS))
            vao            = self.ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
            self._bg_prog  = prog
            self._bg_vbo   = vbo
            self._bg_vao   = vao
        return vao

    def _ensure_readback(self) -> _Readback:
        """Return the Y-flip readback target, creating it lazily on first use."""
        readback = self._readback
        if readback is None:
            tex              = self.ctx.texture(size=(self.width, self.height), components=4)
            tex.filter       = (mgl.NEAREST, mgl.NEAREST)
            fbo              = self.ctx.framebuffer(color_attachments=[tex])
            cuda_read        = GLCUDATexture(
                texture    = tex,
                width      = self.width,
                height     = self.height,
                components = 4,
            )
            readback         = _Readback(fbo=fbo, cuda_read=cuda_read)
            self._read_tex   = tex
            self._readback   = readback
        return readback

    def _release_source(self) -> None:
        """Close and clear the CUDA-writable source texture, if any."""
        if self._cuda_write is not None:
            self._cuda_write.close()
            self._cuda_write = None

        if self._source_tex is not None:
            self._source_tex.release()
            self._source_tex = None

    def _release_readback(self) -> None:
        """Close and clear the Y-flip readback target, if any."""
        if self._readback is not None:
            self._readback.cuda_read.close()
            self._readback.fbo.release()
            self._readback = None

        if self._read_tex is not None:
            self._read_tex.release()
            self._read_tex = None
