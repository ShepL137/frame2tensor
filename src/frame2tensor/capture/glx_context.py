"""Isolated GLX context for XComposite texture capture via PBO."""
import contextlib
import ctypes
from collections.abc import Generator
from ctypes import POINTER, byref, c_int, c_long, c_ulong, c_void_p, cast
from typing import Any, ClassVar, Self

import glfw
from OpenGL import GL
from OpenGL.GLX import (
    glXChooseFBConfig,
    glXCreatePixmap,
    glXDestroyPixmap,
    glXGetFBConfigAttrib,
    glXQueryExtensionsString,
)
from OpenGL.GLX.EXT.texture_from_pixmap import (
    GLX_BIND_TO_TEXTURE_RGBA_EXT,
    GLX_BIND_TO_TEXTURE_TARGETS_EXT,
    GLX_FRONT_EXT,
    GLX_TEXTURE_2D_BIT_EXT,
    GLX_TEXTURE_2D_EXT,
    GLX_TEXTURE_FORMAT_EXT,
    GLX_TEXTURE_FORMAT_RGBA_EXT,
    GLX_TEXTURE_TARGET_EXT,
)

from frame2tensor._cuda import (
    READ_ONLY,
    get_mapped_pointer,
    map_resources,
    memcpy,
    register_gl_buffer,
    set_device,
    unmap_resources,
    unregister_resource,
)
from frame2tensor.exceptions import GLContextError

# PyOpenGL's lazy loader checks GL_EXTENSIONS, not the GLX extension string.
# glXBindTexImageEXT/glXReleaseTexImageEXT are GLX-only and will always appear undefined to PyOpenGL.
# So, we load from libGL directly.
_lib_gl                                = ctypes.CDLL("libGL.so.1")
_lib_gl.glXBindTexImageEXT.restype     = None
_lib_gl.glXBindTexImageEXT.argtypes    = [c_void_p, c_ulong, c_int, c_void_p]
_lib_gl.glXReleaseTexImageEXT.restype  = None
_lib_gl.glXReleaseTexImageEXT.argtypes = [c_void_p, c_ulong, c_int]

# GLX constants absent from PyOpenGL's bindings.
_GLX_DRAWABLE_TYPE = 0x8010
_GLX_DOUBLEBUFFER  = 5
_GLX_VISUAL_ID     = 0x800B
_GLX_PIXMAP_BIT    = 2

# X protocol errors from the GLX calls below (glXCreatePixmap / glXBindTexImageEXT / glXDestroyPixmap)
# arrive asynchronously on libGL's X connection, whose default Xlib handler aborts the process.
# They occur transiently while the source window is resized, moved, or minimized (per Claude).
# Install a no-op handler around those calls, then XSync so the errors are absorbed rather than fatal.
_lib_x11                           = ctypes.CDLL("libX11.so.6")
_XErrorHandler                     = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p)
_lib_x11.XSetErrorHandler.restype  = _XErrorHandler
_lib_x11.XSetErrorHandler.argtypes = [_XErrorHandler]
_lib_x11.XSync.restype             = c_int
_lib_x11.XSync.argtypes            = [c_void_p, c_int]


@_XErrorHandler
def _silence_x_errors(_display: Any, _error: Any) -> int:
    return 0


class _XVisualInfo(ctypes.Structure):
    _fields_: ClassVar[Any] = (
        ("visual"       , c_void_p),
        ("visualid"     , c_ulong),
        ("screen"       , c_int),
        ("depth"        , c_int),
        ("class_"       , c_int),
        ("red_mask"     , c_ulong),
        ("green_mask"   , c_ulong),
        ("blue_mask"    , c_ulong),
        ("colormap_size", c_int),
        ("bits_per_rgb" , c_int),
    )


_VISUAL_ID_MASK                    = 0x1  # VisualIDMask, from Xlib.h
_lib_x11.XGetVisualInfo.restype    = POINTER(_XVisualInfo)
_lib_x11.XGetVisualInfo.argtypes   = [c_void_p, c_long, POINTER(_XVisualInfo), POINTER(c_int)]
_lib_x11.XFree.argtypes            = [c_void_p]


def _visual_depth(x11_dpy: Any, visual_id: int) -> int:
    """Resolve a visual XID to its X protocol depth (24, 32, ...) via ``XGetVisualInfo``."""
    template          = _XVisualInfo()
    template.visualid = visual_id
    n_items           = c_int(0)
    result            = _lib_x11.XGetVisualInfo(x11_dpy, _VISUAL_ID_MASK, byref(template), byref(n_items))
    if not result or n_items.value == 0:
        raise GLContextError(f"XGetVisualInfo found no visual for id 0x{visual_id:x}.")
    depth = result[0].depth
    _lib_x11.XFree(result)

    return depth


@contextlib.contextmanager
def _suppress_x_errors(x11_dpy: Any) -> Generator[None]:
    """Absorb async X protocol errors from GLX calls made within the block.

    Installs a no-op X error handler;
    on exit, XSyncs the display so any pending errors are delivered while that handler is still installed,
    then restores the previous handler.
    """
    previous = _lib_x11.XSetErrorHandler(_silence_x_errors)
    try:
        yield
    finally:
        _lib_x11.XSync(x11_dpy, 0)
        _lib_x11.XSetErrorHandler(previous)


class GLXCaptureContext:
    """GLX context, pixmap texture, PBO, and CUDA resource for per-frame download.

    Owns a hidden GLFW/GLX window used solely as a rendering context,
    a GLXPixmap alias of the XComposite pixmap,
    a pixel buffer object that receives pixels via the GPU copy engine,
    and a CUDA registration of that PBO so frames can be accessed as device pointers.

    All GL and CUDA operations must happen on the thread that created this object.
    """

    def __init__(
        self,
        pixmap_id : int,
        visual_id : int,
        width     : int,
        height    : int,
    ) -> None:
        """Set up the GLX context, texture binding, PBO, and CUDA registration.

        Args:
            pixmap_id: XComposite pixmap XID from :class:`~frame2tensor.capture.xcomposite.XCompositeContext`.
            visual_id: Window visual ID, for FBConfig matching.
            width    : Frame width in pixels.
            height   : Frame height in pixels.

        Raises:
            GLContextError: If GLX or texture setup fails.
            CUDAError     : If CUDA PBO registration fails.
        """
        self._closed       : bool = False
        self._win          : Any  = None
        self._x11_dpy      : Any  = None
        self._glx_pixmap   : int  = 0
        self._tex_id       : int  = 0
        self._pbo          : int  = 0
        self._cuda_resource: Any  = None
        self._frame_bytes  : int  = width * height * 4
        self._fbconfig     : Any  = None

        self._x11_dpy  = self._init_glfw_context()
        self._verify_extension(self._x11_dpy)
        self._fbconfig = self._select_fbconfig(self._x11_dpy, visual_id)
        self._bind_pixmap_texture(self._x11_dpy, self._fbconfig, pixmap_id)
        self._create_pbo()
        self._setup_cuda()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    def update_pixmap(self, pixmap_id: int, width: int, height: int, resize: bool) -> None:
        """Build a fresh GLX pixmap from pixmap_id, then release the old one.

        Called when the source window's backing pixmap was invalidated (resize or remap).
        A pure move does not invalidate the pixmap, so it does not reach here.

        The new pixmap is built and bound before the old one is destroyed, so a transient
        failure (such as a not-yet-committed pixmap mid-resize) leaves the previous pixmap intact.

        Args:
            pixmap_id: New XComposite pixmap XID from :class:`~frame2tensor.capture.xcomposite.XCompositeContext`.
            width    : New frame width (may equal old width if only moved).
            height   : New frame height (may equal old height if only moved).
            resize   : True if width or height changed; triggers PBO and CUDA re-allocation.

        Raises:
            GLContextError: If the new GLXPixmap cannot be created.
            CUDAError     : If CUDA re-registration fails.
        """
        pix_attribs = (c_int * 5)(
            GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT,
            GLX_TEXTURE_FORMAT_EXT, GLX_TEXTURE_FORMAT_RGBA_EXT,
            0,
        )

        old_pixmap = self._glx_pixmap
        with _suppress_x_errors(self._x11_dpy):
            new_pixmap = glXCreatePixmap(self._x11_dpy, self._fbconfig, pixmap_id, pix_attribs)
            if not new_pixmap:
                raise GLContextError("glXCreatePixmap failed during pixmap refresh.")

            if old_pixmap:
                _lib_gl.glXReleaseTexImageEXT(self._x11_dpy, old_pixmap, GLX_FRONT_EXT)
                glXDestroyPixmap(self._x11_dpy, old_pixmap)

            self._glx_pixmap = new_pixmap
            _lib_gl.glXBindTexImageEXT(self._x11_dpy, new_pixmap, GLX_FRONT_EXT, None)

        if resize:
            self._frame_bytes   = width * height * 4
            unregister_resource(resource=self._cuda_resource)
            self._cuda_resource = None
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, self._frame_bytes, None, GL.GL_STREAM_READ)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self._cuda_resource = register_gl_buffer(glo=self._pbo, flags=READ_ONLY)

    def make_current(self) -> None:
        """Make the capture GL context current on the calling thread."""
        glfw.make_context_current(window=self._win)

    def download(self, dst_ptr: int) -> None:
        """Download the current frame to dst_ptr via the GPU copy engine.

        Releases and rebinds the XComposite texture to pick up the latest pixmap contents,
        downloads pixels to the internal PBO via ``glGetTexImage``,
        then copies from the mapped PBO pointer to dst_ptr via ``cudaMemcpy``.

        Args:
            dst_ptr: Destination device pointer (must hold at least width*height*4 bytes).

        Raises:
            GLContextError: If a GL call fails.
            CUDAError     : If CUDA mapping or copy fails.
        """
        # Bind our texture before the rebind so glXBindTexImageEXT targets it explicitly,
        # rather than whatever texture is bound (which matters when another GL context is in use).
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
        with _suppress_x_errors(self._x11_dpy):
            _lib_gl.glXReleaseTexImageEXT(self._x11_dpy, self._glx_pixmap, GLX_FRONT_EXT)
            _lib_gl.glXBindTexImageEXT(self._x11_dpy, self._glx_pixmap, GLX_FRONT_EXT, None)

        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo)
        _ = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, c_void_p(0))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        GL.glFinish()

        map_resources(resource=self._cuda_resource)
        try:
            src_ptr, _ = get_mapped_pointer(resource=self._cuda_resource)
            memcpy(dst=dst_ptr, src=src_ptr, count=self._frame_bytes)
        finally:
            unmap_resources(resource=self._cuda_resource)

    def close(self) -> None:
        """Release CUDA, GL, and GLFW resources."""
        if self._closed:
            return
        self._closed = True

        # The GL and GLX release calls below act on the current GL context, so make this one current first.
        # Without it a second context (e.g. a WindowedRenderer) may be current,
        # and glXReleaseTexImageEXT / glDelete* would fault against a foreign or already-destroyed context.
        self.make_current()

        if self._cuda_resource is not None:
            with contextlib.suppress(Exception):
                unregister_resource(resource=self._cuda_resource)
            self._cuda_resource = None

        if self._pbo:
            GL.glDeleteBuffers(1, [self._pbo])
            self._pbo = 0

        if self._tex_id:
            GL.glDeleteTextures(1, [self._tex_id])
            self._tex_id = 0

        if self._glx_pixmap and self._x11_dpy is not None:
            with contextlib.suppress(Exception):
                _lib_gl.glXReleaseTexImageEXT(self._x11_dpy, self._glx_pixmap, GLX_FRONT_EXT)
            glXDestroyPixmap(self._x11_dpy, self._glx_pixmap)
            self._glx_pixmap = 0

        if self._win is not None:
            glfw.destroy_window(window=self._win)
            self._win = None

    # -----------------------------------------------------------------------------

    def _init_glfw_context(self) -> Any:
        if not glfw.init():
            raise GLContextError("glfw.init() failed.")
        glfw.window_hint(hint=glfw.VISIBLE,              value=glfw.FALSE)
        glfw.window_hint(hint=glfw.CLIENT_API,           value=glfw.OPENGL_API)
        # Force GLX. GLFW may default to EGL on some systems, and EGL contexts don't expose GLX_EXT_texture_from_pixmap.
        glfw.window_hint(hint=glfw.CONTEXT_CREATION_API, value=glfw.NATIVE_CONTEXT_API)
        win = glfw.create_window(width=1, height=1, title="window capture - frame2tensor", monitor=None, share=None)
        if not win:
            raise GLContextError("GLFW hidden window creation failed.")
        glfw.make_context_current(window=win)
        self._win = win

        from OpenGL.raw.GLX._types import struct__XDisplay
        return cast(c_void_p(glfw.get_x11_display()), POINTER(struct__XDisplay))

    def _verify_extension(self, x11_dpy: Any) -> None:
        # Per Claude: PyOpenGL's generated signature claims a None return; the call actually yields bytes | str | None.
        raw: Any = glXQueryExtensionsString(x11_dpy, 0)
        if raw is None:
            raise GLContextError("Could not query GLX extensions; invalid display.")
        exts = raw.decode() if isinstance(raw, bytes) else raw
        if "GLX_EXT_texture_from_pixmap" not in exts:
            raise GLContextError("GLX_EXT_texture_from_pixmap not supported by driver.")

    def _select_fbconfig(self, x11_dpy: Any, visual_id: int) -> Any:
        """Pick a texture_from_pixmap-capable FBConfig whose visual depth matches the window's."""
        fb_attribs = (c_int * 9)(
            GLX_BIND_TO_TEXTURE_RGBA_EXT   , 1,
            _GLX_DRAWABLE_TYPE             , _GLX_PIXMAP_BIT,
            GLX_BIND_TO_TEXTURE_TARGETS_EXT, GLX_TEXTURE_2D_BIT_EXT,
            _GLX_DOUBLEBUFFER              , 0,
            0
        )
        n_configs = c_int(0)
        fbconfigs = glXChooseFBConfig(x11_dpy, 0, fb_attribs, byref(n_configs))
        if not fbconfigs or n_configs.value == 0:
            raise GLContextError("No GLXFBConfig supports texture_from_pixmap.")

        target_depth = _visual_depth(x11_dpy, visual_id)
        for i in range(n_configs.value):
            vis = c_int(0)
            glXGetFBConfigAttrib(x11_dpy, fbconfigs[i], _GLX_VISUAL_ID, byref(vis))
            if _visual_depth(x11_dpy, vis.value) == target_depth:
                return fbconfigs[i]

        raise GLContextError(
            f"No texture_from_pixmap FBConfig matches window visual depth {target_depth}."
        )

    def _bind_pixmap_texture(self, x11_dpy: Any, fbconfig: Any, pixmap_id: int) -> None:
        """Wrap pixmap_id as a GLXPixmap and bind it as the capture texture's image.

        The one-time counterpart to ``update_pixmap``, which rebinds the same way on resize/remap.
        Sets ``self._glx_pixmap`` and ``self._tex_id``.
        """
        pix_attribs = (c_int * 5)(
            GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT,
            GLX_TEXTURE_FORMAT_EXT, GLX_TEXTURE_FORMAT_RGBA_EXT,
            0,
        )
        with _suppress_x_errors(x11_dpy):
            glx_pixmap = glXCreatePixmap(x11_dpy, fbconfig, pixmap_id, pix_attribs)
        if not glx_pixmap:
            raise GLContextError("glXCreatePixmap failed; FBConfig/visual mismatch likely.")
        self._glx_pixmap = glx_pixmap

        tex_id = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        with _suppress_x_errors(x11_dpy):
            _lib_gl.glXBindTexImageEXT(x11_dpy, glx_pixmap, GLX_FRONT_EXT, None)

        gl_err = GL.glGetError()
        if gl_err != GL.GL_NO_ERROR:
            raise GLContextError(f"glXBindTexImageEXT GL error: 0x{gl_err:x}")

        tex_w = int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_WIDTH))
        tex_h = int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_HEIGHT))
        if tex_w == 0 or tex_h == 0:
            raise GLContextError(
                f"glXBindTexImageEXT produced empty texture ({tex_w}x{tex_h}); visual/FBConfig mismatch likely."
            )
        self._tex_id = tex_id

    def _create_pbo(self) -> None:
        pbo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
        GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, self._frame_bytes, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._pbo = pbo

    def _setup_cuda(self) -> None:
        set_device(0)
        self._cuda_resource = register_gl_buffer(glo=self._pbo, flags=READ_ONLY)
