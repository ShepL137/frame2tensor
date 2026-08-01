"""GL-CUDA texture interop.

Provides two classes for device-to-device data exchange between OpenGL textures and CUDA memory:

- ``GLCUDATexture``: read path; copy a GL texture into a :class:`~frame2tensor.types.CUDABuffer`.
- ``CUDAWritableTexture``: write path; copy a CUDA buffer into a GL texture.
"""

from typing import Any, Self

import moderngl as mgl

from frame2tensor._cuda import (
    GL_TEXTURE_2D,
    READ_ONLY,
    WRITE_DISCARD,
    CUDAArray,
    CUDAResource,
    get_mapped_array,
    map_resources,
    memcpy_2d_from_array,
    memcpy_2d_to_array,
    register_gl_image,
    unmap_resources,
    unregister_resource,
)
from frame2tensor.exceptions import Frame2TensorError, InvalidTensorError
from frame2tensor.types import _ITEMSIZES, CUDABuffer, SupportsCUDAArray  # pyright: ignore[reportPrivateUsage]


class GLCUDATexture:
    """Read path: register a GL texture with CUDA and copy its contents into a buffer.

    The texture must already exist and be bound to a framebuffer into which you render before calling ``to_buffer()``.
    """

    def __init__(
        self,
        texture   : mgl.Texture,
        width     : int,
        height    : int,
        components: int = 4,
        typestr   : str = "|u1"
    ) -> None:
        """Register a GL texture with CUDA for reading.

        Args:
            texture   : Existing moderngl texture to register.
            width     : Texture width in texels.
            height    : Texture height in texels.
            components: Number of color components.
            typestr   : Numpy-style type string.

        Raises:
            ValueError: If typestr is not supported.
            CUDAError : If GL texture registration fails.
        """
        self._closed   : bool                = False
        self._resource : CUDAResource | None = None
        self.width     : int                 = width
        self.height    : int                 = height
        self.components: int                 = components

        if typestr not in _ITEMSIZES:
            raise ValueError(f"Unsupported typestr: {typestr!r}")

        self._row_bytes: int                 = width * components * _ITEMSIZES[typestr]  # pitch = width in bytes
        self._resource                       = register_gl_image(glo=texture.glo, target=GL_TEXTURE_2D, flags=READ_ONLY)
        self._buffer   : CUDABuffer | None   = CUDABuffer(shape=(height, width, components), typestr=typestr)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    def to_buffer(self) -> CUDABuffer:
        """Map the GL texture into CUDA, copy its contents to the internal buffer, and unmap.

        Returns:
            A reference to the internal :class:`~frame2tensor.types.CUDABuffer`.
            Using borrowed semantics, the caller does not own this memory.
            The buffer's contents are overwritten on the next call to ``to_buffer()``.
            Clone if the data must outlive the next call (e.g. ``torch.as_tensor(buf, device="cuda").clone()``).

        Raises:
            Frame2TensorError: If the texture has been closed.
            CUDAError        : If the map, copy, or unmap fails.
        """
        if self._buffer is None:
            raise Frame2TensorError("GLCUDATexture is closed")
        map_resources(resource=self._resource)
        try:
            cuda_array: CUDAArray = get_mapped_array(resource=self._resource)
            memcpy_2d_from_array(
                dst_ptr     = self._buffer.ptr,
                dst_pitch   = self._row_bytes,
                src_array   = cuda_array,
                width_bytes = self._row_bytes,
                height      = self.height,
            )
        finally:
            unmap_resources(resource=self._resource)

        return self._buffer

    def close(self) -> None:
        """Unregister the GL texture from CUDA and free the buffer."""
        if self._closed:
            return

        self._closed = True

        if self._resource is not None:
            unregister_resource(resource=self._resource)
            self._resource = None

        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None


class CUDAWritableTexture:
    """Write path: copy CUDA device memory into a GL texture.

    Accepts any object exposing ``__cuda_array_interface__``
    (PyTorch tensors, CuPy arrays, :class:`~frame2tensor.types.CUDABuffer`).
    The source must be C-contiguous; strided or non-contiguous layouts are rejected by ``write()``.

    ``write()`` validates the source and copies on every call.
    """

    def __init__(self, texture: mgl.Texture, width: int, height: int, components: int = 1) -> None:
        """Register a GL texture with CUDA for writing.

        Args:
            texture   : Existing moderngl texture to register.
            width     : Texture width in texels.
            height    : Texture height in texels.
            components: Number of color components.

        Raises:
            CUDAError: If GL texture registration fails.
        """
        self._closed   : bool                = False
        self._resource : CUDAResource | None = None
        self.width     : int                 = width
        self.height    : int                 = height
        self.components: int                 = components
        self._resource                       = register_gl_image(
            glo=texture.glo, target=GL_TEXTURE_2D, flags=WRITE_DISCARD
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------


    def write(self, tensor: SupportsCUDAArray) -> None:
        """Copy device memory into the GL texture.

        Validates the source's shape, dtype, and contiguity.
        Then uses ``cudart.cudaMemcpy2DToArray`` to the mapped CUDA array.
        Validation guards against a source whose dimensions changed since construction.

        Args:
            tensor: An object exposing ``__cuda_array_interface__``.

        Raises:
            InvalidTensorError: If the source layout does not match this texture.
            CUDAError         : If the map, copy, or unmap fails.
        """
        row_bytes = self._validate(tensor)
        ptr       = tensor.__cuda_array_interface__["data"][0]

        map_resources(resource=self._resource)
        try:
            cuda_array: CUDAArray = get_mapped_array(resource=self._resource)
            memcpy_2d_to_array(
                dst_array   = cuda_array,
                src_ptr     = ptr,
                src_pitch   = row_bytes,
                width_bytes = row_bytes,
                height      = self.height,
            )
        finally:
            unmap_resources(resource=self._resource)

    def close(self) -> None:
        """Unregister the GL texture from CUDA."""
        if self._closed:
            return

        self._closed = True
        if self._resource is not None:
            unregister_resource(resource=self._resource)
            self._resource = None

    # -----------------------------------------------------------------------------

    def _validate(self, tensor: SupportsCUDAArray) -> int:
        """Validate a source's CUDA array interface and return its row pitch in bytes.

        Args:
            tensor: Any object exposing ``__cuda_array_interface__``.

        Returns:
            Row pitch (bytes per row) for the validated source.

        Raises:
            InvalidTensorError: If the object lacks the interface, has the wrong shape,
                                uses an unsupported dtype, or is not C-contiguous.
        """
        interface: Any | None = getattr(tensor, "__cuda_array_interface__", None)
        if interface is None:
            raise InvalidTensorError("object does not expose __cuda_array_interface__")

        shape = interface["shape"]
        expected: tuple[int, int] | tuple[int, int, int]
        expected = (self.height, self.width) if self.components == 1 else (self.height, self.width, self.components)

        if shape != expected:
            raise InvalidTensorError(f"expected shape {expected}, got {shape}")

        typestr = interface["typestr"]
        itemsize: int | None = _ITEMSIZES.get(typestr)
        if itemsize is None:
            raise InvalidTensorError(f"unsupported dtype: {typestr!r}\nsupported dtypes are: {sorted(_ITEMSIZES)}")

        strides = interface.get("strides")
        if strides is not None:
            # C-contiguous strides for this shape and itemsize
            expected_strides: list[Any] = []
            stride          : int       = itemsize
            for dim in reversed(shape):
                expected_strides.append(stride)
                stride *= dim
            expected_strides.reverse()

            if tuple(strides) != tuple(expected_strides):
                raise InvalidTensorError("tensor must be contiguous")

        return self.width * self.components * itemsize
