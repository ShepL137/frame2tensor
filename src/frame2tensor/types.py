"""GPU memory types and CUDA array interface protocol."""

from typing import Any, Protocol, Self

from frame2tensor._cuda import free, malloc

# typestr: bytes per element
_ITEMSIZES: dict[str, int] = {
    "|u1": 1,
    "<f2": 2,
    "<f4": 4,
}

CUDAArrayInterface = dict[str, Any]
"""Version 3 CUDA array interface dict."""


class SupportsCUDAArray(Protocol):
    """Any object exposing the CUDA array interface.

    Satisfied by PyTorch tensors, CuPy arrays, and ``CUDABuffer``.
    """

    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterface:
        """Version 3 CUDA array interface dict."""
        ...


class CUDABuffer:
    """Pre-allocated GPU memory with ``__cuda_array_interface__``."""

    def __init__(self, shape: tuple[int, ...], typestr: str) -> None:
        """Allocate GPU memory.

        Args:
            shape  : Dimensions of the buffer.
            typestr: Numpy-style type string.

        Raises:
            ValueError: If typestr is not supported.
            CUDAError : If device memory allocation fails.
        """
        self._closed: bool = False
        self._ptr   : int  = 0

        if typestr not in _ITEMSIZES:
            msg: str = f"Unsupported typestr: {typestr!r}"
            raise ValueError(msg)

        self._shape   : tuple[int, ...] = shape
        self._typestr : str             = typestr
        self._itemsize: int             = _ITEMSIZES[typestr]

        total: int = self._itemsize
        for dim in shape:
            total *= dim

        self._ptr = malloc(size_bytes=total)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    @property
    def ptr(self) -> int:
        """Raw device pointer."""
        return self._ptr

    @property
    def shape(self) -> tuple[int, ...]:
        """Buffer dimensions."""
        return self._shape

    @property
    def typestr(self) -> str:
        """Numpy-style type string."""
        return self._typestr

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return self._itemsize

    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterface:
        """Standard CUDA array interface (version 3).

        Enables zero-copy consumption by frameworks like PyTorch/CuPy via ``torch.as_tensor()``/``cupy.asarray()``.
        """
        return {
            "shape"  : self._shape,
            "typestr": self._typestr,
            "data"   : (self._ptr, False),
            "version": 3,
        }

    # -----------------------------------------------------------------------------

    def close(self) -> None:
        """Free device memory. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._ptr != 0:
            free(ptr=self._ptr)
            self._ptr = 0
