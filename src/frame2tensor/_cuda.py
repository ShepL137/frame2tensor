"""Internal wrappers around ``cuda.bindings.runtime``.

Every other module in the library accesses CUDA through the functions defined here.
This gives us named parameters, automatic error handling via :class:`~frame2tensor.exceptions.CUDAError`,
and a single place to adapt if ever ``cuda-python`` changes.
"""

from typing import Any

import cuda.bindings.runtime as cudart
from cuda.bindings.runtime import cudaGraphicsRegisterFlags, cudaMemcpyKind

from frame2tensor.exceptions import CUDAError

# GL constant not exposed by moderngl
GL_TEXTURE_2D: int = 0x0DE1

# Re-export the register flags so that interop.py never touches cuda.bindings
READ_ONLY    : int = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
WRITE_DISCARD: int = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

# Implicit aliases to support Python <3.12.
CUDAResource = Any
CUDAArray    = Any


def _check(err: Any, operation: str) -> None:
    """Raise :class:`CUDAError` if *err* is not ``cudaSuccess`` (0)."""
    if err != 0:
        raise CUDAError(operation, error_code=int(err))


# ------------------------------------------------------------------------------
# Resource registration
# -----------------------------------------------------------------------------

def register_gl_image(glo: int, target: int, flags: int) -> CUDAResource:
    """Register a GL texture with the CUDA runtime.

    Args:
        glo   : Raw GL texture handle (``texture.glo``).
        target: GL texture target (e.g. ``GL_TEXTURE_2D``).
        flags : Registration flags. Use ``READ_ONLY`` or ``WRITE_DISCARD``.

    Returns:
        An opaque ``cudaGraphicsResource`` handle.

    Raises:
        CUDAError: If registration fails.
    """
    err, resource = cudart.cudaGraphicsGLRegisterImage(glo, target, flags)
    _check(err, operation="register_gl_image")

    return resource


def unregister_resource(resource: Any) -> None:
    """Unregister a previously registered GL resource.

    Args:
        resource: Handle returned by :func:`register_gl_image`.

    Raises:
        CUDAError: If unregistration fails.
    """
    err, = cudart.cudaGraphicsUnregisterResource(resource)
    _check(err, operation="unregister_resource")


# -----------------------------------------------------------------------------
# Memory allocation
# -----------------------------------------------------------------------------

def malloc(size_bytes: int) -> int:
    """Allocate device memory.

    Args:
        size_bytes: Number of bytes to allocate.

    Returns:
        Device pointer as an integer.

    Raises:
        CUDAError: If allocation fails.
    """
    err, ptr = cudart.cudaMalloc(size_bytes)
    _check(err, operation="malloc")

    return int(ptr)


def free(ptr: int) -> None:
    """Free device memory.

    Args:
        ptr: Device pointer returned by :func:`malloc`.

    Raises:
        CUDAError: If deallocation fails.
    """
    err, = cudart.cudaFree(ptr)
    _check(err, operation="free")


# -----------------------------------------------------------------------------
# Map / unmap
# -----------------------------------------------------------------------------

def map_resources(resource: Any, stream: int = 0) -> None:
    """Map a registered resource for CUDA access.

    Args:
        resource: Handle returned by :func:`register_gl_image`.
        stream  : CUDA stream to synchronize on (0 = default stream).

    Raises:
        CUDAError: If mapping fails.
    """
    err, = cudart.cudaGraphicsMapResources(1, resource, stream)
    _check(err, operation="map_resources")


def unmap_resources(resource: Any, stream: int = 0) -> None:
    """Release a mapped resource so GL can use it again.

    Args:
        resource: Handle returned by :func:`register_gl_image`.
        stream  : CUDA stream to synchronize on (0 = default stream).

    Raises:
        CUDAError: If unmapping fails.
    """
    err, = cudart.cudaGraphicsUnmapResources(1, resource, stream)
    _check(err, operation="unmap_resources")


# -----------------------------------------------------------------------------
# Mapped array access
# -----------------------------------------------------------------------------

def get_mapped_array(
    resource   : Any,
    array_index: int = 0,
    mip_level  : int = 0,
) -> CUDAArray:
    """Get the CUDA array backing a mapped resource.

    Args:
        resource   : Mapped resource handle.
        array_index: Array index (0 for non-layered textures).
        mip_level  : Mip-map level (0 for base level).

    Returns:
        An opaque ``cudaArray`` handle.

    Raises:
        CUDAError: If the array cannot be retrieved.
    """
    err, cuda_array = cudart.cudaGraphicsSubResourceGetMappedArray(
        resource, array_index, mip_level,
    )
    _check(err, operation="get_mapped_array")

    return cuda_array


# -----------------------------------------------------------------------------
# Device-device copies
# -----------------------------------------------------------------------------

def memcpy_2d_from_array(
    dst_ptr    : int,
    dst_pitch  : int,
    src_array  : Any,
    width_bytes: int,
    height     : int,
) -> None:
    """Copy from a CUDA array into linear device memory.

    Args:
        dst_ptr    : Destination device pointer (e.g. ``tensor.data_ptr()``).
        dst_pitch  : Pitch (bytes per row) of the destination buffer.
        src_array  : Source ``cudaArray`` from :func:`get_mapped_array`.
        width_bytes: Number of bytes to copy per row.
        height     : Number of rows to copy.

    Raises:
        CUDAError: If the copy fails.
    """
    err, = cudart.cudaMemcpy2DFromArray(
        dst_ptr,      # dst
        dst_pitch,    # dpitch
        src_array,    # src
        0,            # wOffset (byte offset into source row)
        0,            # hOffset (starting row in source)
        width_bytes,  # width   (bytes per row)
        height,       # height  (number of rows)
        cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    )
    _check(err, operation="memcpy_2d_from_array")


def memcpy_2d_to_array(
    dst_array  : Any,
    src_ptr    : int,
    src_pitch  : int,
    width_bytes: int,
    height     : int,
) -> None:
    """Copy from linear device memory into a CUDA array.

    Args:
        dst_array  : Destination ``cudaArray`` from :func:`get_mapped_array`.
        src_ptr    : Source device pointer (e.g. ``tensor.data_ptr()``).
        src_pitch  : Pitch (bytes per row) of the source buffer.
        width_bytes: Number of bytes to copy per row.
        height     : Number of rows to copy.

    Raises:
        CUDAError: If the copy fails.
    """
    err, = cudart.cudaMemcpy2DToArray(
        dst_array,    # dst
        0,            # wOffset: byte offset into destination row
        0,            # hOffset: starting row in destination
        src_ptr,      # src
        src_pitch,    # spitch : bytes per source row
        width_bytes,  # width  : bytes per row
        height,       # height : number of rows
        cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    )
    _check(err, operation="memcpy_2d_to_array")


# -----------------------------------------------------------------------------
# GL buffer resource (PBO path)
# -----------------------------------------------------------------------------

def register_gl_buffer(glo: int, flags: int) -> CUDAResource:
    """Register a GL buffer object with the CUDA runtime.

    Used to give CUDA access to a pixel buffer object (PBO) whose storage lives in GPU memory managed by the GL driver.

    Args:
        glo  : Raw GL buffer handle (e.g. a pixel buffer object name).
        flags: Registration flags. Use ``READ_ONLY`` or ``WRITE_DISCARD``.

    Returns:
        An opaque ``cudaGraphicsResource`` handle.

    Raises:
        CUDAError: If registration fails.
    """
    err, resource = cudart.cudaGraphicsGLRegisterBuffer(glo, flags)
    _check(err, operation="register_gl_buffer")

    return resource


def get_mapped_pointer(resource: Any) -> tuple[int, int]:
    """Return the device pointer and byte count for a mapped buffer resource.

    The returned pointer is valid until :func:`unmap_resources` is called on the same resource.

    Args:
        resource: Handle returned by :func:`register_gl_buffer`, currently mapped.

    Returns:
        Pair (device_pointer, size_in_bytes).

    Raises:
        CUDAError: If the pointer cannot be retrieved.
    """
    err, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(resource)
    _check(err, operation="get_mapped_pointer")

    return int(ptr), size


# -----------------------------------------------------------------------------
# Linear device-device copy
# -----------------------------------------------------------------------------

def memcpy(dst: int, src: int, count: int) -> None:
    """Copy count bytes between two device pointers.

    Args:
        dst  : Destination device pointer.
        src  : Source device pointer.
        count: Number of bytes to copy.

    Raises:
        CUDAError: If the copy fails.
    """
    err, = cudart.cudaMemcpy(dst, src, count, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    _check(err, operation="memcpy")


def memcpy_device_to_host(dst: int, src: int, count: int) -> None:
    """Copy count bytes from device memory into host memory.

    Args:
        dst  : Destination host pointer (e.g. a host buffer's address).
        src  : Source device pointer.
        count: Number of bytes to copy.

    Raises:
        CUDAError: If the copy fails.
    """
    err, = cudart.cudaMemcpy(dst, src, count, cudaMemcpyKind.cudaMemcpyDeviceToHost)
    _check(err, operation="memcpy_device_to_host")


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------

def set_device(device: int=0) -> None:
    """Select the active CUDA device for the current thread.

    Args:
        device: CUDA device index.

    Raises:
        CUDAError: If the device cannot be selected.
    """
    err, = cudart.cudaSetDevice(device)
    _check(err, operation="set_device")
