"""Exception hierarchy for frame2tensor.

All public exceptions inherit from ``Frame2TensorError``
so callers can use a single ``except`` clause to catch any library-level failure.
"""


_CUDA_ERROR_MESSAGES: dict[int, str] = {
    1 : "cudaErrorInvalidValue. An invalid argument was passed.",
    2 : "cudaErrorMemoryAllocation. GPU memory allocation failed.",
    3 : "cudaErrorInitializationError. The CUDA driver or runtime could not be initialized.",
    35: (
        "cudaErrorInsufficientDriver. The installed NVIDIA driver is too old for this version of cuda-python."
        " Check your driver version with `nvidia-smi`"
        " and consult the CUDA toolkit compatibility matrix to find a compatible cuda-python range."
    ),
    100: "cudaErrorNoDevice. No CUDA-capable GPU was detected.",
    101: "cudaErrorInvalidDevice. The specified device index is invalid.",
    208: (
        "cudaErrorAlreadyMapped. The resource is already mapped."
        " This may indicate a previous operation failed without unmapping."
    ),
    400: (
        "cudaErrorInvalidResourceHandle."
        " The resource handle passed to the CUDA runtime is invalid or has already been freed."
    ),
    700: (
        "cudaErrorIllegalAddress. A copy or kernel accessed an invalid device address."
        " Commonly, a freed or dangling device pointer."
        " Ensure the source buffer is still alive when the copy runs."
    ),
}


def _cuda_error_message(error_code: int, operation: str) -> str:
    """Build a human-readable error message for a CUDA runtime failure.

    Args:
        error_code: Numeric ``cudaError_t`` value.
        operation : Name of the wrapper or cudart call that failed.

    Returns:
        Formatted message with the operation name, error code, and actionable guidance when available.
    """
    detail: str = _CUDA_ERROR_MESSAGES.get(
        error_code,
        f"cudart error code {error_code}. Consult the CUDA runtime API documentation for details.",
    )
    return f"{operation} failed: {detail}"


class Frame2TensorError(RuntimeError):
    """Base exception for all frame2tensor errors."""


class CUDAError(Frame2TensorError):
    """A CUDA runtime API call failed.

    Attributes:
        operation : Name of the operation or cudart call that failed.
        error_code: Numeric ``cudaError_t`` value returned by the runtime.
    """

    def __init__(self, operation: str, error_code: int) -> None:
        """Build the message from the cudart code, with guidance for the codes we recognize."""
        self.operation : str = operation
        self.error_code: int = error_code
        super().__init__(_cuda_error_message(error_code, operation))


class GLContextError(Frame2TensorError):
    """GL context or framebuffer creation failed.

    Raised when the EGL/GLFW backend cannot create a usable context.
    """


class InvalidTensorError(Frame2TensorError):
    """A tensor argument did not meet the required constraints.

    Raised when a caller passes a tensor with the wrong shape, dtype, device, or memory layout.
    """


class CaptureSourceLostError(Frame2TensorError):
    """The capture source is gone and cannot produce further frames.

    Raised when the captured window has been destroyed (DestroyNotify).
    On minimized or unmapped window, this is raised until restored.
    """


class WindowNotFoundError(Frame2TensorError):
    """No window could be resolved.

    Raised when nothing is focused, or interactive selection was cancelled.
    """


class OutputError(Frame2TensorError):
    """Writing a frame to an output target failed.

    Raised when an output path cannot accept a frame:
    the descriptor is closed, the reader has gone away (broken pipe), or the underlying write failed.
    """
