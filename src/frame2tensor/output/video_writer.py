"""Video encoding via ffmpeg."""
import shutil
import subprocess
from pathlib import Path
from typing import IO, Self, cast

from frame2tensor.types import SupportsCUDAArray

from .fd_writer import FileDescriptorWriter


class VideoWriter:
    """Encode frames to a video file via an ffmpeg subprocess.

    Wraps :class:`~frame2tensor.output.FileDescriptorWriter` and an ffmpeg process.
    Frames from any :class:`~frame2tensor.types.SupportsCUDAArray` source are copied
    to host and piped as raw RGBA to ffmpeg for encoding.
    The subprocess and its stdin are owned and closed on ``close()``.

    Usage::

        with VideoWriter("out.mp4", width=1280, height=720, fps=60) as rec:
            for frame in source:
                rec.write(frame)
    """

    def __init__(
        self,
        path  : str | Path,
        width : int,
        height: int,
        fps   : int  = 60,
    ) -> None:
        """Open an ffmpeg subprocess and prepare to encode frames.

        Args:
            path  : Output video path. Format is inferred from the file extension.
            width : Width in pixels of the source frame.
            height: Height in pixels of the source frame.
            fps   : Output frame rate.

        Raises:
            FileNotFoundError: If ffmpeg is not found on PATH.
        """
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg not found on PATH")

        self._proc: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603
            [  # noqa: S607
                "ffmpeg", "-y", "-loglevel", "warning",
                "-f", "rawvideo",
                "-pixel_format", "rgba",
                "-video_size", f"{width}x{height}",
                "-framerate", str(fps),
                "-i", "-",
                "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self._stdin : IO[bytes]            = cast("IO[bytes]", self._proc.stdin)
        self._writer: FileDescriptorWriter = FileDescriptorWriter(self._stdin.fileno(), width=width, height=height)
        self._closed: bool                 = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------

    def write(self, source: SupportsCUDAArray) -> None:
        """Encode one frame.

        Args:
            source: Device buffer exposing ``__cuda_array_interface__``.

        Raises:
            InvalidTensorError: If the source shape or dtype does not match the declared geometry.
            OutputError       : If the write to ffmpeg's stdin fails.
        """
        self._writer.write(source)

    def close(self) -> None:
        """Close ffmpeg's stdin and wait for the subprocess to finish encoding."""
        if self._closed:
            return

        self._closed = True
        self._stdin.close()
        _ = self._proc.wait()
