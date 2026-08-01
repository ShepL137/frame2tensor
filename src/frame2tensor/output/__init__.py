"""Output paths: raw frames to a file descriptor, or encoded video via ffmpeg."""

from .fd_writer import FileDescriptorWriter
from .video_writer import VideoWriter

__all__ = ["FileDescriptorWriter", "VideoWriter"]
