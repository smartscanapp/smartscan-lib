
import numpy as np
import subprocess
from numpy.typing import NDArray
from PIL import Image
from smartscan.constants import SupportedFileTypes
from smartscan.errors import SmartScanError, ErrorCode
from smartscan.types import VideoSource



def get_video_metadata(source: str) -> tuple[int, int, float]:
    """
    Return (width, height, duration_seconds) for a video file or URL.
    Uses ffprobe for reliability.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        source
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    lines = out.decode().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Could not determine metadata for {source}")
    width, height, duration = int(lines[0]), int(lines[1]), float(lines[2])
    return width, height, duration


def get_frames_from_video(source: str, n_frames: int, short_video_duration: float = 60.0) -> NDArray[np.uint8]:
    """
    Extract `n` evenly spaced frames from a video file or URL.
    Returns frames as NumPy arrays (H, W, 3, dtype=uint8).
    """
    width, height, duration = get_video_metadata(source)
    if duration < short_video_duration:
        return _get_frames_from_short_video(source, n_frames, width, height, duration)
    else:
        return _get_frames_from_long_video(source, n_frames, width, height, duration)
    

def _get_frames_from_long_video(source: str, n_frames: int, width: int, height: int, duration: float) -> NDArray[np.uint8]:
    """
    Extract `n_frames` evenly spaced frames from a video file or URL efficiently.
    Returns frames as NumPy arrays (H, W, 3, dtype=uint8).
    """
    width, height, duration = get_video_metadata(source)
    frame_size = width * height * 3
    frames = []

    timestamps = [duration * i / n_frames for i in range(n_frames)]

    for t in timestamps:
        # Seek to timestamp t
        cmd = [
            "ffmpeg",
            "-ss", str(t),           
            "-i", source,
            "-frames:v", "1",        
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw = proc.stdout.read(frame_size)
        proc.stdout.close()
        proc.wait()
        if len(raw) == frame_size:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frames.append(frame)

    return np.stack(frames, axis=0)


def _get_frames_from_short_video(source: str, n_frames: int, width: int, height: int, duration: float) -> NDArray[np.uint8]:
    """
    Extract `n` evenly spaced frames from a video file or URL.
    Returns frames as NumPy arrays (H, W, 3, dtype=uint8).
    """

    cmd = [
        "ffmpeg",
        "-i", source,
        "-vf", f"fps={n_frames/duration}",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = width * height * 3
    frames = []

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame)

    proc.stdout.close()
    proc.wait()
    return np.stack(frames, axis=0)


def video_source_to_pil_images(source: VideoSource, n_frames: int = 10, short_video_duration: float = 60.0):
    if isinstance(source, str):
        if source.startswith(("http", "https")):
           frame_arrs = get_frames_from_video(source, n_frames, short_video_duration)
           return [Image.fromarray(frame) for frame in frame_arrs]
        elif source.endswith(SupportedFileTypes.VIDEO):
            frame_arrs = get_frames_from_video(source, n_frames, short_video_duration)
            return [Image.fromarray(frame) for frame in frame_arrs]       
        else:
            raise SmartScanError("Unsupported file type", code=ErrorCode.UNSUPPORTED_FILE_TYPE, details=f"Supported file types: {SupportedFileTypes.IMAGE + SupportedFileTypes.TEXT + SupportedFileTypes.VIDEO}")
    else:
        return [Image.fromarray(frame_arr) for frame_arr in source]