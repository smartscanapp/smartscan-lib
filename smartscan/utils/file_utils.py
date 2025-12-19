import os
import datetime
import numpy as np
import subprocess
import re
from pathlib import Path
from smartscan.errors import SmartScanError, ErrorCode
from numpy.typing import NDArray

def read_text_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()       


def get_days_since_last_modified(file_path: str) -> int:
    last_modified_timestamp = os.path.getmtime(file_path)    
    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp)    
    current_date = datetime.datetime.now()    
    days_since_modified = (current_date - last_modified_date).days
    return days_since_modified 


def get_files_from_dirs(dirs: list[str], dir_skip_patterns: list[str] = [], allowed_exts: tuple[str] | None  = None, limit: int | None = None) -> list[str]:
    if not isinstance(dirs, list):
        raise SmartScanError("Invalid list of directories", code=ErrorCode.INVALID_ARGUMENT)
    
    paths = []

    def walk(base: Path):
        nonlocal paths
        try:
            for entry in base.iterdir():
                if entry.is_dir() and any(entry.match(pat) for pat in dir_skip_patterns):
                    continue
                if entry.is_file():
                    if allowed_exts is not None and not entry.name.endswith(allowed_exts):
                        continue
                    if limit is not None and len(paths) >= limit:
                        return
                    paths.append(str(entry.resolve()))
                elif entry.is_dir():
                    walk(entry)
        except PermissionError:
            print(f"[Skipped] Permission denied: {base}")

    for d in dirs:
        root_dir = Path(d)
        if root_dir.is_dir():
            walk(root_dir)
    
    return paths


def get_child_dirs(dirs: list[str], dir_skip_patterns: list[str] = []) -> list[str]:
    if not isinstance(dirs, list):
        raise SmartScanError("Invalid list of directories", code=ErrorCode.INVALID_ARGUMENT)
    
    paths = []

    def walk(base: Path):
        nonlocal paths
        try:
            for entry in base.iterdir():
                if entry.is_dir():
                    if any(entry.match(pat) for pat in dir_skip_patterns):
                        continue
                    paths.append(str(entry.resolve()))
                    walk(entry)
        except PermissionError:
            print(f"[Skipped] Permission denied: {base}")

    for d in dirs:
        root_dir = Path(d)
        if root_dir.is_dir():
            walk(root_dir)
    
    return paths


def get_frames_from_video(video_path: str, n_frames: int) -> NDArray[np.uint8]:
    """
    Extract `n` evenly spaced frames from a video using one FFmpeg process.
    Returns a list of frames as NumPy arrays (H, W, 3, dtype=uint8) at original resolution.
    """
    cmd_probe = ["ffmpeg", "-i", video_path]
    proc = subprocess.Popen(cmd_probe, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    _, err = proc.communicate()
    err = err.decode()

    # Original resolution
    match = re.search(r", (\d+)x(\d+)", err)
    if not match:
        raise ValueError("Could not determine video dimensions")
    width, height = int(match.group(1)), int(match.group(2))

    # Duration
    match_dur = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
    if not match_dur:
        raise ValueError("Could not determine video duration")
    hours, minutes, seconds = map(float, match_dur.groups())
    duration = hours*3600 + minutes*60 + seconds

    cmd = [
        "ffmpeg",
        "-i", video_path,
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


def are_valid_files(allowed_exts: list[str], files: list[str]) -> bool:
    return all(path.lower().endswith(allowed_exts) for path in files)
    
