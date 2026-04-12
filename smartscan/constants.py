import os
from pathlib import Path

class SupportedFileTypes:
    IMAGE = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    TEXT = ('.txt', '.md', '.rst', '.json')
    VIDEO = ('.mp4', '.mkv', '.webm')

LOCAL_BASE_DIR = Path.home() / ".cache" / "smartscan"
BASE_DIR = Path(os.environ.get("SMARTSCAN_BASE_DIR", LOCAL_BASE_DIR))

