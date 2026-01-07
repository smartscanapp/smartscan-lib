import os
import datetime
import numpy as np
import subprocess
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


def are_valid_files(allowed_exts: list[str], files: list[str]) -> bool:
    return all(path.lower().endswith(allowed_exts) for path in files)
    
