from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


VIDEO_EXTS = ['.mp4', '.mov', '.m4v', '.avi', '.MP4', '.MOV', '.M4V', '.AVI']


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def find_video_by_stem(video_dir: str | Path, stem: str) -> Path | None:
    base = Path(video_dir)
    for ext in VIDEO_EXTS:
        p = base / f'{stem}{ext}'
        if p.exists():
            return p
    return None
