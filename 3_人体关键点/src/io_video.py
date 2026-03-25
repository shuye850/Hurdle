# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2


@dataclass(frozen=True)
class VideoMeta:
    video_path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    out_width: int
    out_height: int
    max_long_edge: int
    used_scale: float


def _compute_scale(width: int, height: int, max_long_edge: int) -> float:
    if max_long_edge is None or max_long_edge <= 0:
        return 1.0
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return 1.0
    return max_long_edge / float(long_edge)


def open_video(video_path: str | Path, max_long_edge: int = 0) -> Tuple[cv2.VideoCapture, VideoMeta]:
    p = Path(video_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"视频不存在：{p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频（可能是编码或路径问题）：{p}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"读取视频分辨率失败：{p}")

    scale = _compute_scale(width, height, int(max_long_edge or 0))
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))

    meta = VideoMeta(
        video_path=p,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        out_width=out_w,
        out_height=out_h,
        max_long_edge=int(max_long_edge or 0),
        used_scale=float(scale),
    )
    return cap, meta


def iter_frames(
    cap: cv2.VideoCapture,
    used_scale: float = 1.0,
) -> Generator[tuple[int, int, "cv2.Mat"], None, None]:
    idx = 0
    last_ts = -1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    step_ms = max(1, int(round(1000.0 / fps)))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if used_scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(round(frame.shape[1] * used_scale)), int(round(frame.shape[0] * used_scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        ts = int(round(float(cap.get(cv2.CAP_PROP_POS_MSEC) or (idx * 1000.0 / fps))))
        if ts <= last_ts:
            ts = last_ts + step_ms
        last_ts = ts

        yield idx, ts, frame
        idx += 1


def safe_release(cap: Optional[cv2.VideoCapture]) -> None:
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
