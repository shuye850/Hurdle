# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (16, 18),
    (15, 19), (16, 20),
    (15, 21), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def draw_pose_on_frame(
    frame_bgr: np.ndarray,
    x2d: np.ndarray,
    y2d: np.ndarray,
    conf: np.ndarray,
    conf_th: float = 0.6,
    frame_index: Optional[int] = None,
    quality: Optional[float] = None,
    infer_mode: Optional[str] = None,
    draw_frame_index: bool = True,
    draw_quality: bool = True,
    draw_infer_mode: bool = True,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for a, b in POSE_CONNECTIONS:
        if conf[a] >= conf_th and conf[b] >= conf_th:
            ax, ay = int(round(x2d[a])), int(round(y2d[a]))
            bx, by = int(round(x2d[b])), int(round(y2d[b]))
            if 0 <= ax < w and 0 <= ay < h and 0 <= bx < w and 0 <= by < h:
                cv2.line(out, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i in range(33):
        if conf[i] < conf_th:
            continue
        px, py = int(round(x2d[i])), int(round(y2d[i]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(out, (px, py), 3, (0, 0, 255), -1)

    text_y = 24
    if draw_frame_index and frame_index is not None:
        cv2.putText(out, f"frame={frame_index}", (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        text_y += 28
    if draw_quality and quality is not None:
        cv2.putText(out, f"quality={quality:.3f}", (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        text_y += 28
    if draw_infer_mode and infer_mode is not None:
        cv2.putText(out, f"mode={infer_mode}", (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return out


def write_overlay_video(
    input_video_path: Path,
    output_video_path: Path,
    fps: float,
    used_scale: float,
    x2d_all: np.ndarray,
    y2d_all: np.ndarray,
    c2d_all: np.ndarray,
    conf_th: float = 0.6,
    quality_all: Optional[np.ndarray] = None,
    infer_modes: Optional[list[str]] = None,
    draw_frame_index: bool = True,
    draw_quality: bool = True,
    draw_infer_mode: bool = True,
) -> None:
    ensure_dir(output_video_path.parent)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频用于生成标定视频：{input_video_path}")

    writer = None
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= x2d_all.shape[0]:
                break

            if used_scale != 1.0:
                frame = cv2.resize(
                    frame,
                    (int(round(frame.shape[1] * used_scale)), int(round(frame.shape[0] * used_scale))),
                    interpolation=cv2.INTER_LINEAR,
                )

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_video_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建输出视频：{output_video_path}")

            q = None
            if quality_all is not None and frame_idx < len(quality_all):
                q = float(quality_all[frame_idx])

            infer_mode = None
            if infer_modes is not None and frame_idx < len(infer_modes):
                infer_mode = infer_modes[frame_idx]

            overlay = draw_pose_on_frame(
                frame_bgr=frame,
                x2d=x2d_all[frame_idx],
                y2d=y2d_all[frame_idx],
                conf=c2d_all[frame_idx],
                conf_th=conf_th,
                frame_index=frame_idx,
                quality=q,
                infer_mode=infer_mode,
                draw_frame_index=draw_frame_index,
                draw_quality=draw_quality,
                draw_infer_mode=draw_infer_mode,
            )
            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
