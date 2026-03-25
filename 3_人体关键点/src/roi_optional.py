# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def compute_roi_from_keypoints(
    x2d: np.ndarray,
    y2d: np.ndarray,
    conf: np.ndarray,
    frame_width: int,
    frame_height: int,
    conf_threshold: float = 0.55,
    padding_ratio: float = 0.15,
    min_valid_points: int = 8,
    min_box_size: int = 96,
    target_aspect_ratio: float | None = None,
) -> Optional[ROIBox]:
    valid = np.isfinite(x2d) & np.isfinite(y2d) & (conf >= conf_threshold)
    if int(np.sum(valid)) < int(min_valid_points):
        return None

    xs = x2d[valid]
    ys = y2d[valid]
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    pad = padding_ratio * max(box_w, box_h)

    x1 = float(x_min - pad)
    y1 = float(y_min - pad)
    x2 = float(x_max + pad)
    y2 = float(y_max + pad)

    if target_aspect_ratio is not None and target_aspect_ratio > 0:
        cur_w = x2 - x1
        cur_h = y2 - y1
        cur_ar = cur_w / max(cur_h, 1.0)
        if cur_ar < target_aspect_ratio:
            target_w = target_aspect_ratio * cur_h
            extra = 0.5 * (target_w - cur_w)
            x1 -= extra
            x2 += extra
        else:
            target_h = cur_w / target_aspect_ratio
            extra = 0.5 * (target_h - cur_h)
            y1 -= extra
            y2 += extra

    if (x2 - x1) < min_box_size:
        extra = 0.5 * (min_box_size - (x2 - x1))
        x1 -= extra
        x2 += extra
    if (y2 - y1) < min_box_size:
        extra = 0.5 * (min_box_size - (y2 - y1))
        y1 -= extra
        y2 += extra

    x1 = int(round(max(0, x1)))
    y1 = int(round(max(0, y1)))
    x2 = int(round(min(frame_width, x2)))
    y2 = int(round(min(frame_height, y2)))

    if x2 <= x1 or y2 <= y1:
        return None
    return ROIBox(x1=x1, y1=y1, x2=x2, y2=y2)


def smooth_roi(prev_roi: ROIBox | None, new_roi: ROIBox | None, momentum: float = 0.65) -> ROIBox | None:
    if new_roi is None:
        return prev_roi
    if prev_roi is None:
        return new_roi

    m = float(momentum)
    x1 = int(round(m * prev_roi.x1 + (1.0 - m) * new_roi.x1))
    y1 = int(round(m * prev_roi.y1 + (1.0 - m) * new_roi.y1))
    x2 = int(round(m * prev_roi.x2 + (1.0 - m) * new_roi.x2))
    y2 = int(round(m * prev_roi.y2 + (1.0 - m) * new_roi.y2))
    if x2 <= x1 or y2 <= y1:
        return new_roi
    return ROIBox(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_frame_by_roi(frame_bgr: np.ndarray, roi: ROIBox) -> np.ndarray:
    return frame_bgr[roi.y1:roi.y2, roi.x1:roi.x2].copy()


def map_pose2d_back_to_full_frame(
    x2d_roi: np.ndarray,
    y2d_roi: np.ndarray,
    roi: ROIBox,
) -> tuple[np.ndarray, np.ndarray]:
    x_full = x2d_roi.astype(np.float32).copy()
    y_full = y2d_roi.astype(np.float32).copy()
    x_full += np.float32(roi.x1)
    y_full += np.float32(roi.y1)
    return x_full, y_full
