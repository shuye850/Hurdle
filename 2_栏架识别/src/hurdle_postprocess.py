# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HurdleFrameResult:
    frame_idx: int
    time_sec: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bar_mid_x: float
    bar_mid_y: float
    post_x: float
conf: float


def select_center_instance(boxes_xyxy: np.ndarray, frame_width: int) -> Optional[int]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None
    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    target_x = frame_width / 2.0
    distances = np.abs(centers_x - target_x)
    return int(np.argmin(distances))


def polygon_to_mask(polygon_xy: np.ndarray, image_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon_xy is None or len(polygon_xy) < 3:
        return mask
    pts = np.round(polygon_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_bar_and_post(mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.nan, np.nan, np.nan

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    h = max(1, y_max - y_min + 1)
    w = max(1, x_max - x_min + 1)
    roi = mask[y_min:y_max + 1, x_min:x_max + 1]

    top_h = max(3, int(round(h * 0.35)))
    top_roi = roi[:top_h, :]
    row_counts = (top_roi > 0).sum(axis=1)
    candidate_rows = np.where(row_counts > max(3, int(round(w * 0.15))))[0]
    bar_row_local = int(candidate_rows[0]) if len(candidate_rows) else int(np.argmax(row_counts))
    bar_row_pixels = np.where(top_roi[bar_row_local] > 0)[0]
    bar_mid_x = float(x_min + (bar_row_pixels.min() + bar_row_pixels.max()) / 2.0) if len(bar_row_pixels) else np.nan
    bar_mid_y = float(y_min + bar_row_local)

    left_w = max(3, int(round(w * 0.35)))
    left_roi = roi[:, :left_w]
    col_counts = (left_roi > 0).sum(axis=0)
    candidate_cols = np.where(col_counts > max(3, int(round(h * 0.2))))[0]
    post_col_local = int(candidate_cols[0]) if len(candidate_cols) else int(np.argmax(col_counts))
    post_x = float(x_min + post_col_local)
    return bar_mid_x, bar_mid_y, post_x
