# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np


def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(v) + eps)


def angle_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    u = p1 - p2
    v = p3 - p2
    denom = safe_norm(u) * safe_norm(v)
    cosine = np.clip(float(np.dot(u, v) / denom), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def angle_line_to_horizontal_min(p_start: np.ndarray, p_end: np.ndarray) -> float:
    v = p_end - p_start
    raw = float(np.degrees(np.arccos(np.clip(v[0] / safe_norm(v), -1.0, 1.0))))
    return min(raw, 180.0 - raw)


def trunk_angle_to_vertical(shoulder_mid: np.ndarray, hip_mid: np.ndarray, move_dir: int) -> float:
    v = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0], dtype=float)
    cosine = np.clip(float(np.dot(v, vertical) / safe_norm(v)), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))

    # 图像坐标里，向右前进时肩点在髋点右侧表示前倾；向左前进时相反。
    dir_sign = 1 if move_dir >= 0 else -1
    signed_x = float(v[0]) * dir_sign
    return angle if signed_x >= 0 else -angle


def infer_move_direction(com_x_series: np.ndarray) -> int:
    n = len(com_x_series)
    if n < 4:
        return 1
    head = float(np.nanmean(com_x_series[: max(2, n // 5)]))
    tail = float(np.nanmean(com_x_series[-max(2, n // 5):]))
    return 1 if tail >= head else -1


def central_difference(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.zeros(n, dtype=float)
    if n == 1:
        return out
    dt = 1.0 / fps
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    if n > 2:
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return out
