# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def valid_point(x: float, y: float, conf: float, thresh: float) -> bool:
    return np.isfinite(x) and np.isfinite(y) and conf >= thresh


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0


def point_from_ratio(proximal: np.ndarray, distal: np.ndarray, ratio: float) -> np.ndarray:
    return proximal + ratio * (distal - proximal)


def estimate_com_for_row(row: pd.Series, cfg: dict) -> Dict[str, float]:
    conf_thresh = float(cfg["com"]["confidence_threshold"])

    def get_pt(name: str):
        x = float(row[f"{name}_x"])
        y = float(row[f"{name}_y"])
        c = float(row.get(f"{name}_score", 1.0))
        return x, y, c

    ls = get_pt("left_shoulder")
    rs = get_pt("right_shoulder")
    lh = get_pt("left_hip")
    rh = get_pt("right_hip")
    lk = get_pt("left_knee")
    rk = get_pt("right_knee")
    la = get_pt("left_ankle")
    ra = get_pt("right_ankle")

    if not (
        valid_point(*ls, conf_thresh) and valid_point(*rs, conf_thresh)
        and valid_point(*lh, conf_thresh) and valid_point(*rh, conf_thresh)
        and valid_point(*lk, conf_thresh) and valid_point(*rk, conf_thresh)
        and valid_point(*la, conf_thresh) and valid_point(*ra, conf_thresh)
    ):
        return {
            "shoulder_mid_x": np.nan,
            "shoulder_mid_y": np.nan,
            "pelvis_mid_x": np.nan,
            "pelvis_mid_y": np.nan,
            "com_x": np.nan,
            "com_y": np.nan,
        }

    p_ls = np.array(ls[:2], dtype=float)
    p_rs = np.array(rs[:2], dtype=float)
    p_lh = np.array(lh[:2], dtype=float)
    p_rh = np.array(rh[:2], dtype=float)
    p_lk = np.array(lk[:2], dtype=float)
    p_rk = np.array(rk[:2], dtype=float)
    p_la = np.array(la[:2], dtype=float)
    p_ra = np.array(ra[:2], dtype=float)

    shoulder_mid = midpoint(p_ls, p_rs)
    hip_mid = midpoint(p_lh, p_rh)

    upper = (
        float(cfg["com"]["upper_anchor_shoulder_weight"]) * shoulder_mid
        + float(cfg["com"]["upper_anchor_hip_weight"]) * hip_mid
    )

    thigh_ratio = float(cfg["com"]["thigh_com_ratio"])
    shank_ratio = float(cfg["com"]["shank_com_ratio"])

    p_lt = point_from_ratio(p_lh, p_lk, thigh_ratio)
    p_rt = point_from_ratio(p_rh, p_rk, thigh_ratio)
    thighs = midpoint(p_lt, p_rt)

    p_ls_seg = point_from_ratio(p_lk, p_la, shank_ratio)
    p_rs_seg = point_from_ratio(p_rk, p_ra, shank_ratio)
    lower = midpoint(p_ls_seg, p_rs_seg)

    com = (
        float(cfg["com"]["upper_weight"]) * upper
        + float(cfg["com"]["thighs_weight"]) * thighs
        + float(cfg["com"]["lower_weight"]) * lower
    )

    return {
        "shoulder_mid_x": float(shoulder_mid[0]),
        "shoulder_mid_y": float(shoulder_mid[1]),
        "pelvis_mid_x": float(hip_mid[0]),
        "pelvis_mid_y": float(hip_mid[1]),
        "com_x": float(com[0]),
        "com_y": float(com[1]),
    }