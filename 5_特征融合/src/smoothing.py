# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd


KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def fill_pose_missing_frames(
    pose_df: pd.DataFrame,
    conf_thresh: float = 0.5,
    interp_limit: int = 8,
) -> pd.DataFrame:
    """
    只补帧，不平滑：
    1. 低置信度置空
    2. 线性插值补缺失
    3. 前后填充首尾空值
    """
    df = pose_df.copy()

    for name in KEYPOINT_NAMES:
        x_col = f"{name}_x"
        y_col = f"{name}_y"
        s_col = f"{name}_score"

        if x_col not in df.columns or y_col not in df.columns:
            continue

        if s_col in df.columns:
            low_conf_mask = df[s_col].astype(float) < conf_thresh
            df.loc[low_conf_mask, x_col] = np.nan
            df.loc[low_conf_mask, y_col] = np.nan

        df[x_col] = df[x_col].astype(float).interpolate(
            method="linear",
            limit=interp_limit,
            limit_direction="both",
        )
        df[y_col] = df[y_col].astype(float).interpolate(
            method="linear",
            limit=interp_limit,
            limit_direction="both",
        )

        df[x_col] = df[x_col].ffill().bfill()
        df[y_col] = df[y_col].ffill().bfill()

    return df


def fill_com_missing_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    CoM 相关列只做插值补帧，不做平滑
    """
    out = df.copy()

    for col in ["com_x", "com_y", "shoulder_mid_x", "shoulder_mid_y", "pelvis_mid_x", "pelvis_mid_y"]:
        if col not in out.columns:
            continue
        out[col] = out[col].astype(float).interpolate(
            method="linear",
            limit_direction="both",
        )
        out[col] = out[col].ffill().bfill()

    return out