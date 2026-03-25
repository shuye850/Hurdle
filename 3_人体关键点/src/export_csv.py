# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYPOINT_NAMES_33 = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_pose2d_csv(
    out_csv: Path,
    fps: float,
    x2d_all: np.ndarray,
    y2d_all: np.ndarray,
    c2d_all: np.ndarray,
    keypoint_names: list[str] | None = None,
) -> None:
    names = keypoint_names or KEYPOINT_NAMES_33
    t = x2d_all.shape[0]
    data = {
        "frame_index": np.arange(t, dtype=np.int32),
        "timestamp_sec": (np.arange(t, dtype=np.float32) / float(fps)).astype(np.float32),
    }
    for i, name in enumerate(names):
        data[f"{name}_x"] = x2d_all[:, i]
        data[f"{name}_y"] = y2d_all[:, i]
        data[f"{name}_c"] = c2d_all[:, i]
    pd.DataFrame(data).to_csv(out_csv, index=False)


def export_pose3d_csv(
    out_csv: Path,
    fps: float,
    x3d_all: np.ndarray,
    y3d_all: np.ndarray,
    z3d_all: np.ndarray,
    c3d_all: np.ndarray,
    keypoint_names: list[str] | None = None,
) -> None:
    names = keypoint_names or KEYPOINT_NAMES_33
    t = x3d_all.shape[0]
    data = {
        "frame_index": np.arange(t, dtype=np.int32),
        "timestamp_sec": (np.arange(t, dtype=np.float32) / float(fps)).astype(np.float32),
    }
    for i, name in enumerate(names):
        data[f"{name}_x"] = x3d_all[:, i]
        data[f"{name}_y"] = y3d_all[:, i]
        data[f"{name}_z"] = z3d_all[:, i]
        data[f"{name}_c"] = c3d_all[:, i]
    pd.DataFrame(data).to_csv(out_csv, index=False)


def export_meta_json(out_json: Path, meta: dict) -> None:
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
