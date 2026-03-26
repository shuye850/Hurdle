from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


FRAME_LABELS = [
    ("takeoff_landing_frame", "起跨着地"),
    ("takeoff_toeoff_frame", "起跨离地"),
    ("com_peak_frame", "重心最高点"),
    ("bar_cross_frame", "过栏瞬间"),
    ("landing_frame", "下栏着地"),
    ("recovery_toeoff_frame", "恢复腿离地"),
]


def _normalized_ids(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    values = [text]
    if text.isdigit():
        values.append(text.lstrip("0") or "0")
        values.append(text.zfill(3))
    return list(dict.fromkeys(values))


def _candidate_names(sample_name: str, video_id: str) -> list[str]:
    names: list[str] = []
    for value in [sample_name, video_id]:
        names.extend(_normalized_ids(value))
    return list(dict.fromkeys(names))


def _resolve_csv_path(base_dir: Path, sample_name: str, video_id: str, suffix: str) -> Path | None:
    for name in _candidate_names(sample_name, video_id):
        candidate = base_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_event_row(event_dir: Path, sample_name: str, video_id: str) -> dict[str, Any] | None:
    csv_path = _resolve_csv_path(event_dir, sample_name, video_id, "_event_frames.csv")
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_frame_features(frame_dir: Path, sample_name: str, video_id: str) -> pd.DataFrame | None:
    csv_path = _resolve_csv_path(frame_dir, sample_name, video_id, "_frame_features.csv")
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df


def resolve_video_path(
    sample_name: str,
    source_video_name: str,
    fusion_video_dir: Path,
    preprocess_video_dir: Path,
) -> Path | None:
    candidates = [
        fusion_video_dir / f"{sample_name}_fusion_result.mp4",
        preprocess_video_dir / f"{sample_name}.mov",
        preprocess_video_dir / source_video_name,
        preprocess_video_dir / "暂存" / f"{sample_name}.mov",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def extract_key_event_screenshots(
    video_path: Path | None,
    event_row: dict[str, Any] | None,
    output_dir: Path,
    sample_name: str,
) -> list[dict[str, Any]]:
    if video_path is None or event_row is None:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    screenshots: list[dict[str, Any]] = []
    try:
        for frame_key, label in FRAME_LABELS:
            frame_value = event_row.get(frame_key)
            if frame_value is None:
                continue
            try:
                frame_idx = int(float(frame_value))
            except Exception:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = output_dir / f"{sample_name}_{frame_key}.png"
            cv2.imwrite(str(out_path), frame)
            screenshots.append(
                {
                    "frame_key": frame_key,
                    "label": label,
                    "frame_index": frame_idx,
                    "path": str(out_path),
                }
            )
    finally:
        cap.release()

    return screenshots


def build_event_frame_records(event_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not event_row:
        return []

    records: list[dict[str, Any]] = []
    for frame_key, label in FRAME_LABELS:
        frame_value = event_row.get(frame_key)
        if frame_value is None or pd.isna(frame_value):
            continue
        try:
            frame_idx = int(float(frame_value))
        except Exception:
            continue
        records.append({"frame_key": frame_key, "label": label, "frame_index": frame_idx})
    return records


def _blank_background(frame_df: pd.DataFrame) -> np.ndarray:
    width = int(float(frame_df.iloc[0].get("width", 1280)))
    height = int(float(frame_df.iloc[0].get("height", 720)))
    return np.full((height, width, 3), 245, dtype=np.uint8)


def _load_background(video_path: Path | None, frame_df: pd.DataFrame) -> np.ndarray:
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
        finally:
            cap.release()
    return _blank_background(frame_df)


def render_com_trajectory_image(
    video_path: Path | None,
    frame_df: pd.DataFrame | None,
    event_row: dict[str, Any] | None,
    output_path: Path,
    sample_name: str,
) -> Path | None:
    if frame_df is None or frame_df.empty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_df = frame_df.copy()
    frame_df["frame"] = pd.to_numeric(frame_df["frame"], errors="coerce")
    frame_df["com_x"] = pd.to_numeric(frame_df["com_x"], errors="coerce")
    frame_df["com_y"] = pd.to_numeric(frame_df["com_y"], errors="coerce")
    valid = frame_df.dropna(subset=["frame", "com_x", "com_y"])
    if valid.empty:
        return None

    canvas = _load_background(video_path, valid)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (255, 255, 255), -1)
    canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)

    points = valid[["com_x", "com_y"]].to_numpy(dtype=float)
    polyline = np.round(points).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [polyline], False, (0, 214, 255), 5, lineType=cv2.LINE_AA)

    for idx in range(0, len(points), max(1, len(points) // 18)):
        px, py = map(int, np.round(points[idx]))
        cv2.circle(canvas, (px, py), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    start_pt = tuple(map(int, np.round(points[0])))
    end_pt = tuple(map(int, np.round(points[-1])))
    cv2.circle(canvas, start_pt, 10, (64, 196, 99), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, end_pt, 10, (64, 82, 255), -1, lineType=cv2.LINE_AA)

    frame_lookup = valid.set_index(valid["frame"].astype(int))
    for index, (frame_key, _) in enumerate(FRAME_LABELS):
        if not event_row or frame_key not in event_row or pd.isna(event_row.get(frame_key)):
            continue
        try:
            frame_idx = int(float(event_row[frame_key]))
        except Exception:
            continue
        if frame_idx not in frame_lookup.index:
            continue
        row = frame_lookup.loc[frame_idx]
        ex, ey = int(round(float(row["com_x"]))), int(round(float(row["com_y"])))
        cv2.circle(canvas, (ex, ey), 9, (255, 64, 154), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (ex, ey), 14, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index + 1),
            (ex + 8, ey - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (36, 36, 36),
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"{sample_name} CoM Trajectory",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (28, 37, 54),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "1-6 mark key event points on the trajectory",
        (28, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (83, 96, 114),
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), canvas)
    return output_path
