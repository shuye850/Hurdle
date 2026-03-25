# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

EVENT_LABELS = {
    "takeoff_landing_frame": "起跨腿着地",
    "takeoff_toeoff_frame": "起跨腿离地",
    "bar_cross_frame": "膝关节过栏",
    "landing_frame": "摆动腿着地",
    "recovery_toeoff_frame": "摆动腿离地",
}

STAGE_LABELS = {
    "起跨": "起跨",
    "腾空": "腾空",
    "下栏": "下栏",
    "未知": "未知",
}

FONT_ZH_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

FONT_EN_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]


def _safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if np.isnan(val):
        return None
    return val


def _safe_int(value) -> Optional[int]:
    val = _safe_float(value)
    return None if val is None else int(round(val))


def _pt(row: pd.Series, name: str):
    xi = _safe_int(row.get(f"{name}_x", np.nan))
    yi = _safe_int(row.get(f"{name}_y", np.nan))
    if xi is None or yi is None:
        return None
    return xi, yi


def _load_font(font_size: int, candidates: list[str]):
    try:
        from PIL import ImageFont
    except ModuleNotFoundError:
        return None

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return None


def _draw_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 2,
    prefer_chinese: bool = False,
) -> None:
    try:
        from PIL import Image, ImageDraw

        font = _load_font(
            max(18, int(28 * font_scale)),
            FONT_ZH_CANDIDATES if prefer_chinese else FONT_EN_CANDIDATES,
        )
        if font is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
            frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return
    except Exception:
        pass

    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _build_writer(output_path: Path, fps: float, width: int, height: int) -> Optional[cv2.VideoWriter]:
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _draw_top_panel(frame: np.ndarray, sample_name: str, stage_name: str, frame_idx: int, event_text: Optional[str]) -> None:
    h, w = frame.shape[:2]
    scale = max(w / 1920.0, h / 1080.0)
    panel_h = int(110 * scale)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (248, 250, 252), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, panel_h), (w, panel_h), (185, 193, 204), max(1, int(2 * scale)))

    title_y = int(16 * scale)
    sub_y = int(62 * scale)

    _draw_text(frame, f"Sample: {sample_name}", (int(24 * scale), title_y), (34, 43, 58), 1.02 * scale)
    _draw_text(frame, f"Frame: {frame_idx}", (int(24 * scale), sub_y), (74, 84, 98), 0.84 * scale)

    stage_label = STAGE_LABELS.get(stage_name, stage_name)
    _draw_text(frame, f"Stage: {stage_label}", (int(w * 0.33), title_y), (42, 112, 72), 1.0 * scale, prefer_chinese=True)

    if event_text is not None:
        _draw_text(frame, f"Event: {event_text}", (int(w * 0.33), sub_y), (198, 97, 54), 0.88 * scale, prefer_chinese=True)


def _draw_bottom_panel(frame: np.ndarray, row: pd.Series) -> None:
    h, w = frame.shape[:2]
    scale = max(w / 1920.0, h / 1080.0)
    panel_h = int(120 * scale)
    y0 = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (22, 28, 34), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (0, y0), (w, y0), (85, 150, 220), max(1, int(2 * scale)))

    cols = [int(24 * scale), int(w * 0.27), int(w * 0.53), int(w * 0.74)]
    label_y = y0 + int(26 * scale)
    value_y = y0 + int(72 * scale)

    items = [
        ("COM (px)", f"{_safe_int(row.get('com_x'))}, {_safe_int(row.get('com_y'))}"),
        ("Bar Mid (px)", f"{_safe_int(row.get('bar_mid_x'))}, {_safe_int(row.get('bar_mid_y'))}"),
        ("Post X", f"{_safe_int(row.get('post_x'))}"),
        ("CoM Vx / Vy", f"{_safe_float(row.get('com_vx_mps')):.2f}, {_safe_float(row.get('com_vy_mps')):.2f}"),
    ]

    for x, (label, value) in zip(cols, items):
        _draw_text(frame, label, (x, label_y), (172, 182, 196), 0.60 * scale)
        _draw_text(frame, value, (x, value_y), (248, 247, 242), 0.82 * scale)


def _draw_hurdle_geometry(frame: np.ndarray, row: pd.Series, scale: float, height: int) -> None:
    x1 = _safe_int(row.get("bbox_x1_hurdle"))
    y1 = _safe_int(row.get("bbox_y1_hurdle"))
    x2 = _safe_int(row.get("bbox_x2_hurdle"))
    y2 = _safe_int(row.get("bbox_y2_hurdle"))
    if None not in (x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 196, 224), max(2, int(2 * scale)), lineType=cv2.LINE_AA)

    bx = _safe_int(row.get("bar_mid_x"))
    by = _safe_int(row.get("bar_mid_y"))
    if None not in (bx, by):
        cv2.circle(frame, (bx, by), max(5, int(7 * scale)), (48, 79, 254), -1, lineType=cv2.LINE_AA)
        cv2.line(frame, (max(0, bx - int(38 * scale)), by), (min(frame.shape[1] - 1, bx + int(38 * scale)), by), (48, 79, 254), max(2, int(2 * scale)), lineType=cv2.LINE_AA)

    px = _safe_int(row.get("post_x"))
    py = _safe_int(row.get("post_base_y"))
    if px is not None:
        cv2.line(frame, (px, 0), (px, height - 1), (193, 74, 255), max(1, int(2 * scale)), lineType=cv2.LINE_AA)
        if py is not None:
            cv2.circle(frame, (px, py), max(5, int(7 * scale)), (193, 74, 255), -1, lineType=cv2.LINE_AA)


def _draw_pose(frame: np.ndarray, row: pd.Series, scale: float) -> None:
    left_color = (72, 171, 255)
    right_color = (93, 203, 128)
    joint_r = max(4, int(5 * scale))

    for a, b in SKELETON:
        p1 = _pt(row, a)
        p2 = _pt(row, b)
        if not p1 or not p2:
            continue
        color = left_color if a.startswith("left") or b.startswith("left") else right_color
        cv2.line(frame, p1, p2, color, max(2, int(3 * scale)), lineType=cv2.LINE_AA)

    for name in [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]:
        pt = _pt(row, name)
        if pt is None:
            continue
        color = left_color if name.startswith("left") else right_color
        cv2.circle(frame, pt, joint_r, color, -1, lineType=cv2.LINE_AA)


def _draw_com(frame: np.ndarray, row: pd.Series, scale: float) -> None:
    cx = _safe_int(row.get("com_x"))
    cy = _safe_int(row.get("com_y"))
    if None in (cx, cy):
        return
    radius = max(6, int(8 * scale))
    cv2.circle(frame, (cx, cy), radius + 2, (18, 24, 32), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, (0, 226, 255), -1, lineType=cv2.LINE_AA)
    cv2.line(frame, (cx - int(12 * scale), cy), (cx + int(12 * scale), cy), (0, 226, 255), max(1, int(2 * scale)), lineType=cv2.LINE_AA)
    cv2.line(frame, (cx, cy - int(12 * scale)), (cx, cy + int(12 * scale)), (0, 226, 255), max(1, int(2 * scale)), lineType=cv2.LINE_AA)


def render_fusion_video(
    sample_name: str,
    video_path: Optional[Path],
    frame_df: pd.DataFrame,
    event_df: pd.DataFrame,
    cfg: dict,
    output_path: Path,
) -> None:
    if video_path is None or not Path(video_path).exists():
        print(f"[视频输出跳过] 未找到原视频: {sample_name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[视频输出跳过] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _build_writer(output_path, fps, width, height)
    if writer is None:
        print(f"[视频输出跳过] 无法创建输出视频: {output_path}")
        cap.release()
        return

    frame_map = {int(row["frame"]): row for _, row in frame_df.iterrows()}
    events = event_df.iloc[0].to_dict()
    event_lookup = {int(events[k]): EVENT_LABELS.get(k, k) for k in EVENT_LABELS if k in events and not pd.isna(events[k])}

    cur_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        row = frame_map.get(cur_frame)
        if row is not None:
            scale = max(width / 1920.0, height / 1080.0)
            stage_name = str(row.get("stage_name_std", "未知"))
            event_text = event_lookup.get(cur_frame)

            _draw_hurdle_geometry(frame, row, scale, height)
            _draw_pose(frame, row, scale)
            _draw_com(frame, row, scale)
            _draw_top_panel(frame, sample_name, stage_name, cur_frame, event_text)
            _draw_bottom_panel(frame, row)

        writer.write(frame)
        cur_frame += 1

    writer.release()
    cap.release()
    print(f"[视频输出] 已生成: {output_path}")
