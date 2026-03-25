# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from .smoothing import fill_pose_missing_frames, fill_com_missing_frames

from .com_estimator import estimate_com_for_row
from .geometry import (
    angle_line_to_horizontal_min,
    angle_three_points,
    trunk_angle_to_vertical,
    infer_move_direction,
    central_difference,
)


def _classify_hurdle_height(box_h: float, box_w: float, cfg: dict) -> Tuple[float, float]:
    ratio = box_h / max(box_w, 1e-8)
    best_height = np.nan
    confidence = 0.0
    for item in cfg["hurdle_height"]["ratio_thresholds"]:
        if item["min"] <= ratio < item["max"]:
            best_height = float(item["height_m"])
            center = (item["min"] + item["max"]) / 2.0
            width = max((item["max"] - item["min"]) / 2.0, 1e-6)
            confidence = max(0.0, 1.0 - abs(ratio - center) / width)
            break
    return best_height, confidence


def _mean_window(arr: np.ndarray, center: int, radius: int) -> float:
    if len(arr) == 0:
        return float("nan")

    left = max(0, center - radius)
    right = min(len(arr), center + radius + 1)

    if left >= right:
        return float("nan")

    window = arr[left:right]
    if len(window) == 0 or np.all(np.isnan(window)):
        return float("nan")

    return float(np.nanmean(window))

def _p(row: pd.Series, name: str) -> np.ndarray:
    return np.array([float(row[f"{name}_x"]), float(row[f"{name}_y"])], dtype=float)


def _segment_length(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    dx = df[f"{end}_x"] - df[f"{start}_x"]
    dy = df[f"{end}_y"] - df[f"{start}_y"]
    return np.sqrt(dx ** 2 + dy ** 2)


def process_sample(
    sample_name: str,
    source_video_name: str,
    source_video_stem: str,
    hurdle_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    events_json: dict,
    phase_json: dict,
    cfg: dict,
) -> Dict[str, pd.DataFrame]:
    # 统一字段
    # 先对人体关键点做逐帧平滑与插值，再参与后续计算
    pose_df = fill_pose_missing_frames(
        pose_df=pose_df,
        conf_thresh=float(cfg["com"]["confidence_threshold"]),
        interp_limit=8,
    )
    h = cfg["columns"]["hurdle"]
    p = cfg["columns"]["pose"]

    hurdle = hurdle_df.rename(columns={
        h["frame"]: "frame",
        h["time_sec"]: "time_sec_hurdle",
        h["box_x1"]: "bbox_x1_hurdle",
        h["box_y1"]: "bbox_y1_hurdle",
        h["box_x2"]: "bbox_x2_hurdle",
        h["box_y2"]: "bbox_y2_hurdle",
        h["barmid_x"]: "bar_mid_x",
        h["barmid_y"]: "bar_mid_y",
        h["post_x"]: "post_x",
        h["conf"]: "hurdle_conf",
    }).copy()

    pose = pose_df.rename(columns={
        p["frame"]: "frame",
        p["time_sec"]: "time_sec_pose",
        p["fps"]: "fps",
        p["width"]: "width",
        p["height"]: "height",
    }).copy()

    merged = pose.merge(hurdle, on="frame", how="inner").sort_values("frame").reset_index(drop=True)

    com_df = merged.apply(lambda r: pd.Series(estimate_com_for_row(r, cfg)), axis=1)
    merged = pd.concat([merged, com_df], axis=1)
    merged = fill_com_missing_frames(merged)

    merged["left_thigh_length_px"] = _segment_length(merged, "left_hip", "left_knee")
    merged["left_shank_length_px"] = _segment_length(merged, "left_knee", "left_ankle")
    merged["left_leg_length_px"] = merged["left_thigh_length_px"] + merged["left_shank_length_px"]
    merged["right_thigh_length_px"] = _segment_length(merged, "right_hip", "right_knee")
    merged["right_shank_length_px"] = _segment_length(merged, "right_knee", "right_ankle")
    merged["right_leg_length_px"] = merged["right_thigh_length_px"] + merged["right_shank_length_px"]
    merged["leg_length_px"] = merged[["left_leg_length_px", "right_leg_length_px"]].mean(axis=1)

    merged["box_width_px"] = merged["bbox_x2_hurdle"] - merged["bbox_x1_hurdle"]
    merged["box_height_px"] = merged["bbox_y2_hurdle"] - merged["bbox_y1_hurdle"]
    merged["box_bottom_y"] = merged["bbox_y2_hurdle"]

    # 当前先按 box 高度比例估计 far_base_top_y
    beta = float(cfg["mask_detection"]["fallback_box_height_ratio"])
    merged["far_base_top_y"] = merged["box_bottom_y"] - beta * merged["box_height_px"]
    merged["post_base_y"] = (merged["box_bottom_y"] + merged["far_base_top_y"]) / 2.0
    merged["post_height_pixel"] = merged["post_base_y"] - merged["bar_mid_y"]
    merged["hurdle_box_ratio"] = merged["box_height_px"] / merged["box_width_px"].clip(lower=1e-8)

    hh = merged.apply(
        lambda r: _classify_hurdle_height(float(r["box_height_px"]), float(r["box_width_px"]), cfg),
        axis=1,
    )
    merged["post_height_real"] = [x[0] for x in hh]
    merged["hurdle_height_confidence"] = [x[1] for x in hh]

    post_height_real = float(merged["post_height_real"].dropna().median())
    if not np.isfinite(post_height_real):
        post_height_real = 0.914

    post_height_pixel = float(merged["post_height_pixel"].dropna().median())
    scale = post_height_real / max(post_height_pixel, 1e-8)

    merged["post_height_real_global"] = post_height_real
    merged["pixel_to_meter_scale"] = scale
    merged["left_thigh_length_m"] = merged["left_thigh_length_px"] * scale
    merged["left_shank_length_m"] = merged["left_shank_length_px"] * scale
    merged["left_leg_length_m"] = merged["left_leg_length_px"] * scale
    merged["right_thigh_length_m"] = merged["right_thigh_length_px"] * scale
    merged["right_shank_length_m"] = merged["right_shank_length_px"] * scale
    merged["right_leg_length_m"] = merged["right_leg_length_px"] * scale
    merged["leg_length_m"] = merged["leg_length_px"] * scale

    # 行进方向：优先读 events_json
    move_dir = int(events_json.get("direction_sign", 0))
    if move_dir == 0:
        move_dir = infer_move_direction(merged["com_x"].to_numpy(dtype=float))
    merged["move_dir"] = move_dir

    leading_leg = events_json["swing_leg"]
    trailing_leg = events_json["takeoff_leg"]

    # 事件帧
    e = events_json["events"]
    t_takeoff_landing = int(e[cfg["events"]["takeoff_landing_frame"]])
    t_takeoff_toeoff = int(e[cfg["events"]["takeoff_toeoff_frame"]])
    t_landing = int(e[cfg["events"]["landing_frame"]])
    t_recovery_toeoff = int(e[cfg["events"]["recovery_toeoff_frame"]])

    # 模块五的过栏帧与模块四保持一致：使用摆动腿膝关节过栏帧
    t_bar_cross_knee = int(events_json["cross_bar_row"])
    t_bar_cross = t_bar_cross_knee

    # 阶段名：把 phase_json 的“落地”统一改成“下栏”
    phase_map = {}
    for seg in phase_json["segments"]:
        start_f = int(seg["start_frame"])
        end_f = int(seg["end_frame"])
        name = seg["phase_name"]
        if name == "落地":
            name = cfg["stage_names"]["landing"]
        for fidx in range(start_f, end_f + 1):
            phase_map[fidx] = name
    merged["stage_name_std"] = merged["frame"].map(phase_map).fillna("未知")

    fps = float(merged["fps"].dropna().iloc[0])

    frame_to_idx = {int(f): i for i, f in enumerate(merged["frame"].tolist())}
    idx_takeoff_landing = frame_to_idx[t_takeoff_landing]
    idx_takeoff_toeoff = frame_to_idx[t_takeoff_toeoff]
    idx_bar_cross = frame_to_idx[t_bar_cross]
    idx_landing = frame_to_idx[t_landing]
    idx_recovery_toeoff = frame_to_idx[t_recovery_toeoff]

    row_takeoff_landing = merged.iloc[idx_takeoff_landing]
    row_takeoff_toeoff = merged.iloc[idx_takeoff_toeoff]
    row_bar_cross = merged.iloc[idx_bar_cross]
    row_landing = merged.iloc[idx_landing]
    row_recovery_toeoff = merged.iloc[idx_recovery_toeoff]

    # 速度
    vx = central_difference(merged["com_x"].to_numpy(dtype=float), fps) * move_dir * scale
    vy = -central_difference(merged["com_y"].to_numpy(dtype=float), fps) * scale
    merged["com_vx_mps"] = vx
    merged["com_vy_mps"] = vy

    def leg_names(side: str):
        if side.lower().startswith("l"):
            return "left_hip", "left_knee", "left_ankle"
        return "right_hip", "right_knee", "right_ankle"

    trail_h, trail_k, trail_a = leg_names(trailing_leg)
    lead_h, lead_k, lead_a = leg_names(leading_leg)

    trail_leg_length_px = float(row_bar_cross[f"{trailing_leg}_leg_length_px"]) if f"{trailing_leg}_leg_length_px" in row_bar_cross else np.nan
    lead_leg_length_px = float(row_bar_cross[f"{leading_leg}_leg_length_px"]) if f"{leading_leg}_leg_length_px" in row_bar_cross else np.nan

    post_x = float(merged["post_x"].dropna().median())
    post_base_y = float(merged["post_base_y"].dropna().median())
    barmid_y = float(merged["bar_mid_y"].dropna().median())

    # 起跨
    trail_ankle_takeoff = _p(row_takeoff_toeoff, trail_a)
    trail_hip_takeoff = _p(row_takeoff_toeoff, trail_h)
    trail_knee_takeoff = _p(row_takeoff_toeoff, trail_k)

    takeoff_distance_px = move_dir * (post_x - float(trail_ankle_takeoff[0]))
    takeoff_distance_m = takeoff_distance_px * scale
    takeoff_angle_deg = angle_line_to_horizontal_min(trail_ankle_takeoff, trail_hip_takeoff)

    trail_ankle_takeoff_land = _p(row_takeoff_landing, trail_a)
    trail_knee_takeoff_land = _p(row_takeoff_landing, trail_k)
    trail_hip_takeoff_land = _p(row_takeoff_landing, trail_h)

    takeoff_landing_angle_deg = angle_line_to_horizontal_min(trail_ankle_takeoff_land, trail_knee_takeoff_land)
    takeoff_landing_knee_angle_deg = angle_three_points(trail_hip_takeoff_land, trail_knee_takeoff_land, trail_ankle_takeoff_land)

    takeoff_span = merged.iloc[idx_takeoff_landing: idx_takeoff_toeoff + 1].copy()
    takeoff_knee_angles = []
    for _, row in takeoff_span.iterrows():
        takeoff_knee_angles.append(
            angle_three_points(_p(row, trail_h), _p(row, trail_k), _p(row, trail_a))
        )
    takeoff_span["trail_knee_angle_deg"] = takeoff_knee_angles
    takeoff_min_idx = int(np.argmin(takeoff_knee_angles))
    takeoff_max_buffer_frame = int(takeoff_span.iloc[takeoff_min_idx]["frame"])
    takeoff_buffer_knee_angle_deg = float(np.min(takeoff_knee_angles))
    takeoff_toeoff_knee_angle_deg = angle_three_points(trail_hip_takeoff, trail_knee_takeoff, trail_ankle_takeoff)
    takeoff_buffer_amplitude_deg = takeoff_landing_knee_angle_deg - takeoff_buffer_knee_angle_deg
    takeoff_push_amplitude_deg = takeoff_toeoff_knee_angle_deg - takeoff_buffer_knee_angle_deg
    takeoff_support_time_s = (t_takeoff_toeoff - t_takeoff_landing) / fps

    takeoff_toeoff_trunk_angle_deg = trunk_angle_to_vertical(
        np.array([row_takeoff_toeoff["shoulder_mid_x"], row_takeoff_toeoff["shoulder_mid_y"]]),
        np.array([row_takeoff_toeoff["pelvis_mid_x"], row_takeoff_toeoff["pelvis_mid_y"]]),
        move_dir,
    )

    pre_w = int(cfg["velocity"]["pre_takeoff_window"])
    toeoff_w = int(cfg["velocity"]["toeoff_window"])
    landing_w = int(cfg["velocity"]["landing_window"])

    com_vx_pre_takeoff_mps = _mean_window(vx, idx_takeoff_landing, pre_w)
    com_vx_takeoff_toeoff_mps = _mean_window(vx, idx_takeoff_toeoff, toeoff_w)
    com_vx_change_mps = com_vx_takeoff_toeoff_mps - com_vx_pre_takeoff_mps

    com_vy_pre_takeoff_mps = _mean_window(vy, idx_takeoff_landing, pre_w)
    com_vy_takeoff_toeoff_mps = _mean_window(vy, idx_takeoff_toeoff, toeoff_w)
    com_vy_change_mps = com_vy_takeoff_toeoff_mps - com_vy_pre_takeoff_mps

    takeoff_rise_angle_deg = float(np.degrees(np.arctan2(
        com_vy_takeoff_toeoff_mps,
        max(abs(com_vx_takeoff_toeoff_mps), 1e-8)
    )))
    takeoff_phase_time_s = takeoff_support_time_s

    # 腾空
    flight_time_s = (t_landing - t_takeoff_toeoff) / fps
    flight_span = merged.iloc[idx_takeoff_toeoff: idx_landing + 1].copy()
    peak_local = int(np.argmin(flight_span["com_y"].to_numpy()))
    com_peak_frame = int(flight_span.iloc[peak_local]["frame"])
    com_peak_y = float(flight_span.iloc[peak_local]["com_y"])
    com_rise_px = float(row_takeoff_toeoff["com_y"] - com_peak_y)
    com_rise_m = com_rise_px * scale
    flight_disp_px = move_dir * (float(row_landing["com_x"]) - float(row_takeoff_toeoff["com_x"]))
    flight_disp_m = flight_disp_px * scale
    flight_com_speed_mps = flight_disp_m / max(flight_time_s, 1e-8)

    com_peak_to_post_px = move_dir * (float(flight_span.iloc[peak_local]["com_x"]) - post_x)
    com_peak_to_post_m = com_peak_to_post_px * scale

    toeoff_to_bar_time_s = (t_bar_cross - t_takeoff_toeoff) / fps
    bar_to_landing_time_s = (t_landing - t_bar_cross) / fps
    toeoff_to_bar_ratio = toeoff_to_bar_time_s / max(flight_time_s, 1e-8)
    bar_to_landing_ratio = bar_to_landing_time_s / max(flight_time_s, 1e-8)

    bar_cross_com_height_px = post_base_y - float(row_bar_cross["com_y"])
    bar_cross_com_height_m = bar_cross_com_height_px * scale
    bar_cross_com_clearance_m = bar_cross_com_height_m - post_height_real

    bar_cross_lead_knee_angle_deg = angle_three_points(_p(row_bar_cross, lead_h), _p(row_bar_cross, lead_k), _p(row_bar_cross, lead_a))
    bar_cross_trunk_angle_deg = trunk_angle_to_vertical(
        np.array([row_bar_cross["shoulder_mid_x"], row_bar_cross["shoulder_mid_y"]]),
        np.array([row_bar_cross["pelvis_mid_x"], row_bar_cross["pelvis_mid_y"]]),
        move_dir,
    )

    # 下栏
    lead_ankle_landing = _p(row_landing, lead_a)
    lead_knee_landing = _p(row_landing, lead_k)
    lead_hip_landing = _p(row_landing, lead_h)

    landing_distance_px = move_dir * (float(lead_ankle_landing[0]) - post_x)
    landing_distance_m = landing_distance_px * scale
    landing_com_dist_px = move_dir * (float(lead_ankle_landing[0]) - float(row_landing["com_x"]))
    landing_com_dist_m = landing_com_dist_px * scale
    landing_shank_angle_deg = angle_line_to_horizontal_min(lead_ankle_landing, lead_knee_landing)
    landing_knee_angle_deg = angle_three_points(lead_hip_landing, lead_knee_landing, lead_ankle_landing)

    landing_span = merged.iloc[idx_landing: idx_recovery_toeoff + 1].copy()
    landing_knee_angles = []
    for _, row in landing_span.iterrows():
        landing_knee_angles.append(angle_three_points(_p(row, lead_h), _p(row, lead_k), _p(row, lead_a)))
    landing_span["lead_knee_angle_deg"] = landing_knee_angles
    buffer_idx = int(np.argmin(landing_knee_angles))
    max_buffer_frame = int(landing_span.iloc[buffer_idx]["frame"])
    buffer_knee_angle_deg = float(np.min(landing_knee_angles))

    lead_ankle_toeoff = _p(row_recovery_toeoff, lead_a)
    lead_knee_toeoff = _p(row_recovery_toeoff, lead_k)
    lead_hip_toeoff = _p(row_recovery_toeoff, lead_h)

    push_shank_angle_deg = angle_line_to_horizontal_min(lead_ankle_toeoff, lead_knee_toeoff)
    toeoff_knee_angle_deg = angle_three_points(lead_hip_toeoff, lead_knee_toeoff, lead_ankle_toeoff)
    buffer_amplitude_deg = landing_knee_angle_deg - buffer_knee_angle_deg
    push_amplitude_deg = toeoff_knee_angle_deg - buffer_knee_angle_deg

    landing_trunk_angle_deg = trunk_angle_to_vertical(
        np.array([row_landing["shoulder_mid_x"], row_landing["shoulder_mid_y"]]),
        np.array([row_landing["pelvis_mid_x"], row_landing["pelvis_mid_y"]]),
        move_dir,
    )
    toeoff_trunk_angle_deg = trunk_angle_to_vertical(
        np.array([row_recovery_toeoff["shoulder_mid_x"], row_recovery_toeoff["shoulder_mid_y"]]),
        np.array([row_recovery_toeoff["pelvis_mid_x"], row_recovery_toeoff["pelvis_mid_y"]]),
        move_dir,
    )
    trunk_angle_change_deg = landing_trunk_angle_deg - toeoff_trunk_angle_deg

    recovery_time_s = (t_recovery_toeoff - t_landing) / fps
    landing_phase_time_s = recovery_time_s
    total_phase_time_s = takeoff_phase_time_s + flight_time_s + landing_phase_time_s
    takeoff_phase_ratio = takeoff_phase_time_s / max(total_phase_time_s, 1e-8)
    flight_phase_ratio = flight_time_s / max(total_phase_time_s, 1e-8)
    landing_phase_ratio = landing_phase_time_s / max(total_phase_time_s, 1e-8)
    com_vx_landing_mps = _mean_window(vx, idx_landing, landing_w)
    com_vy_landing_mps = _mean_window(vy, idx_landing, landing_w)

    metric_row = {
        "video_id": sample_name,
        "sample_name": sample_name,
        "source_video_name": source_video_name,
        "source_video_stem": source_video_stem,
        "fps": fps,
        "total_frames": int(len(merged)),
        "move_dir": move_dir,
        "leading_leg": leading_leg,
        "trail_leg": trailing_leg,

        "takeoff_landing_frame": t_takeoff_landing,
        "takeoff_toeoff_frame": t_takeoff_toeoff,
        "bar_cross_frame": t_bar_cross,
        "knee_bar_cross_frame": t_bar_cross_knee,
        "landing_frame": t_landing,
        "recovery_toeoff_frame": t_recovery_toeoff,

        "post_x": float(merged["post_x"].dropna().median()),
        "barmid_x": float(merged["bar_mid_x"].dropna().median()),
        "barmid_y": barmid_y,
        "box_bottom_y": float(merged["box_bottom_y"].dropna().median()),
        "far_base_top_y": float(merged["far_base_top_y"].dropna().median()),
        "post_base_y": post_base_y,
        "post_height_pixel": float(merged["post_height_pixel"].dropna().median()),
        "hurdle_box_width_px": float(merged["box_width_px"].dropna().median()),
        "hurdle_box_height_px": float(merged["box_height_px"].dropna().median()),
        "hurdle_box_ratio": float(merged["hurdle_box_ratio"].dropna().median()),
        "post_height_real": post_height_real,
        "hurdle_height_method": cfg["hurdle_height"]["method"],
        "hurdle_height_confidence": float(merged["hurdle_height_confidence"].dropna().median()),
        "pixel_to_meter_scale": scale,
        "leg_length_px": float(merged["leg_length_px"].median()),
        "leg_length_m": float(merged["leg_length_m"].median()),
        "left_leg_length_px": float(merged["left_leg_length_px"].median()),
        "left_leg_length_m": float(merged["left_leg_length_m"].median()),
        "right_leg_length_px": float(merged["right_leg_length_px"].median()),
        "right_leg_length_m": float(merged["right_leg_length_m"].median()),
        "leading_leg_length_px": float(merged[f"{leading_leg}_leg_length_px"].median()),
        "leading_leg_length_m": float(merged[f"{leading_leg}_leg_length_m"].median()),
        "trail_leg_length_px": float(merged[f"{trailing_leg}_leg_length_px"].median()),
        "trail_leg_length_m": float(merged[f"{trailing_leg}_leg_length_m"].median()),

        "takeoff_distance_px": takeoff_distance_px,
        "takeoff_distance_m": takeoff_distance_m,
        "takeoff_angle_deg": takeoff_angle_deg,
        "takeoff_landing_angle_deg": takeoff_landing_angle_deg,
        "takeoff_landing_knee_angle_deg": takeoff_landing_knee_angle_deg,
        "takeoff_max_buffer_frame": takeoff_max_buffer_frame,
        "takeoff_buffer_knee_angle_deg": takeoff_buffer_knee_angle_deg,
        "takeoff_toeoff_knee_angle_deg": takeoff_toeoff_knee_angle_deg,
        "takeoff_buffer_amplitude_deg": takeoff_buffer_amplitude_deg,
        "takeoff_push_amplitude_deg": takeoff_push_amplitude_deg,
        "takeoff_support_time_s": takeoff_support_time_s,
        "takeoff_phase_time_s": takeoff_phase_time_s,
        "takeoff_toeoff_trunk_angle_deg": takeoff_toeoff_trunk_angle_deg,

        "com_vx_pre_takeoff_mps": com_vx_pre_takeoff_mps,
        "com_vx_takeoff_toeoff_mps": com_vx_takeoff_toeoff_mps,
        "com_vx_change_mps": com_vx_change_mps,
        "com_vy_pre_takeoff_mps": com_vy_pre_takeoff_mps,
        "com_vy_takeoff_toeoff_mps": com_vy_takeoff_toeoff_mps,
        "com_vy_change_mps": com_vy_change_mps,
        "takeoff_rise_angle_deg": takeoff_rise_angle_deg,

        "flight_time_s": flight_time_s,
        "com_peak_frame": com_peak_frame,
        "com_peak_y": com_peak_y,
        "com_rise_px": com_rise_px,
        "com_rise_m": com_rise_m,
        "flight_disp_px": flight_disp_px,
        "flight_disp_m": flight_disp_m,
        "flight_com_speed_mps": flight_com_speed_mps,
        "com_peak_to_post_px": com_peak_to_post_px,
        "com_peak_to_post_m": com_peak_to_post_m,
        "toeoff_to_bar_time_s": toeoff_to_bar_time_s,
        "bar_to_landing_time_s": bar_to_landing_time_s,
        "toeoff_to_bar_ratio": toeoff_to_bar_ratio,
        "bar_to_landing_ratio": bar_to_landing_ratio,
        "bar_cross_com_height_px": bar_cross_com_height_px,
        "bar_cross_com_height_m": bar_cross_com_height_m,
        "bar_cross_com_clearance_m": bar_cross_com_clearance_m,
        "bar_cross_lead_knee_angle_deg": bar_cross_lead_knee_angle_deg,
        "bar_cross_trunk_angle_deg": bar_cross_trunk_angle_deg,
        "bar_cross_leading_leg_length_px": lead_leg_length_px,
        "bar_cross_leading_leg_length_m": lead_leg_length_px * scale,
        "bar_cross_trail_leg_length_px": trail_leg_length_px,
        "bar_cross_trail_leg_length_m": trail_leg_length_px * scale,

        "landing_distance_px": landing_distance_px,
        "landing_distance_m": landing_distance_m,
        "landing_com_dist_px": landing_com_dist_px,
        "landing_com_dist_m": landing_com_dist_m,
        "landing_shank_angle_deg": landing_shank_angle_deg,
        "push_shank_angle_deg": push_shank_angle_deg,
        "landing_knee_angle_deg": landing_knee_angle_deg,
        "max_buffer_frame": max_buffer_frame,
        "buffer_knee_angle_deg": buffer_knee_angle_deg,
        "toeoff_knee_angle_deg": toeoff_knee_angle_deg,
        "buffer_amplitude_deg": buffer_amplitude_deg,
        "push_amplitude_deg": push_amplitude_deg,
        "landing_trunk_angle_deg": landing_trunk_angle_deg,
        "toeoff_trunk_angle_deg": toeoff_trunk_angle_deg,
        "trunk_angle_change_deg": trunk_angle_change_deg,
        "recovery_time_s": recovery_time_s,
        "landing_phase_time_s": landing_phase_time_s,
        "total_phase_time_s": total_phase_time_s,
        "takeoff_phase_ratio": takeoff_phase_ratio,
        "flight_phase_ratio": flight_phase_ratio,
        "landing_phase_ratio": landing_phase_ratio,
        "com_vx_landing_mps": com_vx_landing_mps,
        "com_vy_landing_mps": com_vy_landing_mps,
    }

    event_df = pd.DataFrame([{
        "video_id": sample_name,
        "sample_name": sample_name,
        "source_video_name": source_video_name,
        "source_video_stem": source_video_stem,
        "takeoff_landing_frame": t_takeoff_landing,
        "takeoff_toeoff_frame": t_takeoff_toeoff,
        "bar_cross_frame": t_bar_cross,
        "knee_bar_cross_frame": t_bar_cross_knee,
        "com_peak_frame": com_peak_frame,
        "landing_frame": t_landing,
        "recovery_toeoff_frame": t_recovery_toeoff,
        "takeoff_max_buffer_frame": takeoff_max_buffer_frame,
        "max_buffer_frame": max_buffer_frame,
    }])

    metric_df = pd.DataFrame([metric_row])
    merged["video_id"] = sample_name
    merged["sample_name"] = sample_name
    merged["source_video_name"] = source_video_name
    merged["source_video_stem"] = source_video_stem

    return {
        "frame_features": merged,
        "event_frames": event_df,
        "technical_metrics": metric_df,
    }
