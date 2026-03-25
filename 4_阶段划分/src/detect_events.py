from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd


@dataclass
class DetectConfig:
    smooth_window: int = 5
    knee_angle_smooth_window: int = 5
    knee_peak_window: int = 3
    knee_peak_prominence_deg: float = 2.0

    # 前两个关键点：只在“过栏前最后一次双踝 y 相交”到“摆动腿膝 x 过栏”之间搜索
    ankle_low_ratio: float = 0.20
    low_segment_min_len: int = 3

    # 后两个关键点：保持原有逻辑
    post_bar_margin_px: float = 10.0
    swing_low_acc_quantile: float = 0.18
    swing_min_segment_len: int = 3

    min_phase_gap_frames: int = 2
    fallback_pre_frames: int = 18
    fallback_post_frames: int = 12


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _angle_deg(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def compute_knee_angle(df: pd.DataFrame, side: str) -> pd.Series:
    vals: List[float] = []
    for row in df.itertuples(index=False):
        vals.append(
            _angle_deg(
                getattr(row, f"{side}_hip_x"),
                getattr(row, f"{side}_hip_y"),
                getattr(row, f"{side}_knee_x"),
                getattr(row, f"{side}_knee_y"),
                getattr(row, f"{side}_ankle_x"),
                getattr(row, f"{side}_ankle_y"),
            )
        )
    return pd.Series(vals, index=df.index, dtype=float)


def compute_features(df: pd.DataFrame, cfg: DetectConfig) -> pd.DataFrame:
    out = df.copy()

    out["pelvis_x"] = (out["left_hip_x"] + out["right_hip_x"]) / 2.0
    out["pelvis_y"] = (out["left_hip_y"] + out["right_hip_y"]) / 2.0

    for side in ["left", "right"]:
        out[f"{side}_ankle_x_smooth"] = smooth_series(out[f"{side}_ankle_x"], cfg.smooth_window)
        out[f"{side}_ankle_y_smooth"] = smooth_series(out[f"{side}_ankle_y"], cfg.smooth_window)
        out[f"{side}_ankle_vy"] = out[f"{side}_ankle_y_smooth"].diff().fillna(0.0)
        out[f"{side}_ankle_ay"] = out[f"{side}_ankle_vy"].diff().fillna(0.0)

        out[f"{side}_knee_x_smooth"] = smooth_series(out[f"{side}_knee_x"], cfg.smooth_window)
        out[f"{side}_knee_y_smooth"] = smooth_series(out[f"{side}_knee_y"], cfg.smooth_window)

        out[f"{side}_knee_angle"] = smooth_series(
            compute_knee_angle(out, side), cfg.knee_angle_smooth_window
        )

    out["pelvis_dx_to_bar"] = out["pelvis_x"] - out["bar_mid_x"]
    out["left_ankle_dx_to_bar"] = out["left_ankle_x_smooth"] - out["bar_mid_x"]
    out["right_ankle_dx_to_bar"] = out["right_ankle_x_smooth"] - out["bar_mid_x"]
    out["ankle_y_diff"] = out["left_ankle_y_smooth"] - out["right_ankle_y_smooth"]

    return out


def detect_direction(df: pd.DataFrame) -> int:
    return 1 if float(df["pelvis_x"].iloc[-1] - df["pelvis_x"].iloc[0]) >= 0 else -1


def find_reference_row(df: pd.DataFrame) -> int:
    return int(df["pelvis_dx_to_bar"].abs().idxmin())


def determine_leg_roles(df: pd.DataFrame, ref_row: int, direction_sign: int) -> Tuple[str, str]:
    row = df.iloc[ref_row]
    lx = float(row["left_ankle_x_smooth"])
    rx = float(row["right_ankle_x_smooth"])
    if direction_sign == 1:
        swing_leg = "left" if lx > rx else "right"
    else:
        swing_leg = "left" if lx < rx else "right"
    takeoff_leg = "right" if swing_leg == "left" else "left"
    return swing_leg, takeoff_leg


def _frame_to_row(df: pd.DataFrame, frame_index: int) -> int:
    matches = df.index[df["frame_index"] == frame_index].tolist()
    if not matches:
        raise ValueError(f"找不到 frame_index={frame_index}")
    return int(matches[0])


def _is_after_bar(dx: float, direction_sign: int, margin_px: float = 0.0) -> bool:
    return dx * direction_sign >= margin_px


def _is_behind(x_a: float, x_b: float, direction_sign: int) -> bool:
    return (x_a - x_b) * direction_sign < 0


def _consecutive_segments(mask: pd.Series, min_len: int) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= min_len:
                segs.append((start, i - 1))
            start = None

    if start is not None and len(mask) - start >= min_len:
        segs.append((start, len(mask) - 1))

    return segs


def _local_maxima(series: pd.Series, start: int, end: int) -> List[int]:
    idxs: List[int] = []
    start = max(1, start)
    end = min(len(series) - 2, end)

    for i in range(start, end + 1):
        prev_v = float(series.iloc[i] - series.iloc[i - 1])
        next_v = float(series.iloc[i + 1] - series.iloc[i])
        if prev_v > 0 and next_v < 0:
            idxs.append(i)

    return idxs


def _find_crossings(diff: pd.Series, start_row: int, end_row: int) -> List[int]:
    rows: List[int] = []
    start_row = max(1, start_row)
    end_row = min(len(diff) - 1, end_row)
    for i in range(start_row, end_row + 1):
        prev_val = float(diff.iloc[i - 1])
        cur_val = float(diff.iloc[i])
        if prev_val == 0.0 or prev_val * cur_val <= 0:
            rows.append(i)
    return rows


def _leading_leg_cross_bar_row(
    df: pd.DataFrame,
    swing_leg: str,
    direction_sign: int,
    joint: str = "knee",
) -> Optional[int]:
    dx = df[f"{swing_leg}_{joint}_x_smooth"] - df["bar_mid_x"]
    for i in range(len(df)):
        if dx.iloc[i] * direction_sign > 0:
            return i
    return None


def _pre_cross_last_ankle_y_cross_row(
    df: pd.DataFrame,
    leading_leg_cross_row: int,
) -> Optional[int]:
    crossings = _find_crossings(df["ankle_y_diff"], 1, max(1, leading_leg_cross_row - 1))
    if not crossings:
        return None
    return crossings[-1]


def _low_y_segments_in_window(
    series: pd.Series,
    start_row: int,
    end_row: int,
    ratio: float,
    min_len: int,
) -> List[Tuple[int, int]]:
    start_row = max(0, start_row)
    end_row = min(len(series) - 1, end_row)
    if start_row > end_row:
        return []

    sub = series.iloc[start_row:end_row + 1].reset_index(drop=True)
    if len(sub) == 0:
        return []

    y_min = float(sub.min())
    y_max = float(sub.max())
    threshold = y_min + ratio * (y_max - y_min)
    mask = sub <= threshold
    segs = _consecutive_segments(mask, min_len)

    return [(start_row + a, start_row + b) for a, b in segs]


def _touchdown_position_constraint(
    df: pd.DataFrame,
    i: int,
    swing_leg: str,
    takeoff_leg: str,
    direction_sign: int,
) -> bool:
    swing_knee_x = float(df.iloc[i][f"{swing_leg}_knee_x_smooth"])
    swing_ankle_x = float(df.iloc[i][f"{swing_leg}_ankle_x_smooth"])
    takeoff_knee_x = float(df.iloc[i][f"{takeoff_leg}_knee_x_smooth"])

    return _is_behind(swing_knee_x, takeoff_knee_x, direction_sign) and _is_behind(
        swing_ankle_x, takeoff_knee_x, direction_sign
    )


def _toeoff_position_constraint(
    df: pd.DataFrame,
    i: int,
    swing_leg: str,
    takeoff_leg: str,
    direction_sign: int,
) -> bool:
    takeoff_knee_x = float(df.iloc[i][f"{takeoff_leg}_knee_x_smooth"])
    swing_knee_x = float(df.iloc[i][f"{swing_leg}_knee_x_smooth"])
    swing_ankle_x = float(df.iloc[i][f"{swing_leg}_ankle_x_smooth"])

    return _is_behind(takeoff_knee_x, swing_knee_x, direction_sign) and _is_behind(
        takeoff_knee_x, swing_ankle_x, direction_sign
    )


def _toeoff_bar_height_constraint(
    df: pd.DataFrame,
    i: int,
    takeoff_leg: str,
) -> bool:
    takeoff_knee_y = float(df.iloc[i][f"{takeoff_leg}_knee_y_smooth"])
    takeoff_ankle_y = float(df.iloc[i][f"{takeoff_leg}_ankle_y_smooth"])
    bar_top_y = float(df.iloc[i]["bar_mid_y"])

    return takeoff_knee_y > bar_top_y and takeoff_ankle_y > bar_top_y


def _peak_candidates(knee_series: pd.Series, segments: List[Tuple[int, int]]) -> List[int]:
    cands: List[int] = []
    for seg_start, seg_end in segments:
        cands.extend(_local_maxima(knee_series, seg_start, seg_end))
    return sorted(set(cands))


def _window_fallback_peak(knee: pd.Series, start_row: int, end_row: int) -> int:
    cands = _local_maxima(knee, start_row, end_row)
    if cands:
        return max(cands, key=lambda i: float(knee.iloc[i]))
    return int(knee.iloc[start_row:end_row + 1].idxmax())


def detect_takeoff_touchdown_and_toeoff(
    df: pd.DataFrame,
    swing_leg: str,
    takeoff_leg: str,
    direction_sign: int,
    cfg: DetectConfig,
    window_start_row: int,
    window_end_row: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    在 [过栏前最后一次双踝 y 相交, 摆动腿膝 x 过栏前一帧] 这个窗口内：
    - 用起跨腿踝 y 低值区间筛候选片段
    - 在片段里找膝角峰值
    - 较早的峰值作为 touchdown
    - 较晚的峰值作为 toeoff
    """
    ankle_y = df[f"{takeoff_leg}_ankle_y_smooth"]
    knee = df[f"{takeoff_leg}_knee_angle"]

    low_segments = _low_y_segments_in_window(
        ankle_y, window_start_row, window_end_row, cfg.ankle_low_ratio, cfg.low_segment_min_len
    )
    if not low_segments:
        low_segments = [(window_start_row, window_end_row)]

    peaks = _peak_candidates(knee, low_segments)
    if not peaks:
        peaks = _local_maxima(knee, window_start_row, window_end_row)

    touchdown_candidates: List[int] = []
    toeoff_candidates: List[int] = []

    for i in peaks:
        if _touchdown_position_constraint(df, i, swing_leg, takeoff_leg, direction_sign):
            touchdown_candidates.append(i)
        if _toeoff_position_constraint(df, i, swing_leg, takeoff_leg, direction_sign) and _toeoff_bar_height_constraint(df, i, takeoff_leg):
            toeoff_candidates.append(i)

    # 前两个点一定优先选膝角峰值
    if not touchdown_candidates:
        touchdown_candidates = [_window_fallback_peak(knee, window_start_row, window_end_row)]

    touchdown_row = min(touchdown_candidates)

    later_toeoff = [i for i in toeoff_candidates if i > touchdown_row]
    if later_toeoff:
        toeoff_row = min(later_toeoff)
    elif toeoff_candidates:
        toeoff_row = max(toeoff_candidates)
        if toeoff_row <= touchdown_row:
            later_all = [i for i in peaks if i > touchdown_row]
            if later_all:
                toeoff_row = later_all[0]
            else:
                toeoff_row = min(window_end_row, touchdown_row + 1)
    else:
        later_all = [i for i in peaks if i > touchdown_row]
        if later_all:
            toeoff_row = later_all[0]
        else:
            toeoff_row = min(window_end_row, touchdown_row + 1)

    return (
        int(df.iloc[touchdown_row]["frame_index"]),
        int(df.iloc[toeoff_row]["frame_index"]),
    )


def detect_swing_touchdown(
    df: pd.DataFrame,
    swing_leg: str,
    takeoff_toeoff_row: int,
    direction_sign: int,
    cfg: DetectConfig,
) -> Optional[int]:
    ay = df[f"{swing_leg}_ankle_ay"]

    cross_row = _leading_leg_cross_bar_row(df, swing_leg, direction_sign, joint="knee")
    if cross_row is None:
        return None
    if cross_row < takeoff_toeoff_row:
        cross_row = takeoff_toeoff_row

    sub_ay = ay.iloc[cross_row:].reset_index(drop=True)
    if len(sub_ay) < cfg.swing_min_segment_len:
        return None

    threshold = float(sub_ay.quantile(cfg.swing_low_acc_quantile))
    if not np.isfinite(threshold):
        return None

    low_mask = sub_ay <= threshold
    segments = _consecutive_segments(low_mask, cfg.swing_min_segment_len)
    if not segments:
        return None

    best = segments[0]
    idx = cross_row + best[0]
    return int(df.iloc[idx]["frame_index"])


def detect_swing_toeoff(
    df: pd.DataFrame,
    start_row: int,
    direction_sign: int,
    cfg: DetectConfig,
) -> Optional[int]:
    diff = df["ankle_y_diff"]
    pelvis_dx = df["pelvis_dx_to_bar"]

    for i in range(max(1, start_row), len(df) - 1):
        if not _is_after_bar(float(pelvis_dx.iloc[i]), direction_sign, cfg.post_bar_margin_px):
            continue

        prev_val = float(diff.iloc[i - 1])
        cur_val = float(diff.iloc[i])

        if prev_val == 0.0 or prev_val * cur_val <= 0:
            return int(df.iloc[min(i + 1, len(df) - 1)]["frame_index"])

    return None


def _fallback_swing_touchdown(df: pd.DataFrame, ref_row: int, cfg: DetectConfig) -> int:
    return int(df.iloc[min(len(df) - 1, ref_row + cfg.fallback_post_frames)]["frame_index"])


def _fallback_swing_toeoff(df: pd.DataFrame, ref_row: int, cfg: DetectConfig) -> int:
    return int(df.iloc[min(len(df) - 1, ref_row + cfg.fallback_post_frames * 2)]["frame_index"])


def validate_and_fix_order(df: pd.DataFrame, events: Dict[str, int], cfg: DetectConfig) -> Dict[str, int]:
    keys = ["takeoff_touchdown", "takeoff_toeoff", "swing_touchdown", "swing_toeoff"]
    rows = [_frame_to_row(df, events[k]) for k in keys]

    for i in range(1, len(rows)):
        if rows[i] <= rows[i - 1] + cfg.min_phase_gap_frames:
            rows[i] = min(len(df) - 1, rows[i - 1] + cfg.min_phase_gap_frames + 1)

    return {k: int(df.iloc[r]["frame_index"]) for k, r in zip(keys, rows)}


def run_event_detection(aligned_df: pd.DataFrame, cfg_dict: Dict) -> Tuple[pd.DataFrame, Dict]:
    det = cfg_dict.get("detection", {})
    smooth = cfg_dict.get("smoothing", {})

    cfg = DetectConfig(
        smooth_window=int(smooth.get("window_size", 5)),
        knee_angle_smooth_window=int(det.get("knee_angle_smooth_window", 5)),
        knee_peak_window=int(det.get("knee_peak_window", 3)),
        knee_peak_prominence_deg=float(det.get("knee_peak_prominence_deg", 2.0)),
        ankle_low_ratio=float(det.get("ankle_low_ratio", 0.20)),
        low_segment_min_len=int(det.get("low_segment_min_len", 3)),
        post_bar_margin_px=float(det.get("post_bar_margin_px", 10.0)),
        swing_low_acc_quantile=float(det.get("swing_low_acc_quantile", 0.18)),
        swing_min_segment_len=int(det.get("swing_min_segment_len", 3)),
        min_phase_gap_frames=int(det.get("min_phase_gap_frames", 2)),
        fallback_pre_frames=int(det.get("fallback_pre_frames", 18)),
        fallback_post_frames=int(det.get("fallback_post_frames", 12)),
    )

    df = compute_features(aligned_df, cfg)
    direction_sign = detect_direction(df)
    ref_row = find_reference_row(df)
    reference_frame = int(df.iloc[ref_row]["frame_index"])
    swing_leg, takeoff_leg = determine_leg_roles(df, ref_row, direction_sign)

    # 时间窗口结束点仍用摆动腿踝关节过栏前一帧
    window_cross_row = _leading_leg_cross_bar_row(df, swing_leg, direction_sign, joint="ankle")
    if window_cross_row is None:
        window_cross_row = ref_row

    # 实际过栏关键帧定义为 leading leg 膝关节过栏
    cross_bar_row = _leading_leg_cross_bar_row(df, swing_leg, direction_sign, joint="knee")
    if cross_bar_row is None:
        cross_bar_row = window_cross_row

    # 窗口起点：过栏前最后一次双踝 y 相交
    pre_bar_last_cross_row = _pre_cross_last_ankle_y_cross_row(df, window_cross_row)
    if pre_bar_last_cross_row is None:
        pre_bar_last_cross_row = 0

    # 窗口终点：踝关节过栏前一帧
    window_start_row = pre_bar_last_cross_row
    window_end_row = max(window_start_row, window_cross_row - 1)

    takeoff_touchdown, takeoff_toeoff = detect_takeoff_touchdown_and_toeoff(
        df=df,
        swing_leg=swing_leg,
        takeoff_leg=takeoff_leg,
        direction_sign=direction_sign,
        cfg=cfg,
        window_start_row=window_start_row,
        window_end_row=window_end_row,
    )

    takeoff_touchdown_row = _frame_to_row(df, takeoff_touchdown)
    takeoff_toeoff_row = _frame_to_row(df, takeoff_toeoff)

    # 前两个关键点强制夹回窗口内
    takeoff_touchdown_row = min(takeoff_touchdown_row + 1, window_end_row)
    takeoff_toeoff_row = min(takeoff_toeoff_row + 1, window_end_row)

    if takeoff_toeoff_row <= takeoff_touchdown_row:
        if takeoff_touchdown_row < window_end_row:
            takeoff_toeoff_row = takeoff_touchdown_row + 1
        else:
            takeoff_touchdown_row = max(window_start_row, takeoff_toeoff_row - 1)

    takeoff_touchdown = int(df.iloc[takeoff_touchdown_row]["frame_index"])
    takeoff_toeoff = int(df.iloc[takeoff_toeoff_row]["frame_index"])

    if takeoff_toeoff_row <= takeoff_touchdown_row:
        if takeoff_touchdown_row < window_end_row:
            takeoff_toeoff_row = takeoff_touchdown_row + 1
        else:
            takeoff_touchdown_row = max(window_start_row, takeoff_toeoff_row - 1)

    takeoff_touchdown = int(df.iloc[takeoff_touchdown_row]["frame_index"])
    takeoff_toeoff = int(df.iloc[takeoff_toeoff_row]["frame_index"])

    swing_touchdown = detect_swing_touchdown(
        df, swing_leg, takeoff_toeoff_row, direction_sign, cfg
    ) or _fallback_swing_touchdown(df, ref_row, cfg)
    swing_touchdown_row = _frame_to_row(df, swing_touchdown)

    swing_toeoff = detect_swing_toeoff(
        df, swing_touchdown_row, direction_sign, cfg
    ) or _fallback_swing_toeoff(df, ref_row, cfg)

    events = {
        "takeoff_touchdown": int(takeoff_touchdown),
        "takeoff_toeoff": int(takeoff_toeoff),
        "swing_touchdown": int(swing_touchdown),
        "swing_toeoff": int(swing_toeoff),
    }
    events = validate_and_fix_order(df, events, cfg)

    # validate 后再次保证前两个点在窗口里
    td_row = min(max(_frame_to_row(df, events["takeoff_touchdown"]), window_start_row), window_end_row)
    to_row = min(max(_frame_to_row(df, events["takeoff_toeoff"]), window_start_row), window_end_row)
    if to_row <= td_row:
        if td_row < window_end_row:
            to_row = td_row + 1
        else:
            td_row = max(window_start_row, to_row - 1)

    events["takeoff_touchdown"] = int(df.iloc[td_row]["frame_index"])
    events["takeoff_toeoff"] = int(df.iloc[to_row]["frame_index"])

    meta = {
        "direction_sign": int(direction_sign),
        "reference_frame": int(reference_frame),
        "swing_leg": swing_leg,
        "takeoff_leg": takeoff_leg,
        "pre_bar_last_cross_row": int(df.iloc[pre_bar_last_cross_row]["frame_index"]),
        "window_cross_row": int(df.iloc[window_cross_row]["frame_index"]),
        "cross_bar_row": int(df.iloc[cross_bar_row]["frame_index"]),
        "events": events,
    }
    return df, meta
