from __future__ import annotations

from typing import Any

import pandas as pd

from problem_library import PROBLEM_LIBRARY, STAGE_LABELS, TASK_DIRECTION_RULES


METRIC_LABELS = {
    "takeoff_distance_m": "起跨点距离",
    "takeoff_support_time_s": "起跨支撑时长",
    "takeoff_phase_ratio": "起跨阶段占比",
    "takeoff_buffer_amplitude_deg": "起跨缓冲幅度",
    "takeoff_landing_knee_angle_deg": "起跨着地膝角",
    "takeoff_angle_deg": "起跨角",
    "takeoff_landing_angle_deg": "起跨着地角",
    "takeoff_toeoff_trunk_angle_deg": "起跨离地躯干角",
    "flight_time_s": "腾空时间",
    "toeoff_to_bar_ratio": "离地到过栏时间占比",
    "com_rise_m": "重心腾起高度",
    "bar_cross_com_clearance_m": "过栏重心超栏高量",
    "flight_com_speed_mps": "腾空平均水平速度",
    "bar_cross_lead_knee_angle_deg": "过栏摆动腿膝角",
    "toeoff_to_bar_time_s": "离地到过栏时间",
    "bar_cross_trunk_angle_deg": "过栏躯干角",
    "landing_distance_m": "下栏距离",
    "landing_com_dist_m": "落地重心距",
    "landing_knee_angle_deg": "下栏着地膝角",
    "landing_shank_angle_deg": "下栏着地小腿角",
    "push_amplitude_deg": "下栏蹬伸幅度",
    "recovery_time_s": "恢复时间",
    "landing_trunk_angle_deg": "下栏落地躯干角",
    "trunk_angle_change_deg": "躯干角变化量",
}


def _float_metric(metrics_row: dict[str, Any], key: str) -> float | None:
    value = metrics_row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _score_value(item: dict[str, Any]) -> float:
    score = item.get("score")
    return float(score) if score is not None else 0.0


def _metric_text(metrics_row: dict[str, Any], key: str) -> str | None:
    value = metrics_row.get(key)
    if value is None or pd.isna(value):
        return None
    if key.endswith("_deg"):
        return f"{METRIC_LABELS.get(key, key)}: {float(value):.2f} deg"
    if key.endswith("_mps"):
        return f"{METRIC_LABELS.get(key, key)}: {float(value):.3f} m/s"
    if key.endswith("_m"):
        return f"{METRIC_LABELS.get(key, key)}: {float(value):.3f} m"
    if key.endswith("_s"):
        return f"{METRIC_LABELS.get(key, key)}: {float(value):.3f} s"
    return f"{METRIC_LABELS.get(key, key)}: {float(value):.3f}"


def _heuristic_joint_diagnosis(task_id: str, metrics_row: dict[str, Any]) -> str | None:
    if task_id == "T4":
        takeoff_angle = _float_metric(metrics_row, "takeoff_angle_deg")
        landing_angle = _float_metric(metrics_row, "takeoff_landing_angle_deg")
        if takeoff_angle is None or landing_angle is None:
            return None
        diff = takeoff_angle - landing_angle
        if takeoff_angle < 60 and landing_angle < 40:
            return "起跨角和着地角都偏小，说明起跨后身体前送不足，入栏准备偏紧。"
        if takeoff_angle > 75 and landing_angle > 50:
            return "起跨角和着地角都偏大，说明起跨向上成分偏多，入栏动作容易发高发拖。"
        if diff > 35:
            return "起跨角与着地角差值偏大，说明起跨后到入栏的角度转换不够顺畅。"
        return None

    if task_id == "F2":
        com_rise = _float_metric(metrics_row, "com_rise_m")
        clearance = _float_metric(metrics_row, "bar_cross_com_clearance_m")
        if com_rise is None or clearance is None:
            return None
        if com_rise > 0.22 and clearance > 0.15:
            return "重心腾起高度和过栏超栏高量都偏大，说明腾空路径偏高，不够经济。"
        if com_rise < 0.10 and clearance < 0.04:
            return "重心腾起高度和过栏余量都偏小，说明过栏空间偏紧，动作调整压力较大。"
        return None

    if task_id == "L2":
        knee = _float_metric(metrics_row, "landing_knee_angle_deg")
        shank = _float_metric(metrics_row, "landing_shank_angle_deg")
        if knee is None or shank is None:
            return None
        if knee > 165 and shank > 65:
            return "下栏着地膝角偏大且小腿角偏立，说明落地偏硬，缓冲不足。"
        if knee < 130 and shank < 50:
            return "下栏着地膝角偏小且小腿前压明显，说明落地缓冲偏深，容易拖慢后续跑出。"
        return None

    if task_id == "L3":
        push_amp = _float_metric(metrics_row, "push_amplitude_deg")
        recovery_time = _float_metric(metrics_row, "recovery_time_s")
        if push_amp is None or recovery_time is None:
            return None
        if recovery_time > 0.09 and push_amp <= 20:
            return "下栏后停留时间偏长，且跑出动作不够紧凑，说明跨后连续跑出不够积极。"
        if recovery_time > 0.09 and push_amp > 20:
            return "下栏后恢复时间偏长，即使后续角度变化较大，也更像是缓冲过多后的补偿性蹬出。"
        if recovery_time <= 0.09 and push_amp <= 10:
            return "下栏后恢复速度尚可，腿部保持相对刚性，当前更应关注跑出是否足够主动和向前。"
        return None

    return None


def build_reference_profiles(
    library_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = []
    for record in library_records:
        payload = record.get("score_payload") or {}
        metrics_row = record.get("technical_metrics") or record.get("metrics_row") or {}
        if not payload or not metrics_row:
            continue
        row = {"video_id": str(record.get("video_id", ""))}
        for item in payload.get("item_scores", []):
            row[item["task_id"]] = item.get("score_round")
        row.update(metrics_row)
        rows.append(row)

    df = pd.DataFrame(rows)
    profiles: dict[str, dict[str, Any]] = {}
    if df.empty:
        return profiles

    for task_id, rule in TASK_DIRECTION_RULES.items():
        metric = rule["metric"]
        if task_id not in df.columns or metric not in df.columns:
            continue
        sub = df[[task_id, metric]].dropna().copy()
        if sub.empty:
            continue
        high = sub[sub[task_id] >= 4][metric]
        if len(high) < 4:
            high = sub[sub[task_id] >= 3][metric]
        if len(high) < 4:
            high = sub[metric]
        profiles[task_id] = {
            "metric": metric,
            "label": rule["label"],
            "low": float(high.quantile(0.25)),
            "high": float(high.quantile(0.75)),
        }
    return profiles


def _dynamic_diagnosis(
    task_id: str,
    base_diagnosis: str,
    metrics_row: dict[str, Any],
    reference_profiles: dict[str, dict[str, Any]],
) -> tuple[str, str | None]:
    rule = TASK_DIRECTION_RULES.get(task_id)
    profile = reference_profiles.get(task_id)
    if rule is None or profile is None:
        return base_diagnosis, None
    value = metrics_row.get(rule["metric"])
    if value is None or pd.isna(value):
        return base_diagnosis, None
    value = float(value)
    low = float(profile["low"])
    high = float(profile["high"])
    if abs(high - low) < 1e-9:
        return base_diagnosis, None
    range_line = f"{profile['label']}参考区间（高分样本常见范围）: {low:.3f} - {high:.3f}"
    if value < low:
        return rule["low_desc"], range_line
    if value > high:
        return rule["high_desc"], range_line
    return rule["within_desc"], range_line


def _build_problem_detail(
    item: dict[str, Any],
    metrics_row: dict[str, Any],
    reference_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_id = item["task_id"]
    profile = PROBLEM_LIBRARY.get(task_id, {})
    evidence = []
    for key in profile.get("metric_keys", []):
        line = _metric_text(metrics_row, key)
        if line is not None:
            evidence.append(line)
    diagnosis_text, range_line = _dynamic_diagnosis(
        task_id=task_id,
        base_diagnosis=profile.get("diagnosis", "该动作环节存在改进空间。"),
        metrics_row=metrics_row,
        reference_profiles=reference_profiles,
    )
    heuristic_text = _heuristic_joint_diagnosis(task_id, metrics_row)
    if heuristic_text:
        diagnosis_text = heuristic_text
    if range_line is not None:
        evidence.append(range_line)
    return {
        "task_id": task_id,
        "task_name": item["task_name"],
        "stage": item["stage"],
        "score": item["score"],
        "score_round": item["score_round"],
        "problem_title": profile.get("title", item["task_name"]),
        "diagnosis": diagnosis_text,
        "impact": profile.get("impact", "如果不调整，可能继续影响整栏动作衔接。"),
        "evidence": evidence,
        "recommended_drills": profile.get("drills", []),
        "next_focus": profile.get("focus", "下次训练时继续关注这一动作环节。"),
    }


def build_diagnosis(
    payload: dict[str, Any],
    metrics_row: dict[str, Any],
    reference_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item_scores = sorted(payload["item_scores"], key=lambda item: (_score_value(item), str(item["task_id"])))
    weak_items = [item for item in item_scores if item.get("score_round") is not None and int(item["score_round"]) <= 2]
    watch_items = [item for item in item_scores if item.get("score_round") is not None and int(item["score_round"]) == 3]
    priority_items = weak_items[:3] if weak_items else watch_items[:3]

    stage_feedback = payload.get("stage_feedback", [])
    weakest_stage = min(stage_feedback, key=lambda x: float(x.get("score") or 0.0)) if stage_feedback else None

    problems = [_build_problem_detail(item, metrics_row, reference_profiles) for item in priority_items]
    drills: list[str] = []
    next_focus: list[str] = []
    for problem in problems:
        drills.extend(problem["recommended_drills"])
        next_focus.append(problem["next_focus"])
    drills = list(dict.fromkeys(drills))[:5]
    next_focus = list(dict.fromkeys(next_focus + payload.get("rule_feedback", [])))[:5]

    overall = payload.get("overall_score")
    overall_text = "本次动作整体还有明显提升空间。"
    if overall and overall.get("score") is not None:
        score = float(overall["score"])
        if score >= 4.0:
            overall_text = "本次整体动作表现较好，主要是少数环节仍可继续优化。"
        elif score >= 3.0:
            overall_text = "本次整体动作处于中等水平，已经有基础，但还有明确短板。"

    return {
        "video_id": payload["video_id"],
        "sample_name": payload["sample_name"],
        "source_video_name": payload["source_video_name"],
        "source_video_stem": payload["source_video_stem"],
        "overall_score": payload.get("overall_score"),
        "overall_summary": overall_text,
        "weakest_stage": weakest_stage,
        "stage_diagnosis": stage_feedback,
        "top_problems": problems,
        "recommended_drills": drills,
        "next_focus": next_focus,
        "all_metrics": metrics_row,
    }


def build_summary_frame(diagnoses: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for diagnosis in diagnoses:
        weakest_stage = diagnosis.get("weakest_stage") or {}
        top_problems = diagnosis.get("top_problems", [])
        rows.append(
            {
                "video_id": diagnosis["video_id"],
                "sample_name": diagnosis["sample_name"],
                "overall_score": float(diagnosis["overall_score"]["score"]) if diagnosis.get("overall_score") else None,
                "weakest_stage": weakest_stage.get("stage_name", STAGE_LABELS.get(weakest_stage.get("stage", ""), "")),
                "top_problem_1": top_problems[0]["problem_title"] if len(top_problems) > 0 else "",
                "top_problem_2": top_problems[1]["problem_title"] if len(top_problems) > 1 else "",
                "top_problem_3": top_problems[2]["problem_title"] if len(top_problems) > 2 else "",
                "next_focus_1": diagnosis["next_focus"][0] if len(diagnosis["next_focus"]) > 0 else "",
                "next_focus_2": diagnosis["next_focus"][1] if len(diagnosis["next_focus"]) > 1 else "",
                "next_focus_3": diagnosis["next_focus"][2] if len(diagnosis["next_focus"]) > 2 else "",
            }
        )
    return pd.DataFrame(rows)
