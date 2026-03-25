from __future__ import annotations

from typing import Any


WEAK_THRESHOLD = 2
WATCH_THRESHOLD = 3

TASK_ADVICE = {
    "T1": "调整起跨点位置感，结合标志物做固定步点练习。",
    "T2": "加强起跨支撑节奏控制，避免支撑过长影响起跨效率。",
    "T3": "加强起跨腿缓冲能力，练习髋膝踝协同屈伸。",
    "T5": "结合分腿跨栏和低栏练习，优化起跨角与着地角匹配。",
    "T6": "关注起跨时上体前倾控制，减少摆动和晃动。",
    "F1": "通过节奏跨栏练习稳定腾空时间，避免过早或过晚下栏。",
    "F2": "优化腾空轨迹，减少重心不必要的上下波动。",
    "F3": "加强腾空中的水平推进，保持跨栏后向前送髋。",
    "F5": "强化摆动腿积极前摆与快速下压，提高过栏紧凑性。",
    "F6": "关注过栏时躯干稳定，保持顺着行进方向的合理前倾。",
    "L1": "调整下栏距离，避免落点过近或过远影响衔接。",
    "L3": "加强下栏着地缓冲与主动扒地，提升落地动作质量。",
    "L5": "强化下栏后蹬伸与快速恢复，提升衔接跑动效率。",
    "L6": "加强下栏阶段上体控制，减少起伏和摆动。",
}

STAGE_ADVICE = {
    "takeoff": "起跨阶段优先关注步点、支撑节奏和上体姿态。",
    "flight": "腾空阶段优先关注摆动腿技术、躯干稳定和水平推进。",
    "landing": "下栏阶段优先关注落点、着地缓冲和恢复蹬伸。",
}

STAGE_LABELS = {
    "takeoff": "起跨",
    "flight": "腾空",
    "landing": "下栏",
}


def attach_rule_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    item_scores = payload["item_scores"]
    weak_items = [item for item in item_scores if item["score_round"] is not None and item["score_round"] <= WEAK_THRESHOLD]
    watch_items = [
        item for item in item_scores
        if item["score_round"] is not None and WEAK_THRESHOLD < item["score_round"] <= WATCH_THRESHOLD
    ]
    strong_items = [item for item in item_scores if item["score_round"] is not None and item["score_round"] >= 4]

    stage_comments: list[dict[str, Any]] = []
    for stage_score in payload["stage_scores"]:
        stage = stage_score["stage"]
        rounded = stage_score["score_round"]
        if rounded is None:
            summary = f"{STAGE_LABELS.get(stage, stage)}阶段暂无评分。"
        elif rounded <= WEAK_THRESHOLD:
            summary = f"{STAGE_LABELS.get(stage, stage)}阶段是当前主要短板。{STAGE_ADVICE.get(stage, '')}"
        elif rounded <= WATCH_THRESHOLD:
            summary = f"{STAGE_LABELS.get(stage, stage)}阶段整体中等，仍有优化空间。{STAGE_ADVICE.get(stage, '')}"
        else:
            summary = f"{STAGE_LABELS.get(stage, stage)}阶段整体表现较好。"
        stage_comments.append(
            {
                "stage": stage,
                "stage_name": STAGE_LABELS.get(stage, stage),
                "score": stage_score["score"],
                "score_round": rounded,
                "summary": summary,
            }
        )

    advice_lines: list[str] = []
    for item in weak_items[:3]:
        advice = TASK_ADVICE.get(item["task_id"])
        if advice:
            advice_lines.append(advice)
    if not advice_lines and watch_items:
        for item in watch_items[:2]:
            advice = TASK_ADVICE.get(item["task_id"])
            if advice:
                advice_lines.append(advice)

    payload["weak_items"] = [
        {"task_id": item["task_id"], "task_name": item["task_name"], "score_round": item["score_round"]}
        for item in weak_items
    ]
    payload["strong_items"] = [
        {"task_id": item["task_id"], "task_name": item["task_name"], "score_round": item["score_round"]}
        for item in strong_items[:5]
    ]
    payload["stage_feedback"] = stage_comments
    payload["rule_feedback"] = advice_lines
    return payload
