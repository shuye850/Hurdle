from __future__ import annotations

from typing import Any


CANONICAL_TASK_NAMES: dict[str, str] = {
    "T1": "起跨点距离",
    "T2": "起跨支撑时长合理性",
    "T3": "起跨腿缓冲质量",
    "T4": "起跨角与着地角合理性",
    "T5": "起跨上体姿态",
    "F1": "腾空时间合理性",
    "F2": "腾空轨迹经济性",
    "F3": "腾空水平推进效果",
    "F4": "过栏摆动腿技术",
    "F5": "过栏上体姿态",
    "L1": "下栏距离合理性",
    "L2": "下栏着地动作质量",
    "L3": "下栏蹬伸质量",
    "L4": "下栏上体控制",
    "A1": "起跨阶段平均分",
    "A2": "腾空阶段平均分",
    "A3": "下栏阶段平均分",
    "A4": "总分",
}

STAGE_SCORE_TASK_IDS = {"A1", "A2", "A3", "A4"}


def canonical_task_id(task_id: Any) -> str:
    return str(task_id)


def canonical_task_name(task_id: Any, fallback_name: Any = "") -> str:
    task_key = canonical_task_id(task_id)
    return CANONICAL_TASK_NAMES.get(task_key, str(fallback_name))


def enrich_task_display(item: dict[str, Any]) -> dict[str, Any]:
    task_key = canonical_task_id(item.get("task_id", ""))
    enriched = dict(item)
    enriched["task_id"] = task_key
    enriched["task_name"] = canonical_task_name(task_key, item.get("task_name", ""))
    enriched["display_task_id"] = task_key
    enriched["display_task_name"] = enriched["task_name"]
    return enriched
