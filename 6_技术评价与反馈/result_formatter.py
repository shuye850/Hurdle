from __future__ import annotations

from typing import Any

import pandas as pd


STAGE_ORDER = ["takeoff", "flight", "landing", "overall"]


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def build_result_payloads(
    metrics_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    manifest_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_map = {str(entry["task_id"]): entry for entry in manifest_entries}
    payloads: list[dict[str, Any]] = []

    for idx, pred_row in prediction_df.iterrows():
        metric_row = metrics_df.iloc[idx]
        item_scores: list[dict[str, Any]] = []
        stage_scores: list[dict[str, Any]] = []
        overall_score: dict[str, Any] | None = None

        for task_id, entry in manifest_map.items():
            score = _safe_float(pred_row.get(task_id))
            rounded = _safe_int(pred_row.get(f"{task_id}_round"))
            score_row = {
                "task_id": task_id,
                "task_name": entry["task_name"],
                "stage": entry["stage"],
                "tier": entry["tier"],
                "model_name": entry["model_name"],
                "feature_mode": entry["feature_mode"],
                "score": score,
                "score_round": rounded,
            }
            if task_id.endswith("_mean"):
                if entry["stage"] == "overall":
                    overall_score = score_row
                else:
                    stage_scores.append(score_row)
            else:
                item_scores.append(score_row)

        item_scores.sort(key=lambda x: (STAGE_ORDER.index(x["stage"]), x["task_id"]))
        stage_scores.sort(key=lambda x: STAGE_ORDER.index(x["stage"]))

        payloads.append(
            {
                "video_id": str(metric_row.get("video_id", "")),
                "sample_name": str(metric_row.get("sample_name", metric_row.get("video_id", ""))),
                "source_video_name": str(metric_row.get("source_video_name", "")),
                "source_video_stem": str(metric_row.get("source_video_stem", "")),
                "item_scores": item_scores,
                "stage_scores": stage_scores,
                "overall_score": overall_score,
            }
        )

    return payloads


def build_flat_result_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        row = {
            "video_id": payload["video_id"],
            "sample_name": payload["sample_name"],
            "source_video_name": payload["source_video_name"],
            "source_video_stem": payload["source_video_stem"],
        }
        for score_item in payload["item_scores"]:
            row[score_item["task_id"]] = score_item["score"]
            row[f"{score_item['task_id']}_round"] = score_item["score_round"]
        for score_item in payload["stage_scores"]:
            row[score_item["task_id"]] = score_item["score"]
            row[f"{score_item['task_id']}_round"] = score_item["score_round"]
        if payload["overall_score"] is not None:
            row["overall_mean"] = payload["overall_score"]["score"]
            row["overall_mean_round"] = payload["overall_score"]["score_round"]
        rows.append(row)
    return pd.DataFrame(rows)
