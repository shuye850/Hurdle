from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _clean_metrics_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in row.items()}


def _record_key(payload: dict[str, Any]) -> str:
    video_id = str(payload.get("video_id", "")).strip()
    sample_name = str(payload.get("sample_name", "")).strip()
    return video_id or sample_name


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def merge_history_library(
    payloads: list[dict[str, Any]],
    metrics_df: pd.DataFrame,
    library_path: Path,
) -> list[dict[str, Any]]:
    existing = _load_existing(library_path)
    merged: dict[str, dict[str, Any]] = {}

    for record in existing:
        key = str(record.get("video_id", "") or record.get("sample_name", "")).strip()
        if key:
            merged[key] = record

    updated_at = datetime.now().isoformat(timespec="seconds")
    for payload, (_, metric_row) in zip(payloads, metrics_df.iterrows()):
        key = _record_key(payload)
        if not key:
            continue
        merged[key] = {
            "video_id": str(payload.get("video_id", "")),
            "sample_name": str(payload.get("sample_name", "")),
            "source_video_name": str(payload.get("source_video_name", "")),
            "updated_at": updated_at,
            "score_payload": payload,
            "technical_metrics": _clean_metrics_row(metric_row.to_dict()),
            "artifacts": payload.get("artifacts", {}),
        }

    library = [merged[key] for key in sorted(merged)]
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    return library
