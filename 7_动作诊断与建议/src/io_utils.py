from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODULE6_HISTORY_JSON = Path(__file__).resolve().parents[2] / "6_技术评价与反馈" / "output" / "json" / "history_sample_library.json"
DEFAULT_SCORE_JSON = Path(__file__).resolve().parents[2] / "6_技术评价与反馈" / "output" / "json" / "score_results_all.json"
DEFAULT_TECHNICAL_DIR = Path(__file__).resolve().parents[2] / "5_特征融合" / "output" / "csv" / "technical"
DEFAULT_FRAME_DIR = Path(__file__).resolve().parents[2] / "5_特征融合" / "output" / "csv" / "frame"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
DEFAULT_EVENT_CSV_DIR = Path(__file__).resolve().parents[2] / "5_特征融合" / "output" / "csv" / "event"
DEFAULT_FUSION_VIDEO_DIR = Path(__file__).resolve().parents[2] / "5_特征融合" / "output" / "video"
DEFAULT_MODULE5_VIS_DIR = Path(__file__).resolve().parents[2] / "5_特征融合" / "output" / "visualization"
DEFAULT_MODULE4_JSON_DIR = Path(__file__).resolve().parents[2] / "4_阶段划分" / "output" / "json"
DEFAULT_PREPROCESS_VIDEO_DIR = Path(__file__).resolve().parents[2] / "1_视频预处理" / "output"
DEFAULT_REFERENCE_LIBRARY_JSON = DEFAULT_OUTPUT_DIR / "json" / "reference_library.json"


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    csv_dir = root / "csv"
    json_dir = root / "json"
    report_dir = root / "reports"
    vis_dir = root / "visualization"
    screenshot_dir = root / "screenshots"
    for path in [root, csv_dir, json_dir, report_dir, vis_dir, screenshot_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "csv": csv_dir,
        "json": json_dir,
        "reports": report_dir,
        "visualization": vis_dir,
        "screenshots": screenshot_dir,
    }


def load_score_results(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_score_result_directory(root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def load_module6_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


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


def _register_metrics_row(metrics_map: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    keys: list[str] = []
    keys.extend(_normalized_ids(row.get("video_id")))
    keys.extend(_normalized_ids(row.get("sample_name")))
    source_name = row.get("source_video_name")
    if isinstance(source_name, str) and source_name:
        keys.extend(_normalized_ids(Path(source_name).stem))
    for key in dict.fromkeys(keys):
        metrics_map[key] = row


def load_metrics_map(technical_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_map: dict[str, dict[str, Any]] = {}
    for csv_path in sorted(technical_dir.glob("*_technical_metrics.csv")):
        df = pd.read_csv(
            csv_path,
            dtype={
                "video_id": "string",
                "sample_name": "string",
                "source_video_name": "string",
                "source_video_stem": "string",
            },
        )
        if df.empty:
            continue
        _register_metrics_row(metrics_map, df.iloc[0].to_dict())
    return metrics_map


def load_event_map(event_dir: Path) -> dict[str, dict[str, Any]]:
    event_map: dict[str, dict[str, Any]] = {}
    for csv_path in sorted(event_dir.glob("*_event_frames.csv")):
        df = pd.read_csv(
            csv_path,
            dtype={
                "video_id": "string",
                "sample_name": "string",
                "source_video_name": "string",
                "source_video_stem": "string",
            },
        )
        if df.empty:
            continue
        _register_metrics_row(event_map, df.iloc[0].to_dict())
    return event_map


def match_video_id(value: str, target: str | None) -> bool:
    if target is None:
        return True
    return any(item in set(_normalized_ids(target)) for item in _normalized_ids(value))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_score_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("score_payload") or {}
    return payload if isinstance(payload, dict) else {}


def get_metrics_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics_row = record.get("technical_metrics") or record.get("metrics_row") or {}
    return metrics_row if isinstance(metrics_row, dict) else {}


def build_records_from_legacy_inputs(
    score_results: list[dict[str, Any]],
    metrics_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in score_results:
        video_id = str(payload.get("video_id", ""))
        sample_name = str(payload.get("sample_name", ""))
        metrics_row = metrics_map.get(video_id) or metrics_map.get(sample_name)
        if not payload or metrics_row is None:
            continue
        records.append(
            {
                "video_id": video_id,
                "sample_name": sample_name,
                "source_video_name": str(payload.get("source_video_name", "")),
                "score_payload": payload,
                "technical_metrics": metrics_row,
                "artifacts": payload.get("artifacts", {}),
            }
        )
    return records


def load_reference_library(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def merge_reference_library(
    current_payloads: list[dict[str, Any]],
    metrics_map: dict[str, dict[str, Any]],
    library_path: Path,
) -> list[dict[str, Any]]:
    existing = load_reference_library(library_path)
    merged: dict[str, dict[str, Any]] = {}

    for record in existing:
        video_id = str(record.get("video_id", ""))
        if video_id:
            merged[video_id] = record

    for payload in current_payloads:
        video_id = str(payload.get("video_id", ""))
        metrics_row = metrics_map.get(video_id) or metrics_map.get(str(payload.get("sample_name", "")))
        if not video_id or metrics_row is None:
            continue
        merged[video_id] = {
            "video_id": video_id,
            "sample_name": str(payload.get("sample_name", "")),
            "score_payload": payload,
            "metrics_row": metrics_row,
        }

    library = [merged[key] for key in sorted(merged)]
    save_json(library_path, library)
    return library
