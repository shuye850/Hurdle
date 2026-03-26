from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from history_library import merge_history_library
from predictor import HurdleScorePredictor
from visualization import render_summary_chart
from report_assets import (
    build_event_frame_records,
    load_event_row,
)
from report_generator import render_html_report
from result_formatter import build_flat_result_frame, build_result_payloads
from rule_feedback import attach_rule_feedback


MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_TECHNICAL_DIR = PROJECT_ROOT / "5_特征融合" / "output" / "csv" / "technical"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "output"
DEFAULT_FRAME_DIR = PROJECT_ROOT / "5_特征融合" / "output" / "csv" / "frame"
DEFAULT_EVENT_DIR = PROJECT_ROOT / "5_特征融合" / "output" / "csv" / "event"
DEFAULT_FUSION_VIDEO_DIR = PROJECT_ROOT / "5_特征融合" / "output" / "video"
DEFAULT_PREPROCESS_VIDEO_DIR = PROJECT_ROOT / "1_视频预处理" / "output"
DEFAULT_HISTORY_LIBRARY_JSON = DEFAULT_OUTPUT_DIR / "json" / "history_sample_library.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模块六：技术评价与反馈")
    parser.add_argument("--technical-dir", type=Path, default=DEFAULT_TECHNICAL_DIR)
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--event-dir", type=Path, default=DEFAULT_EVENT_DIR)
    parser.add_argument("--fusion-video-dir", type=Path, default=DEFAULT_FUSION_VIDEO_DIR)
    parser.add_argument("--preprocess-video-dir", type=Path, default=DEFAULT_PREPROCESS_VIDEO_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODULE_DIR / "deployable_models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-library-json", type=Path, default=DEFAULT_HISTORY_LIBRARY_JSON)
    parser.add_argument("--video-id", default=None, help="只处理指定 video_id")
    return parser.parse_args()


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


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _jsonable_metric_value(value: object) -> object | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _jsonable_metric_value(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        rounded = round(value, 6)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return str(value)


def _build_full_metric_payload(metric_row: pd.Series) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in metric_row.items():
        converted = _jsonable_metric_value(value)
        if converted is None:
            continue
        payload[str(key)] = converted
    return payload


def _cleanup_path(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _build_technical_snapshot(metric_row: pd.Series) -> list[dict[str, object]]:
    metric_defs = [
        ("takeoff_distance_m", "起跨距离", "m", "takeoff"),
        ("takeoff_angle_deg", "起跨角", "deg", "takeoff"),
        ("flight_time_s", "腾空时间", "s", "flight"),
        ("flight_disp_m", "腾空水平位移", "m", "flight"),
        ("com_rise_m", "重心抬升高度", "m", "flight"),
        ("bar_cross_com_clearance_m", "过栏重心净空", "m", "flight"),
        ("landing_distance_m", "下栏距离", "m", "landing"),
        ("landing_knee_angle_deg", "下栏着地膝角", "deg", "landing"),
        ("push_amplitude_deg", "下栏蹬伸幅度", "deg", "landing"),
        ("recovery_time_s", "恢复时间", "s", "landing"),
        ("total_phase_time_s", "总技术阶段时长", "s", "overall"),
        ("takeoff_phase_ratio", "起跨阶段占比", "", "overall"),
        ("flight_phase_ratio", "腾空阶段占比", "", "overall"),
        ("landing_phase_ratio", "下栏阶段占比", "", "overall"),
    ]

    rows: list[dict[str, object]] = []
    for key, label, unit, stage in metric_defs:
        value = metric_row.get(key)
        safe_value = _safe_float(value)
        if safe_value is None:
            continue
        rows.append({"key": key, "label": label, "value": safe_value, "unit": unit, "stage": stage})
    return rows


def _normalized_ids(value: str | int | float | None) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    ids = [text]
    if text.isdigit():
        ids.append(text.lstrip("0") or "0")
        ids.append(text.zfill(3))
    return list(dict.fromkeys(ids))


def _matches_video_id(row: pd.Series, target: str | None) -> bool:
    if target is None:
        return True

    row_ids: list[str] = []
    row_ids.extend(_normalized_ids(row.get("video_id")))
    row_ids.extend(_normalized_ids(row.get("sample_name")))
    source_name = row.get("source_video_name")
    if isinstance(source_name, str) and source_name:
        row_ids.extend(_normalized_ids(Path(source_name).stem))

    target_ids = set(_normalized_ids(target))
    return any(item in target_ids for item in row_ids)


def _read_technical_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        dtype={
            "video_id": "string",
            "sample_name": "string",
            "source_video_name": "string",
            "source_video_stem": "string",
        },
    )


def load_metrics_frame(technical_dir: Path, video_id: str | None = None) -> pd.DataFrame:
    if not technical_dir.exists():
        raise FileNotFoundError(f"技术指标目录不存在: {technical_dir}")

    rows: list[dict[str, object]] = []
    for csv_path in sorted(technical_dir.glob("*_technical_metrics.csv")):
        df = _read_technical_csv(csv_path)
        if df.empty:
            continue
        row = df.iloc[0]
        if not _matches_video_id(row, video_id):
            continue
        rows.append(row.to_dict())

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    print(f"[模块六] 读取单样本技术指标目录: {args.technical_dir}")
    metrics_df = load_metrics_frame(args.technical_dir, args.video_id)

    if metrics_df.empty:
        raise ValueError("没有可评分的样本，请检查模块五单样本 technical CSV 或 video_id。")

    print(f"[模块六] 待处理样本数: {len(metrics_df)}")
    print(f"[模块六] 加载模型目录: {args.model_dir}")
    predictor = HurdleScorePredictor(args.model_dir)
    print("[模块六] 模型加载完成，开始评分")
    pred_df = predictor.predict(metrics_df)

    payloads = build_result_payloads(metrics_df, pred_df, predictor.get_enabled_entries())
    payloads = [attach_rule_feedback(payload) for payload in payloads]
    flat_df = build_flat_result_frame(payloads)

    output_dirs = ensure_output_dirs(args.output_dir)
    print(f"[模块六] 写出评分表: {output_dirs['csv'] / 'score_results.csv'}")
    flat_df.to_csv(output_dirs["csv"] / "score_results.csv", index=False, encoding="utf-8-sig")

    for idx, payload in enumerate(payloads, start=1):
        stem = payload["sample_name"]
        print(f"[模块六] ({idx}/{len(payloads)}) 生成报告: {stem}")
        metric_row = metrics_df.iloc[idx - 1]
        event_row = load_event_row(args.event_dir, stem, payload["video_id"])

        payload["technical_metrics"] = _build_technical_snapshot(metric_row)
        payload["all_metrics"] = _build_full_metric_payload(metric_row)
        payload["event_frames"] = build_event_frame_records(event_row)

        chart_path = render_summary_chart(payload, output_dirs["visualization"] / f"{stem}_summary.png")
        report_path = render_html_report(payload, chart_path, output_dirs["reports"] / f"{stem}_report.html")
        payload["artifacts"] = {
            "chart_png": "",
            "trajectory_png": "",
            "report_html": str(report_path),
        }
        json_path = output_dirs["json"] / f"{stem}_result.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        _cleanup_path(chart_path)
        _cleanup_path(output_dirs["visualization"] / f"{stem}_com_trajectory.png")
        _cleanup_path(output_dirs["screenshots"] / stem)

    summary_path = output_dirs["json"] / "score_results_all.json"
    summary_path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    merge_history_library(payloads, metrics_df, args.history_library_json)

    print(f"已完成 {len(payloads)} 个样本的评分与报告生成。")
    print(f"输出目录: {output_dirs['root']}")


if __name__ == "__main__":
    main()
