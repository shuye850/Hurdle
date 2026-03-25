from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from predictor import HurdleScorePredictor
from report_generator import render_html_report
from result_formatter import build_flat_result_frame, build_result_payloads
from rule_feedback import attach_rule_feedback
from visualization import render_summary_chart


DEFAULT_METRICS_CSV = Path(__file__).resolve().parents[1] / "5_特征融合" / "output" / "csv" / "summary" / "technical_metrics_all.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模块六：技术评价与反馈")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "deployable_models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", default=None, help="只处理指定 video_id")
    return parser.parse_args()


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    csv_dir = root / "csv"
    json_dir = root / "json"
    report_dir = root / "reports"
    vis_dir = root / "visualization"
    for path in [root, csv_dir, json_dir, report_dir, vis_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "csv": csv_dir,
        "json": json_dir,
        "reports": report_dir,
        "visualization": vis_dir,
    }


def main() -> None:
    args = parse_args()
    print(f"[模块六] 读取指标文件: {args.metrics_csv}")
    metrics_df = pd.read_csv(args.metrics_csv)
    if args.video_id is not None:
        metrics_df = metrics_df[metrics_df["video_id"].astype(str) == str(args.video_id)].reset_index(drop=True)

    if metrics_df.empty:
        raise ValueError("没有可评分的样本，请检查 metrics csv 或 video_id。")

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
        chart_path = render_summary_chart(payload, output_dirs["visualization"] / f"{stem}_summary.png")
        report_path = render_html_report(payload, chart_path, output_dirs["reports"] / f"{stem}_report.html")
        payload["artifacts"] = {
            "chart_png": str(chart_path),
            "report_html": str(report_path),
        }
        json_path = output_dirs["json"] / f"{stem}_result.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = output_dirs["json"] / "score_results_all.json"
    summary_path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已完成 {len(payloads)} 个样本的评分与报告生成。")
    print(f"输出目录: {output_dirs['root']}")


if __name__ == "__main__":
    main()
