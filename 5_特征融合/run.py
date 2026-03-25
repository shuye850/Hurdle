# -*- coding: utf-8 -*-
from pathlib import Path
import traceback

from src.config_loader import load_config
from src.io_utils import (
    discover_samples,
    ensure_output_dirs,
    load_sample_inputs,
    save_outputs,
)
from src.metrics import process_sample
from src.video_renderer import render_fusion_video
from src.visualization import render_summary_figure


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cfg = load_config(project_root / "config" / "default.yaml")

    output_dirs = ensure_output_dirs(project_root, cfg)
    samples = discover_samples(cfg)

    if not samples:
        print("未发现可处理样本，请检查上游 output 路径与命名。")
        return

    all_metrics = []
    all_events = []
    all_frames = []

    for sample_name, sample_paths in samples.items():
        print("=" * 80)
        print(f"开始处理样本: {sample_name}")

        try:
            hurdle_df, pose_df, events_json, phase_json, segmented_df, video_path, source_video_name, source_video_stem = load_sample_inputs(
                sample_name=sample_name,
                sample_paths=sample_paths,
            )

            result = process_sample(
                sample_name=sample_name,
                source_video_name=source_video_name,
                source_video_stem=source_video_stem,
                hurdle_df=hurdle_df,
                pose_df=pose_df,
                events_json=events_json,
                phase_json=phase_json,
                cfg=cfg,
            )

            frame_df = result["frame_features"]
            event_df = result["event_frames"]
            metric_df = result["technical_metrics"]

            # 单样本分别保存
            metric_df.to_csv(
                output_dirs["technical"] / f"{sample_name}_technical_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            event_df.to_csv(
                output_dirs["event"] / f"{sample_name}_event_frames.csv",
                index=False,
                encoding="utf-8-sig",
            )
            frame_df.to_csv(
                output_dirs["frame"] / f"{sample_name}_frame_features.csv",
                index=False,
                encoding="utf-8-sig",
            )

            all_frames.append(frame_df)
            all_events.append(event_df)
            all_metrics.append(metric_df)

            sample_video_out = output_dirs["video"] / f"{sample_name}_fusion_result.mp4"
            sample_fig_out = output_dirs["visualization"] / f"{sample_name}_fusion_summary.png"

            render_fusion_video(
                sample_name=sample_name,
                video_path=video_path,
                frame_df=frame_df,
                event_df=event_df,
                cfg=cfg,
                output_path=sample_video_out,
            )

            render_summary_figure(
                sample_name=sample_name,
                frame_df=frame_df,
                event_df=event_df,
                metric_df=metric_df,
                video_path=video_path,
                cfg=cfg,
                output_path=sample_fig_out,
            )

            print(f"完成样本: {sample_name}")

        except Exception as exc:
            print(f"处理失败: {sample_name}")
            print(f"错误信息: {exc}")
            traceback.print_exc()

    save_outputs(
        all_metrics=all_metrics,
        all_events=all_events,
        all_frames=all_frames,
        output_dirs=output_dirs,
    )

    print("=" * 80)
    print("模块五处理完成。")


if __name__ == "__main__":
    main()
