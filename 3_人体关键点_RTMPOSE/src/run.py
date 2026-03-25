# src/run.py

from __future__ import annotations

import os
from pathlib import Path

# Mac 兼容设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from io_video import get_module1_output_dir, list_video_files, ensure_dir
from infer_pose import build_inferencer, infer_single_video
from export_csv import export_pose_csv
from visualize_pose import visualize_single_video


def build_module3_output_dirs(current_file: str | Path) -> dict[str, Path]:
    """
    构建第三模块输出目录。

    目录结构:
        3_人体关键点_RTMPOSE/
        └── output/
            ├── csv/
            └── vis/
    """
    current_file = Path(current_file).resolve()
    module_root = current_file.parents[1]
    output_root = ensure_dir(module_root / "output")
    csv_dir = ensure_dir(output_root / "csv")
    vis_dir = ensure_dir(output_root / "vis")

    return {
        "module_root": module_root,
        "output_root": output_root,
        "csv_dir": csv_dir,
        "vis_dir": vis_dir,
    }


def debug_run_pose_module(
    schema_name: str = "body17",
    device: str = "cpu",
    pose2d: str | None = None,
    pose2d_weights: str | None = None,
    det_model: str = "rtmdet-l",
    det_cat_ids: list[int] | None = None,
    bbox_thr: float = 0.3,
    kpt_thr: float = 0.4,
    vis_score_thr: float = 0.4,
    draw_box: bool = True,
) -> dict:
    """
    第三模块独立调试入口：
    模型只初始化一次，然后按视频顺序逐个处理；每个视频处理完立刻导出 csv 和可视化视频。
    """
    if det_cat_ids is None:
        det_cat_ids = [0]

    module1_output_dir = get_module1_output_dir(__file__)
    video_paths = list_video_files(module1_output_dir)

    if not video_paths:
        raise FileNotFoundError(f"在模块一输出目录中未找到视频文件: {module1_output_dir}")

    output_dirs = build_module3_output_dirs(__file__)

    print("=" * 60)
    print("第三模块调试启动")
    print(f"输入目录: {module1_output_dir}")
    print(f"关键点方案: {schema_name}")
    print(f"检测模型: {det_model}")
    print(f"姿态模型: {pose2d if pose2d is not None else '(使用 schema 默认值)'}")
    print(f"设备: {device}")
    print(f"视频数量: {len(video_paths)}")
    print("=" * 60)

    inferencer = build_inferencer(
        schema_name=schema_name,
        device=device,
        pose2d=pose2d,
        pose2d_weights=pose2d_weights,
        det_model=det_model,
        det_cat_ids=det_cat_ids,
    )

    csv_paths: list[str] = []
    vis_paths: list[str] = []

    for i, video_path in enumerate(video_paths, start=1):
        video_path = Path(video_path)
        video_stem = video_path.stem
        print(f"[{i}/{len(video_paths)}] 开始处理: {video_path.name}")

        frame_results = infer_single_video(
            video_path=video_path,
            schema_name=schema_name,
            inferencer=inferencer,
            device=device,
            pose2d=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_cat_ids=det_cat_ids,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
        )

        csv_path = output_dirs["csv_dir"] / f"{video_stem}_keypoints.csv"
        saved_csv = export_pose_csv(
            frame_results=frame_results,
            output_csv_path=csv_path,
            schema_name=schema_name,
        )
        csv_paths.append(str(saved_csv))

        vis_path = output_dirs["vis_dir"] / f"{video_stem}_vis.mp4"
        saved_vis = visualize_single_video(
            video_path=video_path,
            frame_results=frame_results,
            output_video_path=vis_path,
            schema_name=schema_name,
            score_thr=vis_score_thr,
            draw_box=draw_box,
        )
        vis_paths.append(str(saved_vis))

        print(f"[{i}/{len(video_paths)}] 完成: {video_path.name}")
        print(f"    CSV: {saved_csv}")
        print(f"    VIS: {saved_vis}")

    return {
        "schema_name": schema_name,
        "module1_output_dir": str(module1_output_dir),
        "num_videos": len(video_paths),
        "video_paths": [str(p) for p in video_paths],
        "csv_paths": csv_paths,
        "vis_paths": vis_paths,
    }


def main():
    result = debug_run_pose_module(
        schema_name="body17",
        device="cpu",
        pose2d=None,
        pose2d_weights=None,
        det_model="rtmdet-l",
        det_cat_ids=[0],
        bbox_thr=0.3,
        kpt_thr=0.4,
        vis_score_thr=0.4,
        draw_box=True,
    )

    print("\n第三模块调试完成")
    print(f"关键点方案: {result['schema_name']}")
    print(f"输入目录: {result['module1_output_dir']}")
    print(f"处理视频数: {result['num_videos']}")
    print("CSV 输出:")
    for p in result["csv_paths"]:
        print(f"  - {p}")
    print("可视化视频输出:")
    for p in result["vis_paths"]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
