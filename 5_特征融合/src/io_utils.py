# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd


def ensure_output_dirs(project_root: Path, cfg: dict) -> Dict[str, Path]:
    output_root = Path(cfg["paths"]["current_output_root"])

    csv_root = output_root / "csv"
    technical_dir = csv_root / "technical"
    event_dir = csv_root / "event"
    frame_dir = csv_root / "frame"
    summary_dir = csv_root / "summary"

    video_dir = output_root / "video"
    vis_dir = output_root / "visualization"

    csv_root.mkdir(parents=True, exist_ok=True)
    technical_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_root,
        "csv_root": csv_root,
        "technical": technical_dir,
        "event": event_dir,
        "frame": frame_dir,
        "summary": summary_dir,
        "video": video_dir,
        "visualization": vis_dir,
    }


def discover_samples(cfg: dict) -> Dict[str, Dict[str, Optional[Path]]]:
    hurdle_dir = Path(cfg["paths"]["upstream"]["hurdle_output"])
    pose_dir = Path(cfg["paths"]["upstream"]["pose_output"])
    stage_csv_dir = Path(cfg["paths"]["upstream"]["stage_csv_output"])
    stage_json_dir = Path(cfg["paths"]["upstream"]["stage_json_output"])
    video_dir = Path(cfg["paths"]["upstream"]["preprocess_output"])

    pose_suffix = cfg["file_match"]["pose_csv_suffix"]
    segmented_suffix = cfg["file_match"]["segmented_csv_suffix"]
    events_suffix = cfg["file_match"]["events_json_suffix"]
    phase_suffix = cfg["file_match"]["phase_json_suffix"]
    video_exts = [ext.lower() for ext in cfg["file_match"]["video_extensions"]]

    # 栏架文件：普通 csv，但排除 keypoints / segmented
    hurdle_map = {}
    for p in hurdle_dir.glob("*.csv"):
        if p.name.endswith(pose_suffix) or p.name.endswith(segmented_suffix):
            continue
        hurdle_map[p.stem] = p

    # 人体关键点
    pose_map = {}
    for p in pose_dir.glob(f"*{pose_suffix}"):
        stem = p.name[:-len(pose_suffix)]
        pose_map[stem] = p

    # 模块四 output/csv
    segmented_map = {}
    for p in stage_csv_dir.glob(f"*{segmented_suffix}"):
        stem = p.name[:-len(segmented_suffix)]
        segmented_map[stem] = p

    # 模块四 output/json
    events_map = {}
    for p in stage_json_dir.glob(f"*{events_suffix}"):
        stem = p.name[:-len(events_suffix)]
        events_map[stem] = p

    phase_map = {}
    for p in stage_json_dir.glob(f"*{phase_suffix}"):
        stem = p.name[:-len(phase_suffix)]
        phase_map[stem] = p

    # 视频
    video_map = {}
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in video_exts:
            video_map[p.stem] = p

    sample_names = sorted(set(hurdle_map) & set(pose_map) & set(events_map) & set(phase_map))

    samples = {}
    for name in sample_names:
        samples[name] = {
            "hurdle_csv": hurdle_map.get(name),
            "pose_csv": pose_map.get(name),
            "events_json": events_map.get(name),
            "phase_json": phase_map.get(name),
            "segmented_csv": segmented_map.get(name),
            "video": video_map.get(name),
        }

    print("discover_samples 结果：")
    print("  hurdle:", list(hurdle_map.keys()))
    print("  pose:", list(pose_map.keys()))
    print("  events:", list(events_map.keys()))
    print("  phase:", list(phase_map.keys()))
    print("  segmented:", list(segmented_map.keys()))
    print("  video:", list(video_map.keys()))
    print("  matched:", list(samples.keys()))

    return samples


def load_sample_inputs(
    sample_name: str,
    sample_paths: Dict[str, Optional[Path]],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict, Optional[pd.DataFrame], Optional[Path], str, str]:
    hurdle_df = pd.read_csv(sample_paths["hurdle_csv"])
    pose_df = pd.read_csv(sample_paths["pose_csv"])

    with open(sample_paths["events_json"], "r", encoding="utf-8") as f:
        events_json = json.load(f)

    with open(sample_paths["phase_json"], "r", encoding="utf-8") as f:
        phase_json = json.load(f)

    segmented_df = None
    if sample_paths.get("segmented_csv") and Path(sample_paths["segmented_csv"]).exists():
        segmented_df = pd.read_csv(sample_paths["segmented_csv"])

    video_path = sample_paths["video"]

    if video_path is not None:
        source_video_name = Path(video_path).name
        source_video_stem = Path(video_path).stem
    else:
        source_video_name = f"{sample_name}.unknown"
        source_video_stem = sample_name

    return (
        hurdle_df,
        pose_df,
        events_json,
        phase_json,
        segmented_df,
        video_path,
        source_video_name,
        source_video_stem,
    )


def save_outputs(
    all_metrics: List[pd.DataFrame],
    all_events: List[pd.DataFrame],
    all_frames: List[pd.DataFrame],
    output_dirs: Dict[str, Path],
) -> None:
    # 汇总文件放 summary/
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            output_dirs["summary"] / "technical_metrics_all.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(
            output_dirs["summary"] / "event_frames_all.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_csv(
            output_dirs["summary"] / "frame_features_all.csv",
            index=False,
            encoding="utf-8-sig",
        )