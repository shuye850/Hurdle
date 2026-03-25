# src/export_csv.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from model import get_keypoint_names
from io_video import ensure_dir


def build_base_fields() -> list[str]:
    """
    构建 CSV 基础字段。

    Returns:
        list[str]: 基础字段名列表
    """
    return [
        "frame_id",
        "timestamp_sec",
        "fps",
        "width",
        "height",
        "person_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_score",
    ]


def build_keypoint_fields(schema_name: str) -> list[str]:
    """
    根据关键点方案构建关键点相关字段。

    每个关键点导出 3 列：
    - xxx_x
    - xxx_y
    - xxx_score

    Args:
        schema_name: 关键点方案名，如 body17 / wholebody133

    Returns:
        list[str]: 关键点字段名列表
    """
    keypoint_names = get_keypoint_names(schema_name)
    fields: list[str] = []

    for name in keypoint_names:
        fields.append(f"{name}_x")
        fields.append(f"{name}_y")
        fields.append(f"{name}_score")

    return fields


def build_csv_fieldnames(schema_name: str) -> list[str]:
    """
    构建完整 CSV 表头字段。

    Args:
        schema_name: 关键点方案名

    Returns:
        list[str]: 完整字段列表
    """
    return build_base_fields() + build_keypoint_fields(schema_name)


def frame_result_to_row(frame_result: dict[str, Any], schema_name: str) -> dict[str, Any]:
    """
    将单帧关键点结果转换为一行 CSV 数据。

    Args:
        frame_result: 单帧结果字典
        schema_name: 关键点方案名

    Returns:
        dict[str, Any]: 可直接写入 csv.DictWriter 的行数据
    """
    row: dict[str, Any] = {}

    bbox = frame_result.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    keypoints = frame_result.get("keypoints", [])
    keypoint_scores = frame_result.get("keypoint_scores", [])

    row["frame_id"] = frame_result.get("frame_id", -1)
    row["timestamp_sec"] = frame_result.get("timestamp_sec", 0.0)
    row["fps"] = frame_result.get("fps", 0.0)
    row["width"] = frame_result.get("width", 0)
    row["height"] = frame_result.get("height", 0)
    row["person_index"] = frame_result.get("person_index", -1)
    row["bbox_x1"] = bbox[0] if len(bbox) > 0 else 0.0
    row["bbox_y1"] = bbox[1] if len(bbox) > 1 else 0.0
    row["bbox_x2"] = bbox[2] if len(bbox) > 2 else 0.0
    row["bbox_y2"] = bbox[3] if len(bbox) > 3 else 0.0
    row["bbox_score"] = frame_result.get("bbox_score", 0.0)

    keypoint_names = get_keypoint_names(schema_name)

    for idx, name in enumerate(keypoint_names):
        if idx < len(keypoints) and isinstance(keypoints[idx], (list, tuple)) and len(keypoints[idx]) >= 2:
            row[f"{name}_x"] = keypoints[idx][0]
            row[f"{name}_y"] = keypoints[idx][1]
        else:
            row[f"{name}_x"] = 0.0
            row[f"{name}_y"] = 0.0

        if idx < len(keypoint_scores):
            row[f"{name}_score"] = keypoint_scores[idx]
        else:
            row[f"{name}_score"] = 0.0

    return row


def export_pose_csv(
    frame_results: list[dict[str, Any]],
    output_csv_path: str | Path,
    schema_name: str,
) -> Path:
    """
    将整段视频的关键点结果导出为 CSV 文件。

    Args:
        frame_results: 整段视频的逐帧关键点结果列表
        output_csv_path: 输出 CSV 文件路径
        schema_name: 关键点方案名，如 body17 / wholebody133

    Returns:
        Path: 输出 CSV 文件路径
    """
    output_csv_path = Path(output_csv_path)
    ensure_dir(output_csv_path.parent)

    fieldnames = build_csv_fieldnames(schema_name)

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame_result in frame_results:
            row = frame_result_to_row(frame_result, schema_name)
            writer.writerow(row)

    return output_csv_path


def export_multiple_pose_csv(
    video_results_map: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    schema_name: str,
) -> list[Path]:
    """
    批量导出多个视频的关键点 CSV。

    Args:
        video_results_map: 多视频结果映射，格式：
            {
                "demo1": [...],
                "demo2": [...],
            }
        output_dir: 输出目录
        schema_name: 关键点方案名

    Returns:
        list[Path]: 生成的 CSV 路径列表
    """
    output_dir = ensure_dir(output_dir)
    output_paths: list[Path] = []

    for video_stem, frame_results in video_results_map.items():
        output_csv_path = output_dir / f"{video_stem}_keypoints.csv"
        saved_path = export_pose_csv(
            frame_results=frame_results,
            output_csv_path=output_csv_path,
            schema_name=schema_name,
        )
        output_paths.append(saved_path)

    return output_paths