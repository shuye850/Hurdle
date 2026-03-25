# src/visualize_pose.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from io_video import (
    open_video_capture,
    build_video_writer,
    ensure_dir,
)
from model import get_skeleton


def draw_bbox(
    frame,
    bbox_xyxy: list[float],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
):
    """
    在图像上绘制人体框。

    Args:
        frame: 输入帧
        bbox_xyxy: [x1, y1, x2, y2]
        color: 框颜色
        thickness: 线宽

    Returns:
        frame: 绘制后的帧
    """
    if not bbox_xyxy or len(bbox_xyxy) < 4:
        return frame

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]

    if x2 <= x1 or y2 <= y1:
        return frame

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_keypoints(
    frame,
    keypoints: list[list[float]],
    keypoint_scores: list[float],
    score_thr: float = 0.3,
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 3,
):
    """
    在图像上绘制关键点。

    Args:
        frame: 输入帧
        keypoints: [[x, y], ...]
        keypoint_scores: [score1, score2, ...]
        score_thr: 置信度阈值
        color: 点颜色
        radius: 点半径

    Returns:
        frame: 绘制后的帧
    """
    for idx, point in enumerate(keypoints):
        if idx >= len(keypoint_scores):
            continue
        if keypoint_scores[idx] < score_thr:
            continue
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), radius, color, -1)

    return frame


def draw_skeleton(
    frame,
    keypoints: list[list[float]],
    keypoint_scores: list[float],
    skeleton: list[tuple[int, int]],
    score_thr: float = 0.3,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
):
    """
    在图像上绘制骨架连线。

    Args:
        frame: 输入帧
        keypoints: [[x, y], ...]
        keypoint_scores: [score1, score2, ...]
        skeleton: 骨架连接定义
        score_thr: 置信度阈值
        color: 连线颜色
        thickness: 线宽

    Returns:
        frame: 绘制后的帧
    """
    for i, j in skeleton:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if i >= len(keypoint_scores) or j >= len(keypoint_scores):
            continue
        if keypoint_scores[i] < score_thr or keypoint_scores[j] < score_thr:
            continue

        p1 = keypoints[i]
        p2 = keypoints[j]

        if not isinstance(p1, (list, tuple)) or len(p1) < 2:
            continue
        if not isinstance(p2, (list, tuple)) or len(p2) < 2:
            continue

        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    return frame


def draw_pose_on_frame(
    frame,
    frame_result: dict[str, Any],
    schema_name: str,
    score_thr: float = 0.3,
    draw_box: bool = True,
):
    """
    将单帧关键点结果绘制到原始视频帧上。

    Args:
        frame: 原始帧
        frame_result: 单帧推理结果
        schema_name: 关键点方案名，如 body17 / wholebody133
        score_thr: 关键点显示阈值
        draw_box: 是否绘制人体框

    Returns:
        frame: 绘制后的帧
    """
    bbox_xyxy = frame_result.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    keypoints = frame_result.get("keypoints", [])
    keypoint_scores = frame_result.get("keypoint_scores", [])

    skeleton = get_skeleton(schema_name)

    canvas = frame.copy()

    if draw_box:
        canvas = draw_bbox(canvas, bbox_xyxy)

    canvas = draw_skeleton(
        canvas,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        skeleton=skeleton,
        score_thr=score_thr,
    )

    canvas = draw_keypoints(
        canvas,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        score_thr=score_thr,
    )

    return canvas


def visualize_single_video(
    video_path: str | Path,
    frame_results: list[dict[str, Any]],
    output_video_path: str | Path,
    schema_name: str,
    score_thr: float = 0.3,
    draw_box: bool = True,
) -> Path:
    """
    将单个视频的关键点结果绘制为可视化视频。

    Args:
        video_path: 原始输入视频路径
        frame_results: 该视频逐帧推理结果
        output_video_path: 输出可视化视频路径
        schema_name: 关键点方案名
        score_thr: 显示阈值
        draw_box: 是否绘制人体框

    Returns:
        Path: 输出视频路径
    """
    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    ensure_dir(output_video_path.parent)

    cap = open_video_capture(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = build_video_writer(
        output_path=output_video_path,
        fps=fps,
        frame_size=(width, height),
    )

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id < len(frame_results):
            vis_frame = draw_pose_on_frame(
                frame=frame,
                frame_result=frame_results[frame_id],
                schema_name=schema_name,
                score_thr=score_thr,
                draw_box=draw_box,
            )
        else:
            vis_frame = frame

        writer.write(vis_frame)
        frame_id += 1

    cap.release()
    writer.release()

    return output_video_path


def visualize_multiple_videos(
    video_paths: list[str | Path],
    video_results_map: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    schema_name: str,
    score_thr: float = 0.3,
    draw_box: bool = True,
) -> list[Path]:
    """
    批量生成多个视频的关键点可视化视频。

    Args:
        video_paths: 原始视频路径列表
        video_results_map: 逐视频结果映射，key 为视频 stem
        output_dir: 输出目录
        schema_name: 关键点方案名
        score_thr: 显示阈值
        draw_box: 是否绘制人体框

    Returns:
        list[Path]: 所有生成的视频路径
    """
    output_dir = ensure_dir(output_dir)
    output_paths: list[Path] = []

    for video_path in video_paths:
        video_path = Path(video_path)
        video_stem = video_path.stem

        if video_stem not in video_results_map:
            continue

        output_video_path = output_dir / f"{video_stem}_vis.mp4"

        saved_path = visualize_single_video(
            video_path=video_path,
            frame_results=video_results_map[video_stem],
            output_video_path=output_video_path,
            schema_name=schema_name,
            score_thr=score_thr,
            draw_box=draw_box,
        )
        output_paths.append(saved_path)

    return output_paths