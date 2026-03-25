# src/infer_pose.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Mac 兼容设置：尽量避免部分 NMS / MPS 场景出问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
from mmpose.apis import MMPoseInferencer

from io_video import get_video_meta, frame_id_to_timestamp
from model import get_num_keypoints, get_default_pose2d_alias


def build_inferencer(
    schema_name: str,
    device: str = "cpu",
    pose2d: str | None = None,
    pose2d_weights: str | None = None,
    det_model: str = "rtmdet-l",
    det_cat_ids: list[int] | None = None,
) -> MMPoseInferencer:
    """
    构建 MMPose / RTMPose 推理器。
    默认使用检测器先框人，再做 top-down 姿态估计。
    """
    if pose2d is None:
        pose2d = get_default_pose2d_alias(schema_name)

    if det_cat_ids is None:
        det_cat_ids = [0]

    kwargs: dict[str, Any] = {
        "pose2d": pose2d,
        "device": device,
        "det_model": det_model,
        "det_cat_ids": det_cat_ids,
    }

    if pose2d_weights is not None:
        kwargs["pose2d_weights"] = pose2d_weights

    return MMPoseInferencer(**kwargs)


def normalize_bbox(raw_bbox: Any) -> list[float]:
    if raw_bbox is None:
        return [0.0, 0.0, 0.0, 0.0]

    if isinstance(raw_bbox, np.ndarray):
        if raw_bbox.ndim == 1 and raw_bbox.size >= 4:
            return [float(v) for v in raw_bbox[:4].tolist()]
        if raw_bbox.ndim >= 2 and raw_bbox.shape[0] > 0 and raw_bbox.shape[1] >= 4:
            return [float(v) for v in raw_bbox[0, :4].tolist()]
        return [0.0, 0.0, 0.0, 0.0]

    if isinstance(raw_bbox, (list, tuple)):
        if len(raw_bbox) == 4 and not isinstance(raw_bbox[0], (list, tuple, np.ndarray)):
            return [float(v) for v in raw_bbox]
        if len(raw_bbox) > 0 and isinstance(raw_bbox[0], (list, tuple, np.ndarray)):
            return [float(v) for v in list(raw_bbox[0])[:4]]

    return [0.0, 0.0, 0.0, 0.0]


def normalize_keypoints(raw_keypoints: Any, num_keypoints: int) -> list[list[float]]:
    if not isinstance(raw_keypoints, list):
        return [[0.0, 0.0] for _ in range(num_keypoints)]

    output: list[list[float]] = []
    for p in raw_keypoints[:num_keypoints]:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            output.append([float(p[0]), float(p[1])])
        else:
            output.append([0.0, 0.0])

    while len(output) < num_keypoints:
        output.append([0.0, 0.0])

    return output


def normalize_scores(raw_scores: Any, num_keypoints: int) -> list[float]:
    if not isinstance(raw_scores, list):
        return [0.0 for _ in range(num_keypoints)]

    output = [float(s) for s in raw_scores[:num_keypoints]]
    while len(output) < num_keypoints:
        output.append(0.0)

    return output


def bbox_area_xyxy(bbox_xyxy: list[float]) -> float:
    if len(bbox_xyxy) < 4:
        return 0.0
    x1, y1, x2, y2 = bbox_xyxy[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_main_person(pred_instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    在多人检测结果中选主人体。
    当前策略：优先取 bbox 面积最大者。
    对跨栏单人视频通常比“直接取第一个”更稳。
    """
    if not pred_instances:
        return None

    best_person = None
    best_area = -1.0

    for person in pred_instances:
        bbox = normalize_bbox(person.get("bbox"))
        area = bbox_area_xyxy(bbox)
        if area > best_area:
            best_area = area
            best_person = person

    return best_person


def parse_single_frame_result(
    result: dict[str, Any],
    frame_id: int,
    fps: float,
    width: int,
    height: int,
    schema_name: str,
) -> dict[str, Any]:
    num_keypoints = get_num_keypoints(schema_name)

    predictions = result.get("predictions", [])
    pred_instances = predictions[0] if predictions else []
    person = select_main_person(pred_instances)

    if person is None:
        return {
            "frame_id": frame_id,
            "timestamp_sec": frame_id_to_timestamp(frame_id, fps),
            "fps": fps,
            "width": width,
            "height": height,
            "person_index": -1,
            "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
            "bbox_score": 0.0,
            "keypoints": [[0.0, 0.0] for _ in range(num_keypoints)],
            "keypoint_scores": [0.0 for _ in range(num_keypoints)],
        }

    bbox = normalize_bbox(person.get("bbox"))

    bbox_score = person.get("bbox_score", 1.0)
    if isinstance(bbox_score, list):
        bbox_score = float(bbox_score[0]) if bbox_score else 1.0
    else:
        bbox_score = float(bbox_score)

    keypoints = normalize_keypoints(person.get("keypoints"), num_keypoints=num_keypoints)
    keypoint_scores = normalize_scores(person.get("keypoint_scores"), num_keypoints=num_keypoints)

    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id_to_timestamp(frame_id, fps),
        "fps": fps,
        "width": width,
        "height": height,
        "person_index": 0,
        "bbox_xyxy": bbox,
        "bbox_score": bbox_score,
        "keypoints": keypoints,
        "keypoint_scores": keypoint_scores,
    }


def infer_single_video(
    video_path: str | Path,
    schema_name: str = "body17",
    inferencer: MMPoseInferencer | None = None,
    device: str = "cpu",
    pose2d: str | None = None,
    pose2d_weights: str | None = None,
    det_model: str = "rtmdet-l",
    det_cat_ids: list[int] | None = None,
    bbox_thr: float = 0.3,
    kpt_thr: float = 0.4,
) -> list[dict[str, Any]]:
    """
    对单个视频执行逐帧姿态推理，并返回标准化结果列表。
    当 inferencer 已传入时，会复用已初始化模型，避免重复加载权重。
    """
    video_path = str(video_path)
    meta = get_video_meta(video_path)

    fps = meta["fps"]
    width = meta["width"]
    height = meta["height"]

    if inferencer is None:
        inferencer = build_inferencer(
            schema_name=schema_name,
            device=device,
            pose2d=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_cat_ids=det_cat_ids,
        )

    result_generator = inferencer(
        video_path,
        return_vis=False,
        draw_bbox=False,
        draw_heatmap=False,
        bbox_thr=bbox_thr,
        kpt_thr=kpt_thr,
        show=False,
        return_datasamples=False,
    )

    frame_results: list[dict[str, Any]] = []

    for frame_id, result in enumerate(result_generator):
        parsed = parse_single_frame_result(
            result=result,
            frame_id=frame_id,
            fps=fps,
            width=width,
            height=height,
            schema_name=schema_name,
        )
        frame_results.append(parsed)

    return frame_results


def infer_multiple_videos(
    video_paths: list[str | Path],
    schema_name: str = "body17",
    device: str = "cpu",
    pose2d: str | None = None,
    pose2d_weights: str | None = None,
    det_model: str = "rtmdet-l",
    det_cat_ids: list[int] | None = None,
    bbox_thr: float = 0.3,
    kpt_thr: float = 0.4,
) -> dict[str, list[dict[str, Any]]]:
    """
    批量推理多个视频。
    返回:
        {
            "video_stem": [...逐帧结果...],
            ...
        }
    """
    results_map: dict[str, list[dict[str, Any]]] = {}

    inferencer = build_inferencer(
        schema_name=schema_name,
        device=device,
        pose2d=pose2d,
        pose2d_weights=pose2d_weights,
        det_model=det_model,
        det_cat_ids=det_cat_ids,
    )

    for video_path in video_paths:
        video_path = Path(video_path)
        video_stem = video_path.stem

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
        results_map[video_stem] = frame_results

    return results_map
