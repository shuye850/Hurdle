# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from export_csv import KEYPOINT_NAMES_33, ensure_dir, export_meta_json, export_pose2d_csv, export_pose3d_csv
from io_video import iter_frames, open_video, safe_release
from pose_mediapipe import PoseEstimatorTasks, PoseFrameResult
from postprocess_optional import postprocess_pose, postprocess_pose3d
from roi_optional import ROIBox, compute_roi_from_keypoints, crop_frame_by_roi, map_pose2d_back_to_full_frame, smooth_roi
from visualize_optional import write_overlay_video


DEFAULT_EXTS = [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".flv", ".webm"]


class Module3Runner:
    def __init__(self, module_dir: Path, cfg: dict) -> None:
        self.module_dir = module_dir
        self.cfg = cfg
        self.keypoint_names = KEYPOINT_NAMES_33

    def _resolve_default_video(self) -> Path:
        paths_cfg = self.cfg.get("paths", {})
        default_input_dir = paths_cfg.get("default_input_dir", "../1_视频预处理/clips_one_hurdle")
        clips_dir = (self.module_dir / default_input_dir).resolve()
        if not clips_dir.exists():
            raise FileNotFoundError(f"找不到默认输入目录：{clips_dir}")

        exts = self.cfg.get("video", {}).get("allowed_exts", DEFAULT_EXTS)
        for ext in exts:
            p = next(clips_dir.glob(f"*{ext}"), None)
            if p is not None:
                return p.resolve()
        raise RuntimeError(f"在目录中未找到可用视频：{clips_dir}")

    def _resolve_model_path(self) -> Path:
        paths_cfg = self.cfg.get("paths", {})
        model_rel = paths_cfg.get("model_path", "../3_人体关键点/models/pose_landmarker_heavy.task")
        return (self.module_dir / model_rel).resolve()

    def _infer_full_frame(self, estimator: PoseEstimatorTasks, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrameResult:
        return estimator.infer_bgr_video(frame_bgr, timestamp_ms=timestamp_ms)

    def _next_roi_from_result(self, result: PoseFrameResult, frame_width: int, frame_height: int, prev_roi: ROIBox | None) -> ROIBox | None:
        roi_cfg = self.cfg.get("roi", {})
        if not bool(roi_cfg.get("enabled", False)):
            return None
        new_roi = compute_roi_from_keypoints(
            x2d=result.x2d,
            y2d=result.y2d,
            conf=result.c2d,
            frame_width=frame_width,
            frame_height=frame_height,
            conf_threshold=float(roi_cfg.get("conf_threshold", 0.55)),
            padding_ratio=float(roi_cfg.get("padding_ratio", 0.15)),
            min_valid_points=int(roi_cfg.get("min_valid_points", 8)),
            min_box_size=int(roi_cfg.get("min_box_size", 96)),
            target_aspect_ratio=roi_cfg.get("target_aspect_ratio", None),
        )
        momentum = float(roi_cfg.get("smoothing_momentum", 0.65))
        return smooth_roi(prev_roi, new_roi, momentum=momentum)

    def _infer_with_optional_roi(
        self,
        estimator: PoseEstimatorTasks,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        frame_index: int,
        prev_roi: ROIBox | None,
    ) -> tuple[PoseFrameResult, ROIBox | None, str]:
        roi_cfg = self.cfg.get("roi", {})
        roi_enabled = bool(roi_cfg.get("enabled", False))
        if not roi_enabled:
            result = self._infer_full_frame(estimator, frame_bgr, timestamp_ms)
            next_roi = self._next_roi_from_result(result, frame_bgr.shape[1], frame_bgr.shape[0], prev_roi)
            return result, next_roi, "full"

        refresh_interval = int(roi_cfg.get("refresh_interval", 12))
        force_full = prev_roi is None or frame_index % max(1, refresh_interval) == 0

        if force_full:
            result = self._infer_full_frame(estimator, frame_bgr, timestamp_ms)
            next_roi = self._next_roi_from_result(result, frame_bgr.shape[1], frame_bgr.shape[0], prev_roi)
            return result, next_roi, "full"

        crop = crop_frame_by_roi(frame_bgr, prev_roi)
        roi_result = estimator.infer_bgr_video(crop, timestamp_ms=timestamp_ms)

        if roi_result.ok:
            x_full, y_full = map_pose2d_back_to_full_frame(roi_result.x2d, roi_result.y2d, prev_roi)
            mapped_result = PoseFrameResult(
                x2d=x_full,
                y2d=y_full,
                c2d=roi_result.c2d,
                x3d=roi_result.x3d,
                y3d=roi_result.y3d,
                z3d=roi_result.z3d,
                c3d=roi_result.c3d,
                ok=roi_result.ok,
                quality=roi_result.quality,
            )
            next_roi = self._next_roi_from_result(mapped_result, frame_bgr.shape[1], frame_bgr.shape[0], prev_roi)
            min_quality = float(roi_cfg.get("min_quality", 0.35))
            if next_roi is not None and mapped_result.quality >= min_quality:
                return mapped_result, next_roi, "roi"

        result = self._infer_full_frame(estimator, frame_bgr, timestamp_ms)
        next_roi = self._next_roi_from_result(result, frame_bgr.shape[1], frame_bgr.shape[0], prev_roi)
        return result, next_roi, "fallback_full"

    def run(self, video_path: Path) -> None:
        model_path = self._resolve_model_path()
        video_cfg = self.cfg.get("video", {})
        max_long_edge = int(video_cfg.get("max_long_edge", 0) or 0)

        task_cfg = self.cfg.get("mediapipe_tasks", {})
        num_poses = int(task_cfg.get("num_poses", 1))
        min_det = float(task_cfg.get("min_pose_detection_confidence", 0.5))
        min_pres = float(task_cfg.get("min_pose_presence_confidence", 0.5))
        min_track = float(task_cfg.get("min_tracking_confidence", 0.5))

        pre_cfg = self.cfg.get("preprocess", {})
        enable_pre = bool(pre_cfg.get("enabled", False))
        clahe_clip_limit = float(pre_cfg.get("clahe_clip_limit", 2.0))
        clahe_grid_size = int(pre_cfg.get("clahe_grid_size", 8))
        gamma = float(pre_cfg.get("gamma", 1.0))

        cap, meta = open_video(video_path, max_long_edge=max_long_edge)
        fps = meta.fps if meta.fps > 0 else 30.0
        video_id = meta.video_path.stem

        out_raw_dir = self.module_dir / "outputs" / "pose_raw"
        out_clean_dir = self.module_dir / "outputs" / "pose_clean"
        out_vis_dir = self.module_dir / "outputs" / "vis_overlay"
        ensure_dir(out_raw_dir)
        ensure_dir(out_clean_dir)
        ensure_dir(out_vis_dir)

        pose2d_csv = out_raw_dir / f"{video_id}_pose2d.csv"
        pose3d_csv = out_raw_dir / f"{video_id}_pose3d.csv"
        meta_json = out_raw_dir / f"{video_id}_meta.json"
        overlay_raw_path = out_vis_dir / f"{video_id}_overlay_raw.mp4"
        overlay_smooth_path = out_vis_dir / f"{video_id}_overlay_smooth.mp4"
        pose2d_clean_csv = out_clean_dir / f"{video_id}_pose2d_clean.csv"
        pose3d_clean_csv = out_clean_dir / f"{video_id}_pose3d_clean.csv"

        x2d_list = []
        y2d_list = []
        c2d_list = []
        x3d_list = []
        y3d_list = []
        z3d_list = []
        c3d_list = []
        quality_list = []
        mode_list = []
        ok_count = 0
        prev_roi = None

        try:
            with PoseEstimatorTasks(
                model_path=model_path,
                num_poses=num_poses,
                min_pose_detection_confidence=min_det,
                min_pose_presence_confidence=min_pres,
                min_tracking_confidence=min_track,
                enable_preprocess=enable_pre,
                clahe_clip_limit=clahe_clip_limit,
                clahe_grid_size=clahe_grid_size,
                gamma=gamma,
            ) as estimator:
                for frame_index, timestamp_ms, frame_bgr in iter_frames(cap, used_scale=meta.used_scale):
                    result, prev_roi, infer_mode = self._infer_with_optional_roi(
                        estimator=estimator,
                        frame_bgr=frame_bgr,
                        timestamp_ms=timestamp_ms,
                        frame_index=frame_index,
                        prev_roi=prev_roi,
                    )
                    x2d_list.append(result.x2d.astype(np.float32))
                    y2d_list.append(result.y2d.astype(np.float32))
                    c2d_list.append(result.c2d.astype(np.float32))
                    x3d_list.append(result.x3d.astype(np.float32))
                    y3d_list.append(result.y3d.astype(np.float32))
                    z3d_list.append(result.z3d.astype(np.float32))
                    c3d_list.append(result.c3d.astype(np.float32))
                    quality_list.append(float(result.quality))
                    mode_list.append(infer_mode)
                    ok_count += int(result.ok)
        finally:
            safe_release(cap)

        x2d_all = np.stack(x2d_list, axis=0) if x2d_list else np.zeros((0, 33), dtype=np.float32)
        y2d_all = np.stack(y2d_list, axis=0) if y2d_list else np.zeros((0, 33), dtype=np.float32)
        c2d_all = np.stack(c2d_list, axis=0) if c2d_list else np.zeros((0, 33), dtype=np.float32)
        x3d_all = np.stack(x3d_list, axis=0) if x3d_list else np.zeros((0, 33), dtype=np.float32)
        y3d_all = np.stack(y3d_list, axis=0) if y3d_list else np.zeros((0, 33), dtype=np.float32)
        z3d_all = np.stack(z3d_list, axis=0) if z3d_list else np.zeros((0, 33), dtype=np.float32)
        c3d_all = np.stack(c3d_list, axis=0) if c3d_list else np.zeros((0, 33), dtype=np.float32)
        quality_all = np.asarray(quality_list, dtype=np.float32) if quality_list else np.zeros((0,), dtype=np.float32)

        exports_cfg = self.cfg.get("exports", {})
        if bool(exports_cfg.get("write_raw_csv", True)):
            export_pose2d_csv(pose2d_csv, fps, x2d_all, y2d_all, c2d_all, self.keypoint_names)
            export_pose3d_csv(pose3d_csv, fps, x3d_all, y3d_all, z3d_all, c3d_all, self.keypoint_names)

        pp_cfg = self.cfg.get("postprocess", {})
        pp_enabled = bool(pp_cfg.get("enabled", True))
        x2d_clean = x2d_all
        y2d_clean = y2d_all
        c2d_clean = c2d_all
        x3d_clean = x3d_all
        y3d_clean = y3d_all
        z3d_clean = z3d_all
        c3d_clean = c3d_all

        if pp_enabled and x2d_all.shape[0] > 0:
            conf_th = float(pp_cfg.get("conf_threshold", 0.6))
            max_gap = int(pp_cfg.get("max_interp_gap", 5))
            win = int(pp_cfg.get("savgol_window", 9))
            poly = int(pp_cfg.get("savgol_polyorder", 2))
            enable_support_foot_constraint = bool(pp_cfg.get("enable_support_foot_constraint", True))

            x2d_clean, y2d_clean, c2d_clean = postprocess_pose(
                x_all=x2d_all,
                y_all=y2d_all,
                conf_all=c2d_all,
                conf_th=conf_th,
                max_interp_gap=max_gap,
                savgol_window=win,
                savgol_polyorder=poly,
                enable_support_foot_constraint=enable_support_foot_constraint,
            )
            x3d_clean, y3d_clean, z3d_clean, c3d_clean = postprocess_pose3d(
                x3d_all=x3d_all,
                y3d_all=y3d_all,
                z3d_all=z3d_all,
                conf_all=c3d_all,
                conf_th=conf_th,
                max_interp_gap=max_gap,
                savgol_window=win,
                savgol_polyorder=poly,
            )
            if bool(exports_cfg.get("write_clean_csv", True)):
                export_pose2d_csv(pose2d_clean_csv, fps, x2d_clean, y2d_clean, c2d_clean, self.keypoint_names)
                export_pose3d_csv(pose3d_clean_csv, fps, x3d_clean, y3d_clean, z3d_clean, c3d_clean, self.keypoint_names)
        else:
            pose2d_clean_csv = None
            pose3d_clean_csv = None

        overlay_cfg = self.cfg.get("overlay", {})
        overlay_enabled = bool(overlay_cfg.get("enabled", True))
        draw_conf_th = float(overlay_cfg.get("draw_conf_threshold", 0.6))
        draw_frame_index = bool(overlay_cfg.get("draw_frame_index", True))
        draw_quality = bool(overlay_cfg.get("draw_quality", True))
        draw_infer_mode = bool(overlay_cfg.get("draw_infer_mode", True))

        if overlay_enabled and x2d_all.shape[0] > 0:
            if bool(overlay_cfg.get("write_raw", True)):
                write_overlay_video(
                    input_video_path=meta.video_path,
                    output_video_path=overlay_raw_path,
                    fps=fps,
                    used_scale=meta.used_scale,
                    x2d_all=x2d_all,
                    y2d_all=y2d_all,
                    c2d_all=c2d_all,
                    conf_th=draw_conf_th,
                    quality_all=quality_all,
                    infer_modes=mode_list,
                    draw_frame_index=draw_frame_index,
                    draw_quality=draw_quality,
                    draw_infer_mode=draw_infer_mode,
                )
            else:
                overlay_raw_path = None

            if bool(overlay_cfg.get("write_smooth", True)) and pp_enabled:
                write_overlay_video(
                    input_video_path=meta.video_path,
                    output_video_path=overlay_smooth_path,
                    fps=fps,
                    used_scale=meta.used_scale,
                    x2d_all=x2d_clean,
                    y2d_all=y2d_clean,
                    c2d_all=c2d_clean,
                    conf_th=draw_conf_th,
                    quality_all=quality_all,
                    infer_modes=mode_list,
                    draw_frame_index=draw_frame_index,
                    draw_quality=draw_quality,
                    draw_infer_mode=draw_infer_mode,
                )
            else:
                overlay_smooth_path = None
        else:
            overlay_raw_path = None
            overlay_smooth_path = None

        if bool(exports_cfg.get("write_meta_json", True)):
            meta_out = {
                "video_id": video_id,
                "input_video_path": str(meta.video_path),
                "fps": float(fps),
                "frame_count_declared": int(meta.frame_count),
                "frame_count_read": int(x2d_all.shape[0]),
                "width": int(meta.width),
                "height": int(meta.height),
                "out_width": int(meta.out_width),
                "out_height": int(meta.out_height),
                "max_long_edge": int(meta.max_long_edge),
                "used_scale": float(meta.used_scale),
                "mediapipe_tasks": {
                    "model_path": str(model_path),
                    "num_poses": num_poses,
                    "min_pose_detection_confidence": min_det,
                    "min_pose_presence_confidence": min_pres,
                    "min_tracking_confidence": min_track,
                },
                "preprocess": self.cfg.get("preprocess", {}),
                "roi": self.cfg.get("roi", {}),
                "postprocess": self.cfg.get("postprocess", {}),
                "overlay": self.cfg.get("overlay", {}),
                "export": {
                    "pose2d_csv": str(pose2d_csv) if pose2d_csv.exists() else None,
                    "pose3d_csv": str(pose3d_csv) if pose3d_csv.exists() else None,
                    "pose2d_clean_csv": str(pose2d_clean_csv) if pose2d_clean_csv is not None and pose2d_clean_csv.exists() else None,
                    "pose3d_clean_csv": str(pose3d_clean_csv) if pose3d_clean_csv is not None and pose3d_clean_csv.exists() else None,
                    "overlay_raw_mp4": str(overlay_raw_path) if overlay_raw_path is not None and overlay_raw_path.exists() else None,
                    "overlay_smooth_mp4": str(overlay_smooth_path) if overlay_smooth_path is not None and overlay_smooth_path.exists() else None,
                },
                "quality": {
                    "ok_frames": int(ok_count),
                    "total_frames": int(x2d_all.shape[0]),
                    "mean_quality": float(np.mean(quality_all)) if quality_all.size else 0.0,
                },
                "stats": {
                    "roi_frames": int(sum(m == "roi" for m in mode_list)),
                    "full_frames": int(sum(m == "full" for m in mode_list)),
                    "fallback_full_frames": int(sum(m == "fallback_full" for m in mode_list)),
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "versions": {"python": sys.version},
            }
            export_meta_json(meta_json, meta_out)

        print("✅ 模块3运行完成")
        print("pose2d_raw:", pose2d_csv)
        print("pose3d_raw:", pose3d_csv)
        print("pose2d_clean:", pose2d_clean_csv)
        print("pose3d_clean:", pose3d_clean_csv)
        print("overlay_raw:", overlay_raw_path)
        print("overlay_smooth:", overlay_smooth_path)
        print("meta:", meta_json)


def load_config(module_dir: Path) -> dict:
    cfg_path = module_dir / "config" / "pose.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def main() -> None:
    module_dir = Path(__file__).resolve().parents[1]
    cfg = load_config(module_dir)

    if len(sys.argv) >= 2:
        video_path = Path(sys.argv[1]).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在：{video_path}")
    else:
        runner_temp = Module3Runner(module_dir=module_dir, cfg=cfg)
        video_path = runner_temp._resolve_default_video()

    runner = Module3Runner(module_dir=module_dir, cfg=cfg)
    runner.run(video_path=video_path)


if __name__ == "__main__":
    main()
