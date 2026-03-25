# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass(frozen=True)
class PoseFrameResult:
    x2d: np.ndarray
    y2d: np.ndarray
    c2d: np.ndarray
    x3d: np.ndarray
    y3d: np.ndarray
    z3d: np.ndarray
    c3d: np.ndarray
    ok: bool
    quality: float


class PoseEstimatorTasks:
    def __init__(
        self,
        model_path: str | Path,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_preprocess: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        gamma: float = 1.0,
    ) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 .task 模型文件：{model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self.enable_preprocess = bool(enable_preprocess)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = int(clahe_grid_size)
        self.gamma = float(gamma)
        self._last_timestamp_ms = -1

    def __enter__(self) -> "PoseEstimatorTasks":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _empty_result() -> PoseFrameResult:
        nan33 = np.full((33,), np.nan, dtype=np.float32)
        zero33 = np.zeros((33,), dtype=np.float32)
        return PoseFrameResult(
            x2d=nan33.copy(),
            y2d=nan33.copy(),
            c2d=zero33.copy(),
            x3d=nan33.copy(),
            y3d=nan33.copy(),
            z3d=nan33.copy(),
            c3d=zero33.copy(),
            ok=False,
            quality=0.0,
        )

    def _maybe_preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enable_preprocess:
            return frame_bgr

        out = frame_bgr
        if self.gamma > 0 and abs(self.gamma - 1.0) > 1e-6:
            inv_gamma = 1.0 / self.gamma
            lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            out = cv2.LUT(out, lut)

        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size),
        )
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    def _bgr_to_mp_image(self, frame_bgr: np.ndarray) -> mp.Image:
        frame_bgr = self._maybe_preprocess(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    @staticmethod
    def _conf(lm) -> float:
        return float(getattr(lm, "presence", 0.0))

    def infer_bgr_video(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrameResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return self._empty_result()

        h, w = frame_bgr.shape[:2]
        mp_image = self._bgr_to_mp_image(frame_bgr)
        ts = int(timestamp_ms)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts

        res = self._detector.detect_for_video(mp_image, ts)
        if not res.pose_landmarks:
            return self._empty_result()

        landmarks = res.pose_landmarks[0]
        xy = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        x2d = xy[:, 0] * w
        y2d = xy[:, 1] * h
        c2d = np.array([self._conf(lm) for lm in landmarks], dtype=np.float32)

        if res.pose_world_landmarks:
            wlms = res.pose_world_landmarks[0]
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in wlms], dtype=np.float32)
            x3d = xyz[:, 0]
            y3d = xyz[:, 1]
            z3d = xyz[:, 2]
            c3d = np.array([self._conf(lm) for lm in wlms], dtype=np.float32)
        else:
            nan33 = np.full((33,), np.nan, dtype=np.float32)
            zero33 = np.zeros((33,), dtype=np.float32)
            x3d, y3d, z3d, c3d = nan33, nan33.copy(), nan33.copy(), zero33

        quality = float(np.mean(c2d)) if c2d.size else 0.0
        return PoseFrameResult(
            x2d=x2d,
            y2d=y2d,
            c2d=c2d,
            x3d=x3d,
            y3d=y3d,
            z3d=z3d,
            c3d=c3d,
            ok=True,
            quality=quality,
        )

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass
