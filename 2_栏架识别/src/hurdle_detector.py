# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from hurdle_postprocess import extract_bar_and_post, polygon_to_mask, select_center_instance
from hurdle_visualize import draw_hurdle_result

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}


class HurdleDetector:
    def __init__(self, model_path, conf: float = 0.25, imgsz: int = 960, device=None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f'未找到模型文件: {self.model_path}')
        self.model = YOLO(str(self.model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def iter_videos(self, input_dir):
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f'未找到输入目录: {input_dir}')
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p

    def process_directory(self, input_dir, output_csv_dir, output_video_dir):
        output_csv_dir = Path(output_csv_dir)
        output_video_dir = Path(output_video_dir)
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        output_video_dir.mkdir(parents=True, exist_ok=True)
        videos = list(self.iter_videos(input_dir))
        if not videos:
            raise FileNotFoundError(f'目录中未找到视频文件: {input_dir}')
        for video_path in videos:
            print(f'[INFO] 处理视频: {video_path.name}')
            self.process_video(video_path, output_csv_dir, output_video_dir)

    def process_video(self, video_path, output_csv_dir, output_video_dir):
        video_path = Path(video_path)
        output_csv_dir = Path(output_csv_dir)
        output_video_dir = Path(output_video_dir)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'无法打开视频: {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video_path = output_video_dir / f'{video_path.stem}_vis.mp4'
        writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        if not writer.isOpened():
            raise RuntimeError(f'无法创建输出视频: {out_video_path}')
        frame_rows = []
        try:
            results = self.model.predict(
                source=str(video_path),
                stream=True,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                save=False,
                show=False
            )
            for frame_idx, result in enumerate(results):
                frame = result.orig_img
                row = self._process_single_result(
                    result, frame_idx, fps, frame.shape[1], frame.shape[:2]
                )
                frame_rows.append(row)

                bbox = None if np.isnan(row['bbox_x1']) else (
                    row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
                )
                bar_mid = None if np.isnan(row['bar_mid_x']) else (
                    row['bar_mid_x'], row['bar_mid_y']
                )
                post_x = None if np.isnan(row['post_x']) else row['post_x']
                conf = None if np.isnan(row['conf']) else row['conf']

                polygon = None
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                    idx = select_center_instance(boxes_xyxy, frame.shape[1])
                    if (
                            idx is not None
                            and result.masks is not None
                            and getattr(result.masks, 'xy', None) is not None
                            and len(result.masks.xy) > idx
                    ):
                        polygon = np.asarray(result.masks.xy[idx], dtype=np.float32)

                vis = draw_hurdle_result(frame, bbox, bar_mid, post_x, conf, polygon=polygon)
                writer.write(vis)
        finally:
            cap.release()
            writer.release()
        pd.DataFrame(frame_rows).to_csv(output_csv_dir / f'{video_path.stem}.csv', index=False, encoding='utf-8-sig')
        print(f'[OK] CSV: {output_csv_dir / f"{video_path.stem}.csv"}')
        print(f'[OK] 视频: {out_video_path}')

    def _process_single_result(self, result, frame_idx, fps, frame_width, image_shape_hw):
        base = {
            'video_id': Path(result.path).stem if getattr(result, 'path', None) else 'unknown',
            'frame_idx': frame_idx,
            'time_sec': frame_idx / fps,
            'bbox_x1': np.nan, 'bbox_y1': np.nan, 'bbox_x2': np.nan, 'bbox_y2': np.nan,
            'bar_mid_x': np.nan, 'bar_mid_y': np.nan, 'post_x': np.nan, 'conf': np.nan,
        }
        if result.boxes is None or len(result.boxes) == 0:
            return base
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.full((len(boxes_xyxy),), np.nan)
        idx = select_center_instance(boxes_xyxy, frame_width)
        if idx is None:
            return base
        x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
        base.update({'bbox_x1': float(x1), 'bbox_y1': float(y1), 'bbox_x2': float(x2), 'bbox_y2': float(y2), 'conf': float(confs[idx])})
        if result.masks is not None and getattr(result.masks, 'xy', None) is not None and len(result.masks.xy) > idx:
            polygon = np.asarray(result.masks.xy[idx], dtype=np.float32)
            mask = polygon_to_mask(polygon, image_shape_hw)
            bar_mid_x, _, post_x = extract_bar_and_post(mask)
            base.update({'bar_mid_x': float(bar_mid_x),'bar_mid_y': float(y1),'post_x':float(bar_mid_x)})
        else:
            base.update({'bar_mid_x': float((x1 + x2) / 2.0), 'bar_mid_y': float(y1), 'post_x': float((x1 + x2) / 2.0)})
        return base
