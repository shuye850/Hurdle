from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


HURDLE_REQUIRED = [
    'video_id', 'frame_idx', 'time_sec',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'bar_mid_x', 'bar_mid_y', 'post_x', 'conf'
]

POSE_REQUIRED = [
    'frame_id', 'timestamp_sec', 'fps', 'width', 'height', 'person_index',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_score',
    'left_hip_x', 'left_hip_y', 'left_hip_score',
    'right_hip_x', 'right_hip_y', 'right_hip_score',
    'left_knee_x', 'left_knee_y', 'left_knee_score',
    'right_knee_x', 'right_knee_y', 'right_knee_score',
    'left_ankle_x', 'left_ankle_y', 'left_ankle_score',
    'right_ankle_x', 'right_ankle_y', 'right_ankle_score',
]


COCO17_ORDER = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]


def _check_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{name} 缺少字段: {missing}')



def load_hurdle_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, HURDLE_REQUIRED, '栏架CSV')
    return df.copy()



def load_pose_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, POSE_REQUIRED, '人体关键点CSV')
    return df.copy()



def infer_pose_stem(path: Path) -> str:
    name = path.stem
    if name.endswith('_keypoints'):
        return name[:-10]
    return name



def auto_pair_csvs(module2_csv_dir: str | Path, module3_csv_dir: str | Path) -> List[Tuple[Path, Path, str]]:
    m2 = Path(module2_csv_dir)
    m3 = Path(module3_csv_dir)
    hurdle_map = {p.stem: p for p in sorted(m2.glob('*.csv'))}
    pose_map = {infer_pose_stem(p): p for p in sorted(m3.glob('*.csv'))}
    common = sorted(set(hurdle_map) & set(pose_map))
    return [(hurdle_map[stem], pose_map[stem], stem) for stem in common]



def align_hurdle_and_pose(hurdle_df: pd.DataFrame, pose_df: pd.DataFrame) -> pd.DataFrame:
    hurdle = hurdle_df.rename(columns={'frame_idx': 'frame_index'})
    pose = pose_df.rename(columns={'frame_id': 'frame_index', 'timestamp_sec': 'time_sec_pose'})
    merged = pd.merge(pose, hurdle, on='frame_index', how='inner', suffixes=('_pose', '_hurdle'))
    if merged.empty:
        raise ValueError('按 frame_index 对齐后为空，请检查两个CSV是否来自同一视频。')

    merged['time_sec'] = merged['time_sec']
    merged['time_sec_delta'] = (merged['time_sec_pose'] - merged['time_sec']).abs()

    ordered_cols = [
        'frame_index', 'time_sec', 'time_sec_pose', 'time_sec_delta',
        'fps', 'width', 'height', 'person_index', 'video_id',
        'bbox_x1_pose', 'bbox_y1_pose', 'bbox_x2_pose', 'bbox_y2_pose', 'bbox_score',
    ]
    for kp in COCO17_ORDER:
        ordered_cols.extend([f'{kp}_x', f'{kp}_y', f'{kp}_score'])
    ordered_cols.extend([
        'bbox_x1_hurdle', 'bbox_y1_hurdle', 'bbox_x2_hurdle', 'bbox_y2_hurdle',
        'bar_mid_x', 'bar_mid_y', 'post_x', 'conf'
    ])
    existing = [c for c in ordered_cols if c in merged.columns]
    remaining = [c for c in merged.columns if c not in existing]
    return merged[existing + remaining].copy()



def export_json(data: Dict, path: str | Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
