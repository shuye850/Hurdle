from __future__ import annotations

from typing import Dict, List

import pandas as pd


PHASES = [
    (1, '起跨', 'takeoff_touchdown', 'takeoff_toeoff'),
    (2, '腾空', 'takeoff_toeoff', 'swing_touchdown'),
    (3, '落地', 'swing_touchdown', 'swing_toeoff'),
]



def _frame_to_row(df: pd.DataFrame, frame_index: int) -> int:
    matches = df.index[df['frame_index'] == frame_index].tolist()
    if not matches:
        raise ValueError(f'找不到 frame_index={frame_index}')
    return int(matches[0])



def build_phase_segments(df: pd.DataFrame, event_meta: Dict, video_stem: str) -> Dict:
    events = event_meta['events']
    segments: List[Dict] = []
    for phase_id, phase_name, start_key, end_key in PHASES:
        start_frame = int(events[start_key])
        end_frame = int(events[end_key])
        start_row = _frame_to_row(df, start_frame)
        end_row = _frame_to_row(df, end_frame)
        segments.append({
            'phase_id': phase_id,
            'phase_name': phase_name,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time_sec': float(df.iloc[start_row]['time_sec']),
            'end_time_sec': float(df.iloc[end_row]['time_sec']),
        })
    return {'video_stem': video_stem, 'segments': segments}



def attach_phase_labels(df: pd.DataFrame, event_meta: Dict, drop_non_phase_frames: bool = True) -> pd.DataFrame:
    out = df.copy()
    out['phase_id'] = 0
    out['phase_name'] = '非跨栏步阶段'
    events = event_meta['events']
    ranges = [
        (1, '起跨', events['takeoff_touchdown'], events['takeoff_toeoff']),
        (2, '腾空', events['takeoff_toeoff'], events['swing_touchdown']),
        (3, '落地', events['swing_touchdown'], events['swing_toeoff']),
    ]
    for phase_id, phase_name, start_f, end_f in ranges:
        mask = (out['frame_index'] >= start_f) & (out['frame_index'] <= end_f)
        out.loc[mask, 'phase_id'] = phase_id
        out.loc[mask, 'phase_name'] = phase_name
    if drop_non_phase_frames:
        out = out[out['phase_id'] > 0].copy()
    leading = ['frame_index', 'time_sec', 'phase_id', 'phase_name']
    others = [c for c in out.columns if c not in leading and not c.endswith('_smooth') and not c.endswith('_vx') and not c.endswith('_vy') and c not in ['pelvis_x', 'pelvis_y', 'pelvis_dx_to_bar', 'left_ankle_dx_to_bar', 'right_ankle_dx_to_bar']]
    return out[leading + others]
