from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from detect_events import run_event_detection
from io_utils import align_hurdle_and_pose, auto_pair_csvs, export_json, load_hurdle_csv, load_pose_csv
from segment_phases import attach_phase_labels, build_phase_segments
from utils import ensure_dir, find_video_by_stem, load_yaml
from visualize import render_event_montage, render_phase_overlay, save_debug_plot

DEFAULTS = {
    'module1_video_dir': '../1_视频预处理/output',
    'module2_csv_dir': '../2_栏架识别/outputs/csv',
    'module3_csv_dir': '../3_人体关键点_RTMPOSE/output/csv',
    'output_dir': 'output',
}

def resolve_path(module_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (module_root / p).resolve()



def process_one(hurdle_csv: Path, pose_csv: Path, video_path: Optional[Path], output_dir: Path, cfg: Dict, video_stem: str) -> Dict:
    hurdle_df = load_hurdle_csv(hurdle_csv)
    pose_df = load_pose_csv(pose_csv)
    aligned = align_hurdle_and_pose(hurdle_df, pose_df)

    csv_dir = ensure_dir(output_dir / 'csv')
    json_dir = ensure_dir(output_dir / 'json')
    video_dir = ensure_dir(output_dir / 'video')
    vis_dir = ensure_dir(output_dir / 'visualization')

    feature_df, meta = run_event_detection(aligned, cfg)
    segments_payload = build_phase_segments(feature_df, meta, video_stem)
    segmented = attach_phase_labels(feature_df, meta, drop_non_phase_frames=bool(cfg.get('export', {}).get('drop_non_phase_frames', True)))

    segmented.to_csv(csv_dir / f'{video_stem}_segmented_coordinates.csv', index=False, encoding='utf-8-sig')
    export_json(segments_payload, json_dir / f'{video_stem}_phase_segments.json')

    save_debug_plot(feature_df, meta, output_dir / 'json' / f'{video_stem}_phase_debug_plot.png')
    export_json(meta, json_dir / f'{video_stem}_events.json')

    overlay_path = None
    montage_path = None
    if video_path is not None:
        overlay_path = render_phase_overlay(video_path, segmented, meta, video_dir / f'{video_stem}_phase_overlay.mp4')
        montage_path = render_event_montage(video_path, segmented, meta, vis_dir / f'{video_stem}_event_montage.png', video_stem)

    return {
        'video_stem': video_stem,
        'hurdle_csv': str(hurdle_csv),
        'pose_csv': str(pose_csv),
        'video_path': str(video_path) if video_path else None,
        'segmented_csv': str(csv_dir / f'{video_stem}_segmented_coordinates.csv'),
        'phase_segments_json': str(json_dir / f'{video_stem}_phase_segments.json'),
        'phase_overlay_video': str(overlay_path) if overlay_path else None,
        'event_montage_image': str(montage_path) if montage_path else None,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='模块4：阶段划分')
    parser.add_argument('--config', default='configs/phase_rules.yaml')
    parser.add_argument('--module1_video_dir', default=DEFAULTS['module1_video_dir'])
    parser.add_argument('--module2_csv_dir', default=DEFAULTS['module2_csv_dir'])
    parser.add_argument('--module3_csv_dir', default=DEFAULTS['module3_csv_dir'])
    parser.add_argument('--output_dir', default=DEFAULTS['output_dir'])
    parser.add_argument('--video_stem', default=None)
    parser.add_argument('--hurdle_csv', default=None)
    parser.add_argument('--pose_csv', default=None)
    parser.add_argument('--video_path', default=None)
    parser.add_argument('--no_render_video', action='store_true')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    module_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(resolve_path(module_root, args.config))
    output_dir = ensure_dir(resolve_path(module_root, args.output_dir))
    summary: List[Dict] = []

    if args.hurdle_csv and args.pose_csv:
        hurdle_csv = Path(args.hurdle_csv).resolve()
        pose_csv = Path(args.pose_csv).resolve()
        video_stem = args.video_stem or hurdle_csv.stem
        video_path = None if args.no_render_video else (Path(args.video_path).resolve() if args.video_path else None)
        summary.append(process_one(hurdle_csv, pose_csv, video_path, output_dir, cfg, video_stem))
    else:
        module1_video_dir = resolve_path(module_root, args.module1_video_dir)
        module2_csv_dir = resolve_path(module_root, args.module2_csv_dir)
        module3_csv_dir = resolve_path(module_root, args.module3_csv_dir)
        pairs = auto_pair_csvs(module2_csv_dir, module3_csv_dir)
        if args.video_stem:
            pairs = [x for x in pairs if x[2] == args.video_stem]
        if not pairs:
            raise ValueError('没有找到可配对的 CSV 文件。')
        for hurdle_csv, pose_csv, stem in pairs:
            video_path = None if args.no_render_video else find_video_by_stem(module1_video_dir, stem)
            summary.append(process_one(hurdle_csv, pose_csv, video_path, output_dir, cfg, stem))

    export_json({'results': summary}, output_dir / 'json' / 'batch_summary.json')
    print(f'处理完成，共 {len(summary)} 个视频。')
    for item in summary:
        print(item['video_stem'])


if __name__ == '__main__':
    main()
