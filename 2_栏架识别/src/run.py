#-*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from hurdle_detector import HurdleDetector

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'weights' / 'best.pt'
INPUT_DIR = Path(__file__).resolve().parents[2] / '1_视频预处理' / 'output'
OUTPUT_CSV_DIR = ROOT / 'outputs' / 'csv'
OUTPUT_VIDEO_DIR = ROOT / 'outputs' / 'videos'


def main() -> None:
    detector = HurdleDetector(model_path=MODEL_PATH, conf=0.25, imgsz=960, device=None)
    detector.process_directory(INPUT_DIR, OUTPUT_CSV_DIR, OUTPUT_VIDEO_DIR)


if __name__ == '__main__':
    main()
