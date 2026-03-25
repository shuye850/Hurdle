# src/io_video.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}


def ensure_dir(path: str | Path) -> Path:
    """
    Args:
        path: 目录路径
    Returns:
        Path: 规范化后的 Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root(current_file: str | Path) -> Path:
    """
    根据当前脚本路径，向上回溯到项目根目录。
    文件位于：3_人体关键点_RTMPOSE/src/io_video.py
    项目根目录即其上两级目录的父目录。
    Args:
        current_file: 当前脚本文件路径，一般传 __file__
    Returns:
        Path: 项目根目录
    """
    return Path(current_file).resolve().parents[2]


def get_module1_output_dir(current_file: str | Path) -> Path:
    """
    获取模块一的视频输出目录路径。

    约定路径：
        项目根目录 / 1_视频预处理 / output

    Args:
        current_file: 当前脚本文件路径，一般传 __file__

    Returns:
        Path: 模块一 output 目录路径
    """
    project_root = get_project_root(current_file)
    return project_root / "1_视频预处理" / "output"


def is_video_file(file_path: str | Path) -> bool:
    """
    判断一个文件是否为常见视频文件。

    Args:
        file_path: 文件路径

    Returns:
        bool: 是否为视频文件
    """
    return Path(file_path).suffix in VIDEO_SUFFIXES


def list_video_files(video_dir: str | Path) -> list[Path]:
    """
    列出目录下全部视频文件，按文件名排序返回。

    Args:
        video_dir: 视频目录

    Returns:
        list[Path]: 视频文件路径列表

    Raises:
        FileNotFoundError: 目录不存在
        NotADirectoryError: 路径不是目录
    """
    video_dir = Path(video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"给定路径不是目录: {video_dir}")

    video_files = [p for p in video_dir.iterdir() if p.is_file() and is_video_file(p)]
    video_files.sort(key=lambda x: x.name)

    return video_files


def check_video_exists(video_path: str | Path) -> Path:
    """
    检查输入视频是否存在。

    Args:
        video_path: 视频路径

    Returns:
        Path: 规范化后的 Path 对象

    Raises:
        FileNotFoundError: 当视频不存在时抛出
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"输入视频不存在: {p}")
    return p


def open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
    """
    打开视频读取对象。

    Args:
        video_path: 输入视频路径

    Returns:
        cv2.VideoCapture: 视频读取对象

    Raises:
        RuntimeError: 当视频无法打开时抛出
    """
    video_path = check_video_exists(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    return cap


def get_video_meta(video_path: str | Path) -> dict:
    """
    读取视频基础信息。

    Args:
        video_path: 输入视频路径

    Returns:
        dict: 包含 fps、width、height、frame_count、duration_sec
    """
    cap = open_video_capture(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    duration_sec = frame_count / fps if fps > 0 else 0.0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def build_video_writer(
    output_path: str | Path,
    fps: float,
    frame_size: tuple[int, int],
    fourcc_str: str = "mp4v",
) -> cv2.VideoWriter:
    """
    创建视频写入器，用于导出带关键点遮罩的视频。

    Args:
        output_path: 输出视频路径
        fps: 帧率
        frame_size: (width, height)
        fourcc_str: 编码器，默认 mp4v

    Returns:
        cv2.VideoWriter: 视频写入对象

    Raises:
        RuntimeError: 当写入器创建失败时抛出
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频写入器: {output_path}")

    return writer


def frame_id_to_timestamp(frame_id: int, fps: float) -> float:
    """
    将帧号转换为秒级时间戳。

    Args:
        frame_id: 当前帧号，从 0 开始
        fps: 视频帧率

    Returns:
        float: 秒级时间戳
    """
    if fps <= 0:
        return 0.0
    return frame_id / fps


def build_video_output_stem(video_path: str | Path) -> str:
    """
    根据输入视频文件名生成输出名前缀。

    例如：
        demo.mp4 -> demo

    Args:
        video_path: 输入视频路径

    Returns:
        str: 不带后缀的文件名
    """
    return Path(video_path).stem