from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频预处理：按配置调整分辨率")
    parser.add_argument("--input", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--output", type=Path, required=True, help="输出视频路径")
    parser.add_argument("--max-width", type=int, default=1280, help="目标最大宽度")
    parser.add_argument("--max-height", type=int, default=720, help="目标最大高度")
    parser.add_argument("--keep-aspect-ratio", action="store_true", help="保持纵横比")
    parser.add_argument("--upscale", action="store_true", help="允许放大到目标尺寸")
    return parser.parse_args()


def compute_target_size(
    src_width: int,
    src_height: int,
    *,
    max_width: int,
    max_height: int,
    keep_aspect_ratio: bool,
    upscale: bool,
) -> tuple[int, int]:
    if src_width <= 0 or src_height <= 0:
        raise ValueError("无法读取源视频分辨率。")

    if not keep_aspect_ratio:
        return _ensure_even_size(max_width, max_height)

    width_scale = max_width / src_width
    height_scale = max_height / src_height
    scale = min(width_scale, height_scale)
    if not upscale:
        scale = min(scale, 1.0)

    target_width = max(2, int(round(src_width * scale)))
    target_height = max(2, int(round(src_height * scale)))
    return _ensure_even_size(target_width, target_height)


def _ensure_even_size(width: int, height: int) -> tuple[int, int]:
    width = max(2, width if width % 2 == 0 else width - 1)
    height = max(2, height if height % 2 == 0 else height - 1)
    return width, height


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开输入视频: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, frame_count
    finally:
        cap.release()


def _validate_output(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"输出视频不可读取: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0:
            raise RuntimeError(f"输出视频帧数异常: {path}")
        return frame_count, width, height
    finally:
        cap.release()


def _resize_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    *,
    target_width: int,
    target_height: int,
    fps: float,
) -> None:
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError("未安装 imageio-ffmpeg，无法使用 ffmpeg 预处理。") from exc

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale={target_width}:{target_height}",
        "-r",
        f"{fps:.6f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg 预处理失败:\n" + completed.stdout[-4000:]
        )


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入视频不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps, src_width, src_height, src_frame_count = _probe_video(input_path)
    target_width, target_height = compute_target_size(
        src_width,
        src_height,
        max_width=args.max_width,
        max_height=args.max_height,
        keep_aspect_ratio=args.keep_aspect_ratio,
        upscale=args.upscale,
    )

    _resize_with_ffmpeg(
        input_path,
        output_path,
        target_width=target_width,
        target_height=target_height,
        fps=fps,
    )
    out_frame_count, out_width, out_height = _validate_output(output_path)

    print(
        f"[预处理完成] {input_path.name} -> {output_path.name} | "
        f"{src_width}x{src_height} -> {out_width}x{out_height} | "
        f"frames={out_frame_count}/{src_frame_count}"
    )


if __name__ == "__main__":
    main()
