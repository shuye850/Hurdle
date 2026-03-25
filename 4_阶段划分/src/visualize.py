from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_LABEL_EN = {
    "起跨": "Takeoff",
    "腾空": "Flight",
    "落地": "Landing",
    "非跨栏步阶段": "Non-phase",
    "非阶段区间": "Non-phase",
}

EVENT_LABEL_ZH = {
    "takeoff_touchdown": "起跨腿着地",
    "takeoff_toeoff": "起跨腿离地",
    "swing_touchdown": "摆动腿着地",
    "swing_toeoff": "摆动腿离地",
}

EVENT_LABEL_EN = {
    "takeoff_touchdown": "Takeoff TD",
    "takeoff_toeoff": "Takeoff TO",
    "swing_touchdown": "Swing TD",
    "swing_toeoff": "Swing TO",
}

EVENT_ORDER = [
    "takeoff_touchdown",
    "takeoff_toeoff",
    "swing_touchdown",
    "swing_toeoff",
]

EVENT_ACCENT = {
    "takeoff_touchdown": (126, 94, 62),
    "takeoff_toeoff": (74, 112, 168),
    "swing_touchdown": (88, 130, 92),
    "swing_toeoff": (126, 92, 122),
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

EN_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]


def _load_font(font_size: int, candidates: list[str]):
    try:
        from PIL import ImageFont
    except ModuleNotFoundError:
        return None

    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    return None


def _load_chinese_font(font_size: int):
    return _load_font(font_size, FONT_CANDIDATES)


def _load_english_font(font_size: int):
    return _load_font(font_size, EN_FONT_CANDIDATES)


def _draw_text(
    frame,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 1.0,
    thickness: int = 2,
    prefer_chinese: bool = False,
    prefer_english_serif: bool = False,
):
    use_pil = False
    if prefer_chinese or prefer_english_serif:
        try:
            from PIL import Image, ImageDraw
            if prefer_chinese:
                font = _load_chinese_font(max(24, int(30 * font_scale)))
            else:
                font = _load_english_font(max(22, int(30 * font_scale)))
            if font is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(pil_img)
                draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
                frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                use_pil = True
        except Exception:
            use_pil = False

    if not use_pil:
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _get_text_size(
    text: str,
    font_scale: float = 1.0,
    prefer_chinese: bool = False,
    prefer_english_serif: bool = False,
) -> tuple[int, int]:
    try:
        from PIL import Image, ImageDraw
        if prefer_chinese:
            font = _load_chinese_font(max(24, int(30 * font_scale)))
        elif prefer_english_serif:
            font = _load_english_font(max(22, int(30 * font_scale)))
        else:
            font = None

        if font is not None:
            dummy = Image.new("RGB", (32, 32), (255, 255, 255))
            draw = ImageDraw.Draw(dummy)
            bbox = draw.textbbox((0, 0), text, font=font)
            return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
    except Exception:
        pass

    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    return w, h


def _draw_status_bar(frame, phase_name: str, event_key: Optional[str], idx: int) -> None:
    h, w = frame.shape[:2]
    scale = max(w / 1920.0, h / 1080.0)
    bar_h = max(int(110 * scale), int(h * 0.13))
    y0 = h - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.line(frame, (0, y0), (w, y0), (90, 190, 255), max(2, int(2 * scale)))
    cv2.line(frame, (int(24 * scale), h - int(16 * scale)), (w - int(24 * scale), h - int(16 * scale)), (70, 80, 96), max(1, int(scale)))

    label_fs = 0.82 * scale
    value_fs = 1.38 * scale
    event_fs = 1.22 * scale
    thick = max(2, int(2 * scale))

    col1 = int(34 * scale)
    col2 = int(w * 0.30)
    col3 = int(w * 0.60)
    label_y = y0 + int(34 * scale)
    value_y = y0 + int(80 * scale)

    _draw_text(frame, "FRAME", (col1, label_y), (150, 165, 180), font_scale=label_fs, thickness=1, prefer_english_serif=True)
    _draw_text(frame, f"{idx:04d}", (col1, value_y), (255, 245, 210), font_scale=value_fs, thickness=thick, prefer_english_serif=True)

    phase_label = phase_name
    phase_text_is_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in phase_label)
    if not phase_text_is_chinese:
        phase_label = PHASE_LABEL_EN.get(phase_name, phase_name)
    _draw_text(frame, "PHASE", (col2, label_y), (150, 165, 180), font_scale=label_fs, thickness=1, prefer_english_serif=True)
    phase_text = phase_label if phase_text_is_chinese else phase_label.upper()
    _draw_text(
        frame,
        phase_text,
        (col2, value_y),
        (130, 255, 170),
        font_scale=1.28 * scale,
        thickness=thick,
        prefer_chinese=phase_text_is_chinese,
        prefer_english_serif=not phase_text_is_chinese,
    )

    _draw_text(frame, "EVENT", (col3, label_y), (150, 165, 180), font_scale=label_fs, thickness=1, prefer_english_serif=True)
    if event_key is not None:
        event_zh = EVENT_LABEL_ZH.get(event_key, event_key)
        event_en = EVENT_LABEL_EN.get(event_key, event_key)
        event_text = event_zh
        prefer_chinese = True
        if _load_chinese_font(max(24, int(30 * scale))) is None:
            event_text = event_en.upper()
            prefer_chinese = False
        event_color = (110, 170, 255)
    else:
        event_text = "NONE"
        prefer_chinese = False
        event_color = (110, 118, 130)

    _draw_text(
        frame,
        event_text,
        (col3, value_y),
        event_color,
        font_scale=event_fs,
        thickness=thick,
        prefer_chinese=prefer_chinese,
        prefer_english_serif=not prefer_chinese,
    )


def _event_display_name(event_key: str) -> tuple[str, bool]:
    label_zh = EVENT_LABEL_ZH.get(event_key, event_key)
    if _load_chinese_font(30) is not None:
        return label_zh, True
    return EVENT_LABEL_EN.get(event_key, event_key).upper(), False


def _phase_display_name(phase_name: str) -> tuple[str, bool]:
    if any("\u4e00" <= ch <= "\u9fff" for ch in phase_name) and _load_chinese_font(28) is not None:
        return phase_name, True
    return PHASE_LABEL_EN.get(phase_name, phase_name).upper(), False


def _fit_crop_to_card(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.full((target_h, target_w, 3), 245, dtype=np.uint8)

    scale = max(target_w / src_w, target_h / src_h)
    resized = cv2.resize(image, (int(round(src_w * scale)), int(round(src_h * scale))), interpolation=cv2.INTER_LINEAR)
    rh, rw = resized.shape[:2]
    x0 = max(0, (rw - target_w) // 2)
    y0 = max(0, (rh - target_h) // 2)
    return resized[y0:y0 + target_h, x0:x0 + target_w].copy()


def _crop_canvas_whitespace(image: np.ndarray, pad: int = 12, white_thresh: int = 248) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < white_thresh
    if not np.any(mask):
        return image

    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    return image[y0:y1, x0:x1].copy()


def render_event_montage(
    video_path: str,
    segmented_df: pd.DataFrame,
    meta: Dict,
    output_path: str,
    video_stem: str,
) -> Optional[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    frame_to_phase = {int(r.frame_index): str(r.phase_name) for r in segmented_df.itertuples(index=False)}
    events = meta.get("events", {})

    canvas_w, canvas_h = 1920, 1080
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    title_text = "Key Event Frames"
    video_text = f"Video ID: {video_stem}"
    title_w, title_h = _get_text_size(title_text, font_scale=1.32, prefer_english_serif=True)
    video_w, video_h = _get_text_size(video_text, font_scale=0.94, prefer_english_serif=True)
    header_h = 26 + title_h + 10 + video_h + 14
    cv2.line(canvas, (74, header_h), (canvas_w - 74, header_h), (188, 194, 204), 2)

    title_y = 14
    video_y = title_y + title_h + 10
    _draw_text(canvas, title_text, (80, title_y), (36, 42, 54), font_scale=1.32, thickness=2, prefer_english_serif=True)
    _draw_text(canvas, video_text, (82, video_y), (82, 90, 102), font_scale=0.94, thickness=1, prefer_english_serif=True)

    left_margin = 80
    top_margin = header_h + 16
    gap_x = 26
    gap_y = 14
    card_w = (canvas_w - 2 * left_margin - gap_x) // 2
    image_h = 352
    detail_probe = "Event: 摆动腿离地    Frame: 47    Time: 0.783 s"
    _, detail_h = _get_text_size(detail_probe, font_scale=1.08, prefer_chinese=True)
    meta_h = detail_h + 26
    card_h = image_h + meta_h + 30

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for idx, event_key in enumerate(EVENT_ORDER):
        frame_idx = int(events.get(event_key, -1))
        row = idx // 2
        col = idx % 2
        x0 = left_margin + col * (card_w + gap_x)
        y0 = top_margin + row * (card_h + gap_y)
        x1 = x0 + card_w
        y1 = y0 + card_h

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (208, 212, 218), 1)
        accent = EVENT_ACCENT.get(event_key, (90, 140, 220))

        ok = frame_idx >= 0
        frame_img = np.full((image_h, card_w - 40, 3), 242, dtype=np.uint8)
        if ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, raw = cap.read()
            if ret and raw is not None:
                frame_img = _fit_crop_to_card(raw, card_w - 40, image_h)

        img_x0 = x0 + 20
        img_y0 = y0 + 18
        canvas[img_y0:img_y0 + image_h, img_x0:img_x0 + (card_w - 40)] = frame_img
        cv2.rectangle(canvas, (img_x0, img_y0), (img_x0 + card_w - 40, img_y0 + image_h), (214, 218, 224), 1)

        event_label, event_is_zh = _event_display_name(event_key)
        meta_y0 = y0 + image_h + 18 + detail_h
        time_text = f"{(frame_idx / fps):.3f} s" if ok else "N/A"
        detail_text = f"Event: {event_label}    Frame: {frame_idx if ok else 'N/A'}    Time: {time_text}"
        _draw_text(
            canvas,
            detail_text,
            (x0 + 28, meta_y0 - detail_h),
            (64, 72, 84),
            font_scale=1.08,
            thickness=1,
            prefer_chinese=event_is_zh,
            prefer_english_serif=not event_is_zh,
        )

        # subtle separator between image and metadata
        cv2.line(canvas, (x0 + 24, y0 + image_h + 8), (x1 - 24, y0 + image_h + 8), (226, 230, 236), 1)

    cropped = _crop_canvas_whitespace(canvas, pad=10)
    cv2.imwrite(str(output_path), cropped)
    cap.release()
    return str(output_path)


def save_debug_plot(df: pd.DataFrame, meta: Dict, output_path: str) -> None:
    output_path = str(output_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 16,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    })

    events = meta.get("events", {})
    window_start = meta.get("pre_bar_last_cross_row", None)
    window_end = meta.get("cross_bar_row", None)

    color_map = {
        "takeoff_touchdown": "tab:blue",
        "takeoff_toeoff": "tab:orange",
        "swing_touchdown": "tab:green",
        "swing_toeoff": "tab:red",
    }

    # 1) 踝点 y 曲线图
    plt.figure(figsize=(14, 7.5))
    plt.plot(df["frame_index"], df["left_ankle_y_smooth"], label="left_ankle_y_smooth", linewidth=2.6)
    plt.plot(df["frame_index"], df["right_ankle_y_smooth"], label="right_ankle_y_smooth", linewidth=2.6)

    if "bar_mid_y" in df.columns:
        plt.plot(df["frame_index"], df["bar_mid_y"], "--", label="bar_mid_y", linewidth=2.2)

    # 阴影窗口：过栏前最后一次双踝 y 交替 到 膝关节过栏
    if window_start is not None and window_end is not None:
        plt.axvspan(
            float(window_start),
            float(window_end),
            color="gray",
            alpha=0.18,
            label="search_window",
        )

    for name, frame in events.items():
        plt.axvline(frame, linestyle=":", linewidth=2.0, color=color_map.get(name, "k"), label=name)

    plt.xlabel("Frame Index")
    plt.ylabel("Image Y (downward positive)")
    plt.title("Ankle Trajectory and Search Window")
    plt.gca().invert_yaxis()
    plt.grid(axis="y", linestyle="--", linewidth=0.8, color="#D7DCE2", alpha=0.9)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("#9AA4B2")
    plt.gca().spines["bottom"].set_color("#9AA4B2")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    # 2) 膝关节角度图
    knee_out = out.with_name(out.stem.replace("_phase_debug_plot", "_knee_angle_plot") + out.suffix)

    plt.figure(figsize=(14, 7.5))
    if "left_knee_angle" in df.columns:
        plt.plot(df["frame_index"], df["left_knee_angle"], label="left_knee_angle", linewidth=2.6)
    if "right_knee_angle" in df.columns:
        plt.plot(df["frame_index"], df["right_knee_angle"], label="right_knee_angle", linewidth=2.6)

    # 同样给膝角图加阴影窗口
    if window_start is not None and window_end is not None:
        plt.axvspan(
            float(window_start),
            float(window_end),
            color="gray",
            alpha=0.18,
            label="search_window",
        )

    for name, frame in events.items():
        plt.axvline(frame, linestyle=":", linewidth=2.0, color=color_map.get(name, "k"), label=name)

    plt.xlabel("Frame Index")
    plt.ylabel("Knee Angle (deg)")
    plt.title("Knee Angle Curve")
    plt.grid(axis="y", linestyle="--", linewidth=0.8, color="#D7DCE2", alpha=0.9)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("#9AA4B2")
    plt.gca().spines["bottom"].set_color("#9AA4B2")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(knee_out, dpi=150)
    plt.close()


def render_phase_overlay(
    video_path: str,
    segmented_df: pd.DataFrame,
    meta: Dict,
    output_path: str,
) -> Optional[str]:
    def _valid_point(*values: object) -> bool:
        for value in values:
            if value is None or pd.isna(value):
                return False
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )

    phase_map = {}
    for row in segmented_df.itertuples(index=False):
        phase_map[int(row.frame_index)] = str(row.phase_name)

    events = meta.get("events", {})
    event_name_by_frame = {int(v): k for k, v in events.items()}

    df_by_frame = {int(r.frame_index): r for r in segmented_df.itertuples(index=False)}

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        row = df_by_frame.get(idx)
        phase_name = phase_map.get(idx, "非阶段区间")
        event_key = event_name_by_frame.get(idx)
        _draw_status_bar(frame, phase_name, event_key, idx)

        if row is not None:
            # 栏架关键点
            if hasattr(row, "bar_mid_x") and hasattr(row, "bar_mid_y") and _valid_point(row.bar_mid_x, row.bar_mid_y):
                cv2.circle(frame, (int(row.bar_mid_x), int(row.bar_mid_y)), 5, (0, 255, 0), -1)

            # 髋膝踝
            for side, color in [("left", (255, 0, 0)), ("right", (0, 140, 255))]:
                hip_x_name = f"{side}_hip_x"
                hip_y_name = f"{side}_hip_y"
                knee_x_name = f"{side}_knee_x"
                knee_y_name = f"{side}_knee_y"
                ankle_x_name = f"{side}_ankle_x"
                ankle_y_name = f"{side}_ankle_y"

                if all(
                    hasattr(row, n)
                    for n in [
                        hip_x_name,
                        hip_y_name,
                        knee_x_name,
                        knee_y_name,
                        ankle_x_name,
                        ankle_y_name,
                    ]
                ):
                    hip_x = getattr(row, hip_x_name)
                    hip_y = getattr(row, hip_y_name)
                    knee_x = getattr(row, knee_x_name)
                    knee_y = getattr(row, knee_y_name)
                    ankle_x = getattr(row, ankle_x_name)
                    ankle_y = getattr(row, ankle_y_name)

                    if not _valid_point(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
                        continue

                    hip = (int(hip_x), int(hip_y))
                    knee = (int(knee_x), int(knee_y))
                    ankle = (int(ankle_x), int(ankle_y))

                    cv2.circle(frame, hip, 4, color, -1)
                    cv2.circle(frame, knee, 4, color, -1)
                    cv2.circle(frame, ankle, 4, color, -1)
                    cv2.line(frame, hip, knee, color, 2)
                    cv2.line(frame, knee, ankle, color, 2)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    return str(output_path)
