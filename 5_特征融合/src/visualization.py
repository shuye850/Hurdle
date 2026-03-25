# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Songti SC", "Songti TC", "STSong", "SimSun", "Noto Serif CJK SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


PHASE_COLORS = {
    "起跨": "#E8C97C",
    "腾空": "#90AED2",
    "下栏": "#A5CFA9",
}

EVENT_LABELS = {
    "takeoff_landing_frame": "起跨腿着地",
    "takeoff_toeoff_frame": "起跨腿离地",
    "bar_cross_frame": "膝关节过栏",
    "com_peak_frame": "重心最高点",
    "landing_frame": "摆动腿着地",
    "recovery_toeoff_frame": "摆动腿离地",
}

EVENT_ORDER = [
    "takeoff_landing_frame",
    "takeoff_toeoff_frame",
    "bar_cross_frame",
    "com_peak_frame",
    "landing_frame",
    "recovery_toeoff_frame",
]

EVENT_COLORS = {
    "takeoff_landing_frame": "#B86A2E",
    "takeoff_toeoff_frame": "#CC8C22",
    "bar_cross_frame": "#2E6DB2",
    "com_peak_frame": "#6C62A9",
    "landing_frame": "#3D8D5C",
    "recovery_toeoff_frame": "#A5507B",
}


def _style_axis(ax, show_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB8C3")
    ax.spines["bottom"].set_color("#AEB8C3")
    ax.tick_params(colors="#445266", labelsize=15)
    ax.set_facecolor("white")
    if show_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.9, color="#D7DDE6", alpha=0.95)


def _event_items(events: dict) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for key in EVENT_ORDER:
        if key in events and pd.notna(events[key]):
            items.append((key, int(events[key])))
    return items


def _shade_phases(ax, events: dict) -> None:
    spans = [
        ("起跨", int(events["takeoff_landing_frame"]), int(events["takeoff_toeoff_frame"])),
        ("腾空", int(events["takeoff_toeoff_frame"]), int(events["landing_frame"])),
        ("下栏", int(events["landing_frame"]), int(events["recovery_toeoff_frame"])),
    ]
    for name, start, end in spans:
        ax.axvspan(start, end, color=PHASE_COLORS[name], alpha=0.18, linewidth=0)


def _mark_events(ax, events: dict) -> None:
    items = _event_items(events)
    if not items:
        return

    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)

    for idx, (key, frame) in enumerate(items):
        color = EVENT_COLORS[key]
        ax.axvline(frame, linestyle="--", linewidth=1.5, color=color, alpha=0.92)
        y = ymax - (0.055 + 0.06 * (idx % 2)) * span
        ax.scatter([frame], [y], s=54, color=color, zorder=6)
        ax.text(frame, y, str(idx + 1), ha="center", va="center", fontsize=9.5, color="white", zorder=7)


def _fmt(value, fmt: str) -> str:
    try:
        return format(float(value), fmt)
    except Exception:
        return "N/A"


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def _draw_metric_column(
    ax,
    x_left: float,
    x_right: float,
    y_top: float,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    y = y_top
    for title, items in sections:
        ax.text(x_left, y, title, fontsize=18, fontweight="bold", color="#334155", va="top")
        ax.hlines(y - 0.035, x_left, x_right, colors="#E2E8F0", linewidth=1.0)
        y -= 0.082
        for label, value in items:
            ax.text(x_left, y, label, fontsize=14.2, color="#64748B", va="top")
            ax.text(x_right, y, value, fontsize=14.2, color="#1F2933", va="top", ha="right")
            y -= 0.058
        y -= 0.038


def _metric_sections(metrics: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("基础信息", [
            ("摆动腿", str(metrics.get("leading_leg", "N/A"))),
            ("起跨腿", str(metrics.get("trail_leg", "N/A"))),
            ("栏高", f"{_fmt(metrics.get('post_height_real'), '.3f')} m"),
            ("比例尺", f"{_fmt(metrics.get('pixel_to_meter_scale'), '.5f')} m/px"),
        ]),
        ("阶段时长", [
            ("起跨时长", f"{_fmt(metrics.get('takeoff_phase_time_s'), '.3f')} s"),
            ("腾空时长", f"{_fmt(metrics.get('flight_time_s'), '.3f')} s"),
            ("下栏时长", f"{_fmt(metrics.get('landing_phase_time_s'), '.3f')} s"),
            ("总时长", f"{_fmt(metrics.get('total_phase_time_s'), '.3f')} s"),
            ("起跨占比", _fmt_pct(metrics.get("takeoff_phase_ratio"))),
            ("腾空占比", _fmt_pct(metrics.get("flight_phase_ratio"))),
            ("下栏占比", _fmt_pct(metrics.get("landing_phase_ratio"))),
        ]),
        ("时空参数", [
            ("起跨距离", f"{_fmt(metrics.get('takeoff_distance_m'), '.3f')} m"),
            ("腾空位移", f"{_fmt(metrics.get('flight_disp_m'), '.3f')} m"),
            ("下栏距离", f"{_fmt(metrics.get('landing_distance_m'), '.3f')} m"),
            ("着地点重心距", f"{_fmt(metrics.get('landing_com_dist_m'), '.3f')} m"),
            ("离地到过栏", f"{_fmt(metrics.get('toeoff_to_bar_time_s'), '.3f')} s"),
            ("过栏到着地", f"{_fmt(metrics.get('bar_to_landing_time_s'), '.3f')} s"),
        ]),
        ("重心特征", [
            ("腾起高度", f"{_fmt(metrics.get('com_rise_m'), '.3f')} m"),
            ("过栏重心高度", f"{_fmt(metrics.get('bar_cross_com_height_m'), '.3f')} m"),
            ("过栏净空", f"{_fmt(metrics.get('bar_cross_com_clearance_m'), '.3f')} m"),
            ("最高点到栏架", f"{_fmt(metrics.get('com_peak_to_post_m'), '.3f')} m"),
            ("最高点帧", _fmt(metrics.get("com_peak_frame"), ".0f")),
        ]),
        ("速度与角度", [
            ("起跨前 Vx", f"{_fmt(metrics.get('com_vx_pre_takeoff_mps'), '.2f')} m/s"),
            ("离地瞬间 Vx", f"{_fmt(metrics.get('com_vx_takeoff_toeoff_mps'), '.2f')} m/s"),
            ("着地后 Vx", f"{_fmt(metrics.get('com_vx_landing_mps'), '.2f')} m/s"),
            ("离地瞬间 Vy", f"{_fmt(metrics.get('com_vy_takeoff_toeoff_mps'), '.2f')} m/s"),
            ("起跨角", f"{_fmt(metrics.get('takeoff_angle_deg'), '.2f')} °"),
            ("过栏躯干角", f"{_fmt(metrics.get('bar_cross_trunk_angle_deg'), '.2f')} °"),
            ("过栏摆动腿膝角", f"{_fmt(metrics.get('bar_cross_lead_knee_angle_deg'), '.2f')} °"),
            ("下栏着地角", f"{_fmt(metrics.get('landing_shank_angle_deg'), '.2f')} °"),
        ]),
        ("时序比例", [
            ("离地到过栏", f"{_fmt(metrics.get('toeoff_to_bar_time_s'), '.3f')} s"),
            ("过栏到着地", f"{_fmt(metrics.get('bar_to_landing_time_s'), '.3f')} s"),
            ("离地到过栏占比", _fmt_pct(metrics.get("toeoff_to_bar_ratio"))),
            ("过栏到着地占比", _fmt_pct(metrics.get("bar_to_landing_ratio"))),
            ("最高点帧", _fmt(metrics.get("com_peak_frame"), ".0f")),
            ("总帧数", _fmt(metrics.get("total_frames"), ".0f")),
        ]),
    ]


def _draw_metrics_panel(ax, metrics: dict) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.00, 0.99, "关键量化指标", fontsize=24, fontweight="bold", color="#1F2933", va="top")
    ax.hlines(0.935, 0.00, 1.00, colors="#D7DDE6", linewidth=1.1)

    sections = _metric_sections(metrics)
    columns = [
        [sections[0], sections[3]],
        [sections[1], sections[2]],
        [sections[4], sections[5]],
    ]
    x_pairs = [
        (0.03, 0.30),
        (0.39, 0.66),
        (0.74, 0.97),
    ]

    for (x_left, x_right), column_sections in zip(x_pairs, columns):
        _draw_metric_column(ax, x_left, x_right, 0.88, column_sections)


def render_summary_figure(
    sample_name: str,
    frame_df: pd.DataFrame,
    event_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    video_path: Optional[Path],
    cfg: dict,
    output_path: Path,
) -> None:
    _ = video_path
    events = event_df.iloc[0].to_dict()
    metrics = metric_df.iloc[0].to_dict()

    frames = frame_df["frame"].to_numpy(dtype=float)
    scale = float(metrics["pixel_to_meter_scale"])
    com_h = (float(metrics["post_base_y"]) - frame_df["com_y"].to_numpy(dtype=float)) * scale
    vx = frame_df["com_vx_mps"].to_numpy(dtype=float)
    vy = frame_df["com_vy_mps"].to_numpy(dtype=float)

    dpi = int(cfg["visualization"]["dpi"])
    fig = plt.figure(figsize=(18.4, 15.4), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(
        nrows=5,
        ncols=12,
        height_ratios=[0.78, 3.00, 2.45, 5.20, 2.95],
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_height = fig.add_subplot(gs[1, :])
    ax_velocity = fig.add_subplot(gs[2, :])
    ax_metrics = fig.add_subplot(gs[3, :])
    ax_timeline = fig.add_subplot(gs[4, :])
    fig.subplots_adjust(left=0.045, right=0.985, top=0.972, bottom=0.055, hspace=0.58)

    ax_header.axis("off")
    ax_header.text(
        0.00,
        0.82,
        f"{sample_name} 技术特征综合分析图",
        fontsize=31,
        fontweight="bold",
        color="#1F2933",
        ha="left",
        va="center",
    )
    ax_header.text(
        0.00,
        0.18,
        "围绕重心轨迹、速度变化、阶段时长与关键事件，对跨栏动作完成过程进行量化分析。",
        fontsize=16.5,
        color="#5E6B7A",
        ha="left",
        va="center",
    )
    ax_header.axhline(0.02, color="#D9DEE6", linewidth=1.2)

    _style_axis(ax_height, show_grid=True)
    _shade_phases(ax_height, events)
    ax_height.plot(frames, com_h, color="#2F6FB4", linewidth=3.3, label="重心高度")
    ax_height.set_title("重心高度变化与关键事件", fontsize=24, loc="left", pad=10)
    ax_height.set_xlabel("Frame", fontsize=17)
    ax_height.set_ylabel("CoM Height (m)", fontsize=17)
    ax_height.set_ylim(float(np.nanmin(com_h)) - 0.02, float(np.nanmax(com_h)) + 0.03)
    _mark_events(ax_height, events)
    ax_height.legend(loc="upper right", frameon=False, fontsize=16)

    _style_axis(ax_velocity, show_grid=True)
    _shade_phases(ax_velocity, events)
    ax_velocity.plot(frames, vx, color="#C7772C", linewidth=2.8, label="水平速度 Vx")
    ax_velocity.plot(frames, vy, color="#4F8F67", linewidth=2.8, label="垂直速度 Vy")
    ax_velocity.set_title("重心速度变化", fontsize=22, loc="left", pad=8)
    ax_velocity.set_xlabel("Frame", fontsize=16)
    ax_velocity.set_ylabel("Velocity (m/s)", fontsize=16)
    _mark_events(ax_velocity, events)
    ax_velocity.legend(loc="upper right", frameon=False, fontsize=15)

    _draw_metrics_panel(ax_metrics, metrics)

    ax_timeline.set_facecolor("white")
    ax_timeline.set_xlim(frames.min(), frames.max())
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.spines["top"].set_visible(False)
    ax_timeline.spines["right"].set_visible(False)
    ax_timeline.spines["left"].set_visible(False)
    ax_timeline.spines["bottom"].set_color("#AEB8C3")
    ax_timeline.tick_params(colors="#445266", labelsize=15)
    ax_timeline.set_title("阶段划分与关键事件定位", fontsize=22, loc="left", pad=10)

    phase_spans = [
        ("起跨", int(events["takeoff_landing_frame"]), int(events["takeoff_toeoff_frame"]), metrics.get("takeoff_phase_time_s")),
        ("腾空", int(events["takeoff_toeoff_frame"]), int(events["landing_frame"]), metrics.get("flight_time_s")),
        ("下栏", int(events["landing_frame"]), int(events["recovery_toeoff_frame"]), metrics.get("landing_phase_time_s")),
    ]
    for name, start, end, phase_time in phase_spans:
        ax_timeline.axvspan(start, end, ymin=0.56, ymax=0.84, color=PHASE_COLORS[name], alpha=0.92)
        ax_timeline.text(
            (start + end) / 2,
            0.70,
            f"{name}\n{_fmt(phase_time, '.3f')} s",
            ha="center",
            va="center",
            fontsize=18.2,
            color="#243041",
            fontweight="bold",
        )

    for idx, (key, frame) in enumerate(_event_items(events)):
        color = EVENT_COLORS[key]
        ax_timeline.axvline(frame, ymin=0.08, ymax=0.52, linestyle="--", linewidth=1.4, color=color, alpha=0.90)
        ax_timeline.scatter(frame, 0.42, s=64, color=color, zorder=5)
        close_left = idx > 0 and abs(frame - _event_items(events)[idx - 1][1]) <= 3
        close_right = idx < len(_event_items(events)) - 1 and abs(frame - _event_items(events)[idx + 1][1]) <= 3
        if close_left and not close_right:
            y_text = 0.06
            x_text = frame + 0.6
        elif close_right and not close_left:
            y_text = 0.18
            x_text = frame - 0.6
        elif close_left and close_right:
            y_text = 0.12
            x_text = frame + (0.8 if idx % 2 == 0 else -0.8)
        else:
            y_text = 0.18 if idx % 2 == 0 else 0.08
            x_text = frame
        ax_timeline.text(
            x_text,
            y_text,
            f"{EVENT_LABELS[key]}\nFrame {frame}",
            ha="center",
            va="center",
            fontsize=12.8,
            color=color,
        )

    ax_timeline.set_xlabel("Frame", fontsize=17)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0.03)
    plt.close(fig)
    print(f"[可视化输出] 已生成: {output_path}")
