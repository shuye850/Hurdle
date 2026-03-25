from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["grid.linewidth"] = 0.6


STAGE_COLORS = {
    "takeoff": "#4C78A8",
    "flight": "#59A14F",
    "landing": "#F28E2B",
    "overall": "#B07AA1",
}

STAGE_LABELS = {
    "takeoff": "起跨",
    "flight": "腾空",
    "landing": "下栏",
    "overall": "总评",
}


def _group_items(payload: dict[str, Any]) -> tuple[list[str], list[float], list[str], list[int]]:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    separators: list[int] = []

    items = payload["item_scores"]
    for idx, item in enumerate(items):
        labels.append(item["task_id"])
        values.append(float(item["score"] or 0.0))
        colors.append(STAGE_COLORS.get(item["stage"], "#7F7F7F"))
        if idx < len(items) - 1 and item["stage"] != items[idx + 1]["stage"]:
            separators.append(idx)
    return labels, values, colors, separators


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9DEE7", alpha=0.9)
    ax.spines["left"].set_color("#7A869A")
    ax.spines["bottom"].set_color("#7A869A")
    ax.tick_params(colors="#344054", labelsize=10)


def render_summary_chart(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), dpi=240)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.2, 1.25], hspace=0.42)

    overall_score = payload.get("overall_score")
    stage_feedback = payload["stage_feedback"]
    item_labels, item_values, item_colors, separators = _group_items(payload)

    ax0 = fig.add_subplot(gs[0, 0])
    stage_names = [item["stage_name"] for item in stage_feedback]
    stage_scores = [float(item["score"] or 0.0) for item in stage_feedback]
    stage_colors = [STAGE_COLORS.get(item["stage"], "#7F7F7F") for item in stage_feedback]
    x0 = np.arange(len(stage_names))
    bars0 = ax0.bar(x0, stage_scores, color=stage_colors, width=0.58, edgecolor="#4A5568", linewidth=0.4)
    if overall_score is not None and overall_score.get("score") is not None:
        ax0.axhline(
            float(overall_score["score"]),
            color=STAGE_COLORS["overall"],
            linewidth=1.6,
            linestyle="--",
            label=f"总评分 {overall_score['score']:.2f}",
        )
        ax0.legend(frameon=False, loc="upper right", fontsize=10)
    for bar, val in zip(bars0, stage_scores):
        ax0.text(bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=10, color="#253047")
    ax0.set_xticks(x0, stage_names)
    ax0.set_ylim(0, 5.2)
    ax0.set_ylabel("预测分值", fontsize=11)
    ax0.set_title(f"{payload['sample_name']} 评分结果概览", loc="left", fontsize=16, pad=10, color="#1F2937")
    _style_axis(ax0)

    ax1 = fig.add_subplot(gs[1, 0])
    x1 = np.arange(len(item_labels))
    bars1 = ax1.bar(x1, item_values, color=item_colors, width=0.72, edgecolor="#4A5568", linewidth=0.35)
    for sep in separators:
        ax1.axvline(sep + 0.5, color="#B8C1CC", linewidth=1.0)
    for bar, val in zip(bars1, item_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#253047")
    ax1.set_xticks(x1, item_labels)
    ax1.set_ylim(0, 5.2)
    ax1.set_ylabel("预测分值", fontsize=11)
    ax1.set_title("单项评分分布", loc="left", fontsize=14, pad=8, color="#1F2937")
    _style_axis(ax1)

    stage_ranges = []
    start = 0
    items = payload["item_scores"]
    for idx in separators + [len(items) - 1]:
        stage = items[start]["stage"]
        stage_ranges.append(((start + idx) / 2, STAGE_LABELS.get(stage, stage)))
        start = idx + 1
    top_y = 5.05
    for center, label in stage_ranges:
        ax1.text(center, top_y, label, ha="center", va="bottom", fontsize=10, color="#667085")

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis("off")
    weak_items = payload.get("weak_items", [])
    strong_items = payload.get("strong_items", [])
    stage_text = "  ".join(
        f"{item['stage_name']} {float(item['score'] or 0.0):.2f}"
        for item in stage_feedback
    )
    weak_text = "、".join(f"{item['task_id']} {item['task_name']}" for item in weak_items[:4]) or "暂无明显薄弱项"
    strong_text = "、".join(f"{item['task_id']} {item['task_name']}" for item in strong_items[:4]) or "暂无"
    overall_text = f"{float(overall_score['score']):.2f}" if overall_score and overall_score.get("score") is not None else "暂无"

    ax2.text(0.00, 0.95, "结果摘要", fontsize=13, fontweight="bold", color="#1F2937", transform=ax2.transAxes)
    ax2.text(0.00, 0.68, f"总评分: {overall_text}", fontsize=11, color="#344054", transform=ax2.transAxes)
    ax2.text(0.23, 0.68, f"阶段评分: {stage_text}", fontsize=11, color="#344054", transform=ax2.transAxes)
    ax2.text(0.00, 0.40, f"重点薄弱项: {weak_text}", fontsize=11, color="#344054", transform=ax2.transAxes)
    ax2.text(0.00, 0.16, f"优势项: {strong_text}", fontsize=11, color="#344054", transform=ax2.transAxes)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
