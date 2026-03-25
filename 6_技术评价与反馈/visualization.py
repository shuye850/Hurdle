from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def render_summary_chart(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [item["task_id"] for item in payload["item_scores"]]
    values = [item["score_round"] or 0 for item in payload["item_scores"]]
    colors = []
    for value in values:
        if value <= 2:
            colors.append("#d1495b")
        elif value == 3:
            colors.append("#edae49")
        else:
            colors.append("#3f88c5")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), dpi=180)

    ax = axes[0]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 5)
    ax.set_title(f"{payload['sample_name']} 单项评分")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.2)

    stage_labels = [item["stage_name"] for item in payload["stage_feedback"]]
    stage_values = [item["score_round"] or 0 for item in payload["stage_feedback"]]
    stage_colors = ["#2a9d8f", "#457b9d", "#8ab17d"]

    ax = axes[1]
    ax.bar(stage_labels, stage_values, color=stage_colors[: len(stage_labels)])
    if payload.get("overall_score") is not None:
        ax.axhline(payload["overall_score"]["score_round"] or 0, color="#1d3557", linestyle="--", label="总评")
    ax.set_ylim(0, 5)
    ax.set_title("阶段评分")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.2)
    if payload.get("overall_score") is not None:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
