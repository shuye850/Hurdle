from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def render_html_report(payload: dict[str, Any], chart_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_ref = Path("..") / chart_path.parent.name / chart_path.name

    overall_text = "暂无总评分"
    if payload.get("overall_score") is not None:
        overall_text = f"{payload['overall_score']['score']:.2f} / 5"

    item_rows = "\n".join(
        f"<tr><td>{escape(item['task_id'])}</td><td>{escape(item['task_name'])}</td><td>{escape(item['stage'])}</td><td>{item['score']:.2f}</td><td>{item['score_round']}</td></tr>"
        for item in payload["item_scores"]
        if item["score"] is not None and item["score_round"] is not None
    )
    stage_rows = "\n".join(
        f"<tr><td>{escape(item['stage_name'])}</td><td>{item['score']:.2f}</td><td>{item['score_round']}</td><td>{escape(item['summary'])}</td></tr>"
        for item in payload["stage_feedback"]
        if item["score"] is not None and item["score_round"] is not None
    )
    feedback_rows = "\n".join(f"<li>{escape(line)}</li>" for line in payload.get("rule_feedback", []))
    weak_rows = "\n".join(
        f"<li>{escape(item['task_id'])} {escape(item['task_name'])} ({item['score_round']}分)</li>"
        for item in payload.get("weak_items", [])
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(payload['sample_name'])} 技术评价报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; margin: 32px; color: #1f2937; background: linear-gradient(180deg, #f9fafb 0%, #eef6ff 100%); }}
    h1, h2 {{ margin-bottom: 8px; }}
    .hero {{ background: white; border-radius: 18px; padding: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }}
    .meta {{ color: #4b5563; margin-bottom: 16px; }}
    .score {{ font-size: 40px; font-weight: 700; color: #14532d; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }}
    .card {{ background: white; border-radius: 18px; padding: 20px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }}
    img {{ width: 100%; border-radius: 12px; }}
    ul {{ margin: 10px 0 0 18px; }}
  </style>
</head>
<body>
  <section class="hero">
    <h1>{escape(payload['sample_name'])} 技术评价报告</h1>
    <div class="meta">视频：{escape(payload['source_video_name'])}</div>
    <div class="score">总评分：{overall_text}</div>
  </section>
  <section class="grid">
    <div class="card">
      <h2>阶段评价</h2>
      <table>
        <thead><tr><th>阶段</th><th>得分</th><th>整数分</th><th>总结</th></tr></thead>
        <tbody>{stage_rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>练习建议</h2>
      <div>重点薄弱项：</div>
      <ul>{weak_rows or '<li>暂无明显薄弱项</li>'}</ul>
      <div style="margin-top: 14px;">规则建议：</div>
      <ul>{feedback_rows or '<li>当前表现较均衡，可继续保持。</li>'}</ul>
    </div>
  </section>
  <section class="card" style="margin-top: 24px;">
    <h2>评分可视化</h2>
    <img src="{escape(str(chart_ref))}" alt="summary chart">
  </section>
  <section class="card" style="margin-top: 24px;">
    <h2>单项评分明细</h2>
    <table>
      <thead><tr><th>编号</th><th>评价项</th><th>阶段</th><th>预测分</th><th>整数分</th></tr></thead>
      <tbody>{item_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
