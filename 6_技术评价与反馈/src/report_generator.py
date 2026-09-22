from __future__ import annotations

import base64
import mimetypes
from html import escape
from pathlib import Path
from typing import Any


STAGE_LABELS = {
    "takeoff": "起跨",
    "flight": "腾空",
    "landing": "下栏",
    "overall": "整体",
}


def _embed_image(target_path: str | Path | None) -> str:
    if not target_path:
        return ""
    path = Path(target_path)
    if not path.exists():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_html_report(payload: dict[str, Any], chart_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_src = _embed_image(chart_path)
    chart_html = f'<img src="{chart_src}" alt="score chart">' if chart_src else "<p>暂无评分图。</p>"
    overall_score = payload.get("overall_score")
    overall_text = f"{overall_score['score']:.2f} / 5" if overall_score and overall_score.get("score") is not None else "暂无"

    stage_rows = "\n".join(
        f"<tr><td>{escape(item['stage_name'])}</td><td>{float(item['score']):.2f}</td><td>{item['score_round']}</td></tr>"
        for item in payload["stage_feedback"]
        if item["score"] is not None and item["score_round"] is not None
    )
    item_rows = "\n".join(
        f"<tr><td>{escape(item.get('display_task_id', item['task_id']))}</td><td>{escape(item.get('display_task_name', item['task_name']))}</td><td>{escape(STAGE_LABELS.get(item['stage'], item['stage']))}</td><td>{item['score']:.2f}</td><td>{item['score_round']}</td></tr>"
        for item in payload["item_scores"]
        if item["score"] is not None and item["score_round"] is not None
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(payload['sample_name'])} 技术评分报告</title>
  <style>
    :root {{ --fg:#1f2328;--muted:#59636e;--subtle:#f6f8fa;--border:#d1d9e0;--green:#1f883d;--dark:#25292e;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    * {{ box-sizing: border-box; }}
    body {{ max-width:1200px; margin:0 auto; padding:20px; color:var(--fg); background:#fff; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Helvetica,Arial,sans-serif; font-size:13px; line-height:1.5; }}
    h1,h2 {{ margin:0 0 10px; }} h1 {{ font-size:24px; }} h2 {{ font-size:16px; }}
    .hero,.card,.wide-card {{ background:#fff; border:1px solid var(--border); border-radius:6px; overflow:hidden; }}
    .hero {{ margin-bottom:16px; padding:18px 20px; border-left:4px solid var(--green); }}
    .grid {{ display:grid; grid-template-columns:.8fr 1.2fr; gap:14px; }}
    .card,.wide-card {{ padding:16px; }} .wide-card {{ margin-top:14px; }}
    .score {{ margin-top:10px; color:var(--green); font:700 34px/1 var(--mono); }}
    .meta {{ color:var(--muted); margin-top:6px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ padding:9px 10px; border-top:1px solid var(--border); text-align:left; font-size:12px; vertical-align:top; }}
    th {{ color:var(--muted); background:var(--subtle); font:600 10px var(--mono); }} tbody tr:hover {{ background:var(--subtle); }}
    img {{ width:100%; border:1px solid var(--border); border-radius:4px; display:block; }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <section class="hero">
    <h1>{escape(payload['sample_name'])} 技术评分报告</h1>
    <div class="meta">视频：{escape(payload['source_video_name'])}</div>
    <div class="score">总评分：{overall_text}</div>
  </section>

  <section class="grid">
    <div class="card">
      <h2>阶段评分</h2>
      <table>
        <thead><tr><th>阶段</th><th>预测分</th><th>整数分</th></tr></thead>
        <tbody>{stage_rows or "<tr><td colspan='3'>暂无</td></tr>"}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>评分可视化</h2>
      {chart_html}
    </div>
  </section>

  <section class="wide-card">
    <h2>单项评分明细</h2>
    <table>
      <thead><tr><th>编号</th><th>评价项</th><th>阶段</th><th>预测分</th><th>整数分</th></tr></thead>
      <tbody>{item_rows or "<tr><td colspan='5'>暂无</td></tr>"}</tbody>
    </table>
  </section>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
