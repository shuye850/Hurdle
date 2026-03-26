from __future__ import annotations

import base64
import mimetypes
from html import escape
from pathlib import Path
from typing import Any


KEY_LABELS = {
    "fps": "视频帧率",
    "total_frames": "总帧数",
    "move_dir": "运动方向",
    "leading_leg": "摆动腿",
    "trail_leg": "起跨腿",
    "takeoff_landing_frame": "起跨着地帧",
    "takeoff_toeoff_frame": "起跨离地帧",
    "bar_cross_frame": "过栏帧",
    "knee_bar_cross_frame": "膝过栏帧",
    "landing_frame": "下栏着地帧",
    "recovery_toeoff_frame": "恢复腿离地帧",
    "com_peak_frame": "重心最高点帧",
    "post_x": "栏架立柱 x 坐标",
    "barmid_x": "横杆中心 x 坐标",
    "barmid_y": "横杆中心 y 坐标",
    "box_bottom_y": "检测框底边 y 坐标",
    "far_base_top_y": "远侧底座顶点 y 坐标",
    "post_base_y": "立柱底部 y 坐标",
    "post_height_pixel": "栏高像素值",
    "hurdle_box_width_px": "栏架检测框宽度",
    "hurdle_box_height_px": "栏架检测框高度",
    "hurdle_box_ratio": "栏架检测框宽高比",
    "post_height_real": "栏高",
    "hurdle_height_method": "栏高估计方法",
    "hurdle_height_confidence": "栏高估计置信度",
    "pixel_to_meter_scale": "像素与米换算比例",
    "leg_length_px": "双腿平均长度",
    "leg_length_m": "双腿平均长度",
    "left_leg_length_px": "左腿长度",
    "left_leg_length_m": "左腿长度",
    "right_leg_length_px": "右腿长度",
    "right_leg_length_m": "右腿长度",
    "leading_leg_length_px": "摆动腿长度",
    "leading_leg_length_m": "摆动腿长度",
    "trail_leg_length_px": "起跨腿长度",
    "trail_leg_length_m": "起跨腿长度",
    "takeoff_distance_px": "起跨距离",
    "takeoff_distance_m": "起跨距离",
    "takeoff_angle_deg": "起跨角",
    "takeoff_landing_angle_deg": "起跨着地角",
    "takeoff_landing_knee_angle_deg": "起跨着地膝角",
    "takeoff_max_buffer_frame": "起跨最大缓冲帧",
    "takeoff_buffer_knee_angle_deg": "起跨缓冲膝角",
    "takeoff_toeoff_knee_angle_deg": "起跨离地膝角",
    "takeoff_buffer_amplitude_deg": "起跨缓冲幅度",
    "takeoff_push_amplitude_deg": "起跨蹬伸幅度",
    "takeoff_support_time_s": "起跨支撑时长",
    "takeoff_phase_time_s": "起跨阶段时长",
    "takeoff_toeoff_trunk_angle_deg": "起跨离地躯干角",
    "takeoff_rise_angle_deg": "起跨腾起角",
    "com_vx_pre_takeoff_mps": "起跨前重心水平速度",
    "com_vx_takeoff_toeoff_mps": "起跨支撑段重心水平速度",
    "com_vx_change_mps": "起跨前后水平速度变化",
    "com_vy_pre_takeoff_mps": "起跨前重心垂直速度",
    "com_vy_takeoff_toeoff_mps": "起跨支撑段重心垂直速度",
    "com_vy_change_mps": "起跨前后垂直速度变化",
    "flight_time_s": "腾空时间",
    "flight_disp_px": "腾空水平位移",
    "flight_disp_m": "腾空水平位移",
    "flight_com_speed_mps": "腾空平均水平速度",
    "com_rise_px": "重心抬升高度",
    "com_rise_m": "重心抬升高度",
    "com_peak_y": "重心最高点纵坐标",
    "com_peak_to_post_px": "重心峰值距栏架",
    "com_peak_to_post_m": "重心峰值距栏架",
    "toeoff_to_bar_time_s": "离地到过栏时间",
    "bar_to_landing_time_s": "过栏到落地时间",
    "toeoff_to_bar_ratio": "离地到过栏时间占比",
    "bar_to_landing_ratio": "过栏到落地时间占比",
    "bar_cross_com_height_px": "过栏重心高度",
    "bar_cross_com_height_m": "过栏重心高度",
    "bar_cross_com_clearance_m": "过栏重心净空",
    "bar_cross_lead_knee_angle_deg": "过栏摆动腿膝角",
    "bar_cross_trunk_angle_deg": "过栏躯干角",
    "bar_cross_leading_leg_length_px": "过栏摆动腿长度",
    "bar_cross_leading_leg_length_m": "过栏摆动腿长度",
    "bar_cross_trail_leg_length_px": "过栏起跨腿长度",
    "bar_cross_trail_leg_length_m": "过栏起跨腿长度",
    "landing_distance_px": "下栏距离",
    "landing_distance_m": "下栏距离",
    "landing_com_dist_px": "下栏落地重心距",
    "landing_com_dist_m": "下栏落地重心距",
    "landing_shank_angle_deg": "下栏着地小腿角",
    "push_shank_angle_deg": "下栏蹬伸小腿角",
    "landing_knee_angle_deg": "下栏着地膝角",
    "max_buffer_frame": "下栏最大缓冲帧",
    "buffer_knee_angle_deg": "下栏缓冲膝角",
    "toeoff_knee_angle_deg": "恢复离地膝角",
    "buffer_amplitude_deg": "下栏缓冲幅度",
    "push_amplitude_deg": "下栏蹬伸幅度",
    "landing_trunk_angle_deg": "下栏落地躯干角",
    "toeoff_trunk_angle_deg": "恢复离地躯干角",
    "trunk_angle_change_deg": "躯干角变化量",
    "recovery_time_s": "恢复时间",
    "landing_phase_time_s": "下栏阶段时长",
    "total_phase_time_s": "总技术阶段时长",
    "takeoff_phase_ratio": "起跨阶段占比",
    "flight_phase_ratio": "腾空阶段占比",
    "landing_phase_ratio": "下栏阶段占比",
    "com_vx_landing_mps": "下栏后重心水平速度",
    "com_vy_landing_mps": "下栏后重心垂直速度",
}

IGNORE_KEYS = {"video_id", "sample_name", "source_video_name", "source_video_stem"}

EVENT_KEYS = {
    "takeoff_landing_frame",
    "takeoff_toeoff_frame",
    "bar_cross_frame",
    "knee_bar_cross_frame",
    "landing_frame",
    "recovery_toeoff_frame",
    "com_peak_frame",
    "takeoff_max_buffer_frame",
    "max_buffer_frame",
}

TIME_RATIO_KEYS = {
    "takeoff_support_time_s",
    "takeoff_phase_time_s",
    "flight_time_s",
    "toeoff_to_bar_time_s",
    "bar_to_landing_time_s",
    "recovery_time_s",
    "landing_phase_time_s",
    "total_phase_time_s",
    "toeoff_to_bar_ratio",
    "bar_to_landing_ratio",
    "takeoff_phase_ratio",
    "flight_phase_ratio",
    "landing_phase_ratio",
}

SPEED_KEYS = {
    "flight_com_speed_mps",
    "com_vx_pre_takeoff_mps",
    "com_vx_takeoff_toeoff_mps",
    "com_vx_change_mps",
    "com_vy_pre_takeoff_mps",
    "com_vy_takeoff_toeoff_mps",
    "com_vy_change_mps",
    "com_vx_landing_mps",
    "com_vy_landing_mps",
}

STAGE_LABELS = {
    "takeoff": "起跨",
    "flight": "腾空",
    "landing": "下栏",
    "overall": "整体",
}

PERCENT_KEYS = {
    "toeoff_to_bar_ratio",
    "bar_to_landing_ratio",
    "takeoff_phase_ratio",
    "flight_phase_ratio",
    "landing_phase_ratio",
}


def _embed_file(target_path: str | Path | None, default_mime: str) -> str:
    if not target_path:
        return ""
    path = Path(target_path)
    if not path.exists():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or default_mime
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _embed_image(target_path: str | Path | None) -> str:
    return _embed_file(target_path, "image/png")


def _embed_video(target_path: str | Path | None) -> str:
    return _embed_file(target_path, "video/mp4")


def _metric_label(key: str) -> str:
    return KEY_LABELS.get(key, "未命名指标")


def _metric_unit(key: str, value: object) -> str:
    if key in PERCENT_KEYS:
        return "%"
    if key.endswith("_deg"):
        return "deg"
    if key.endswith("_mps"):
        return "m/s"
    if key.endswith("_m"):
        return "m"
    if key.endswith("_s"):
        return "s"
    if key.endswith("_px") or key.endswith("_x") or key.endswith("_y"):
        return "px"
    if key.endswith("_frame") or key.endswith("_frames"):
        return "frame"
    if key == "fps":
        return "fps"
    if key == "pixel_to_meter_scale":
        return "m/px"
    if isinstance(value, str):
        return ""
    return ""


def _format_metric_value(key: str, value: object) -> str:
    if key == "move_dir":
        if str(value) == "1":
            return "向右"
        if str(value) == "-1":
            return "向左"
    if isinstance(value, float):
        number = float(value)
        if key in PERCENT_KEYS:
            return f"{number * 100:.2f}".rstrip("0").rstrip(".")
        return f"{number:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _display_metric_value(key: str, value: object) -> str:
    text = _format_metric_value(key, value)
    unit = _metric_unit(key, value)
    if unit:
        return f"{text} {unit}"
    return text


def _render_image_card(title: str, src: str) -> str:
    if not src:
        return ""
    return f"""
    <div class="media-card">
      <div class="media-title">{escape(title)}</div>
      <img src="{src}" alt="{escape(title)}">
    </div>
    """


def _render_video_card(title: str, src: str) -> str:
    if not src:
        return ""
    return f"""
    <div class="media-card media-video-card">
      <div class="media-title">{escape(title)}</div>
      <video controls preload="metadata" playsinline>
        <source src="{src}" type="video/mp4">
        当前浏览器不支持视频播放。
      </video>
    </div>
    """


def _collect_group_rows(
    metrics: dict[str, Any],
    used: set[str],
    title: str,
    predicate,
) -> tuple[str, list[tuple[str, str]]] | None:
    rows: list[tuple[str, str]] = []
    for key, value in metrics.items():
        if key in used or key in IGNORE_KEYS or value is None:
            continue
        if predicate(key):
            rows.append((_metric_label(key), _display_metric_value(key, value)))
            used.add(key)
    if not rows:
        return None
    return title, rows


def _metric_groups(metrics: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    used: set[str] = set()
    groups: list[tuple[str, list[tuple[str, str]]]] = []

    rules = [
        (
            "基础与标定",
            lambda key: key
            in {
                "fps",
                "total_frames",
                "move_dir",
                "leading_leg",
                "trail_leg",
                "post_x",
                "barmid_x",
                "barmid_y",
                "box_bottom_y",
                "far_base_top_y",
                "post_base_y",
                "post_height_pixel",
                "hurdle_box_width_px",
                "hurdle_box_height_px",
                "hurdle_box_ratio",
                "post_height_real",
                "hurdle_height_method",
                "hurdle_height_confidence",
                "pixel_to_meter_scale",
                "leg_length_px",
                "leg_length_m",
                "left_leg_length_px",
                "left_leg_length_m",
                "right_leg_length_px",
                "right_leg_length_m",
                "leading_leg_length_px",
                "leading_leg_length_m",
                "trail_leg_length_px",
                "trail_leg_length_m",
            },
        ),
        ("关键事件帧", lambda key: key in EVENT_KEYS),
        ("阶段时长与比例", lambda key: key in TIME_RATIO_KEYS),
        (
            "起跨参数",
            lambda key: key.startswith("takeoff_") and key not in EVENT_KEYS and key not in TIME_RATIO_KEYS,
        ),
        ("速度参数", lambda key: key in SPEED_KEYS),
        (
            "腾空参数",
            lambda key: (
                key.startswith("flight_")
                or key.startswith("bar_")
                or key.startswith("com_")
            )
            and key not in EVENT_KEYS
            and key not in TIME_RATIO_KEYS
            and key not in SPEED_KEYS,
        ),
        (
            "下栏参数",
            lambda key: (
                key.startswith("landing_")
                or key.startswith("recovery_")
                or key.startswith("buffer_")
                or key.startswith("push_")
                or key in {"toeoff_knee_angle_deg", "toeoff_trunk_angle_deg", "trunk_angle_change_deg"}
            )
            and key not in EVENT_KEYS
            and key not in TIME_RATIO_KEYS,
        ),
    ]

    for title, predicate in rules:
        group = _collect_group_rows(metrics, used, title, predicate)
        if group is not None:
            groups.append(group)

    extra_rows: list[tuple[str, str]] = []
    for key, value in metrics.items():
        if key in used or key in IGNORE_KEYS or value is None:
            continue
        extra_rows.append((_metric_label(key), _display_metric_value(key, value)))
    if extra_rows:
        groups.append(("其他参数", extra_rows))

    return groups


def _render_metric_groups(metrics: dict[str, Any]) -> str:
    cards = []
    for title, rows in _metric_groups(metrics):
        row_html = "".join(
            f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
            for label, value in rows
        )
        cards.append(
            f"""
            <div class="metric-card">
              <h3>{escape(title)}</h3>
              <table>
                <thead><tr><th>指标</th><th>数值</th></tr></thead>
                <tbody>{row_html or "<tr><td colspan='2'>暂无</td></tr>"}</tbody>
              </table>
            </div>
            """
        )
    return "".join(cards) or "<p>暂无</p>"


def _embed_html_document(target_path: str | Path | None) -> str:
    if not target_path:
        return ""
    path = Path(target_path)
    if not path.exists():
        return ""
    return escape(path.read_text(encoding="utf-8"), quote=True)


def render_html_report(diagnosis: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall = diagnosis.get("overall_score")
    overall_text = f"{float(overall['score']):.2f} / 5" if overall and overall.get("score") is not None else "暂无"
    weakest_stage = diagnosis.get("weakest_stage") or {}

    stage_rows = []
    for item in diagnosis.get("stage_diagnosis", []):
        stage_label = item.get("task_name") or item.get("stage_name") or item.get("stage")
        stage_rows.append(
            f"<tr><td>{escape(str(stage_label))}</td><td>{float(item['score']):.2f}</td><td>{escape(str(item['summary']))}</td></tr>"
        )
    stage_table_rows = "".join(stage_rows) or "<tr><td colspan='3'>暂无</td></tr>"

    problem_cards = []
    for idx, problem in enumerate(diagnosis.get("top_problems", []), start=1):
        evidence_items = "".join(
            f"<li>{escape(str(line))}</li>" for line in problem.get("evidence", [])
        ) or "<li>暂无直接指标证据。</li>"
        drill_items = "".join(
            f"<li>{escape(str(name))}</li>" for name in problem.get("recommended_drills", [])
        ) or "<li>暂无</li>"
        problem_cards.append(
            f"""
            <section class="problem-card">
              <h3>问题 {idx}：{escape(problem['problem_title'])}</h3>
              <p><strong>对应评分项：</strong>{escape(problem['task_id'])} {escape(problem['task_name'])}（{problem['score']:.2f} 分）</p>
              <p><strong>动作诊断：</strong>{escape(problem['diagnosis'])}</p>
              <p><strong>影响：</strong>{escape(problem['impact'])}</p>
              <div><strong>证据：</strong></div>
              <ul>{evidence_items}</ul>
              <div><strong>推荐练习：</strong></div>
              <ul>{drill_items}</ul>
            </section>
            """
        )
    problem_section = "".join(problem_cards) or "<section class='problem-card'><p>当前没有明显低分问题，可继续保持现有训练节奏。</p></section>"

    drill_list = "".join(f"<li>{escape(str(line))}</li>" for line in diagnosis.get("recommended_drills", [])) or "<li>暂无</li>"
    next_focus_list = "".join(f"<li>{escape(str(line))}</li>" for line in diagnosis.get("next_focus", [])) or "<li>暂无</li>"
    top_problem_titles = "".join(
        f"<span class='summary-chip'>{escape(str(problem.get('problem_title', '')))}</span>"
        for problem in diagnosis.get("top_problems", [])[:3]
        if problem.get("problem_title")
    ) or "<span class='summary-chip'>暂无明显突出问题</span>"

    module6_artifacts = diagnosis.get("module6_artifacts") or {}
    chart_src = _embed_image(module6_artifacts.get("chart_png"))
    trajectory_src = _embed_image(diagnosis.get("trajectory_image") or module6_artifacts.get("trajectory_png"))
    speed_src = _embed_image(diagnosis.get("speed_image"))
    module5_summary_src = _embed_image(diagnosis.get("module5_summary_image"))
    module5_video_src = _embed_video(diagnosis.get("module5_video_path"))
    module6_report_srcdoc = _embed_html_document(diagnosis.get("score_report_html") or module6_artifacts.get("report_html"))
    speed_card_title = "重心速度变化图" if speed_src else "模块五综合特征图（含重心速度）"

    score_card_media = _render_image_card("评分总览", chart_src) if chart_src else "<p>暂无评分图。</p>"
    module6_toggle = (
        f"""
        <div class="toggle-wrap">
          <button class="toggle-btn" type="button" onclick="togglePanel('module6-report-panel', this, '展开完整评分报告', '收起完整评分报告')">展开完整评分报告</button>
        </div>
        <div id="module6-report-panel" class="toggle-panel">
          <iframe class="report-frame" loading="lazy" srcdoc="{module6_report_srcdoc}"></iframe>
        </div>
        """
        if module6_report_srcdoc
        else "<p class='meta'>当前没有可用的完整评分报告。</p>"
    )

    media_cards = "".join(
        [
            _render_image_card("评分总览", chart_src),
            _render_image_card("重心轨迹图", trajectory_src),
            _render_image_card(speed_card_title, speed_src or module5_summary_src),
        ]
    )
    media_section = (
        f"""
        <section class="card" style="margin-top: 20px;">
          <h2>图表与轨迹</h2>
          <div class="media-grid">{media_cards}</div>
        </section>
        """
        if media_cards
        else ""
    )

    screenshot_cards = []
    for item in diagnosis.get("screenshots", []):
        image_src = _embed_image(item.get("path"))
        if not image_src:
            continue
        screenshot_cards.append(
            f"""
            <div class="shot-card">
              <img src="{image_src}" alt="{escape(str(item.get('label', '关键帧')))}">
              <div class="shot-meta">{escape(str(item.get("label", "关键帧")))} / 帧 {escape(str(item.get("frame_index", "-")))}</div>
            </div>
            """
        )
    screenshot_section = (
        f"""
        <section class="card" style="margin-top: 20px;">
          <h2>关键时刻截图</h2>
          <div class="shot-grid">{"".join(screenshot_cards) or "<p>暂无</p>"}</div>
        </section>
        """
        if screenshot_cards
        else ""
    )

    metric_rows = []
    for item in diagnosis.get("technical_metrics_highlights", []):
        value = item.get("value")
        if value is None:
            continue
        unit = str(item.get("unit", "")).strip()
        value_text = _format_metric_value(str(item.get("key", "")), value)
        if unit:
            value_text = f"{value_text} {unit}"
        metric_rows.append(
            f"<tr><td>{escape(STAGE_LABELS.get(str(item.get('stage', '')), str(item.get('stage', ''))))}</td><td>{escape(str(item.get('label', item.get('key', ''))))}</td><td>{escape(value_text)}</td></tr>"
        )
    metric_section = f"""
    <section class="card" style="margin-top: 20px;">
      <h2>关键技术指标摘要</h2>
      <table>
        <thead><tr><th>阶段</th><th>指标</th><th>数值</th></tr></thead>
        <tbody>{"".join(metric_rows) or "<tr><td colspan='3'>暂无</td></tr>"}</tbody>
      </table>
    </section>
    """

    full_metric_section = f"""
    <section class="card" style="margin-top: 20px;">
      <h2>全部指标</h2>
      <div class="metric-grid">{_render_metric_groups(diagnosis.get("all_metrics") or {})}</div>
    </section>
    """

    detail_toolbar = """
    <div class="toggle-row">
      <button class="toggle-btn secondary small" type="button" onclick="togglePanel('diagnosis-panel', this, '展开问题诊断', '收起问题诊断')">展开问题诊断</button>
      <button class="toggle-btn secondary small" type="button" onclick="togglePanel('advice-panel', this, '展开训练建议', '收起训练建议')">展开训练建议</button>
      <button class="toggle-btn secondary small" type="button" onclick="togglePanel('metrics-panel', this, '展开指标信息', '收起指标信息')">展开指标信息</button>
      <button class="toggle-btn secondary small" type="button" onclick="togglePanel('shots-panel', this, '展开关键截图', '收起关键截图')">展开关键截图</button>
      <button class="toggle-btn secondary small" type="button" onclick="togglePanel('charts-panel', this, '展开图表轨迹', '收起图表轨迹')">展开图表轨迹</button>
    </div>
    """

    detail_sections = f"""
    <section id="diagnosis-panel" class="toggle-panel">
      <div class="grid" style="margin-top: 20px;">
        <div class="card">
          <h2>分阶段诊断</h2>
          <table>
            <thead><tr><th>阶段</th><th>得分</th><th>结论</th></tr></thead>
            <tbody>{stage_table_rows}</tbody>
          </table>
        </div>
        <div class="card">
          <h2>下次训练重点</h2>
          <ul>{next_focus_list}</ul>
        </div>
      </div>
      <section class="stack" style="margin-top: 20px;">
        {problem_section}
      </section>
    </section>

    <section id="advice-panel" class="toggle-panel">
      <section class="card" style="margin-top: 20px;">
        <h2>训练建议</h2>
        <ul>{drill_list}</ul>
      </section>
    </section>

    <section id="metrics-panel" class="toggle-panel">
      {metric_section}
      {full_metric_section}
    </section>

    <section id="shots-panel" class="toggle-panel">
      {screenshot_section}
    </section>

    <section id="charts-panel" class="toggle-panel">
      {media_section}
    </section>
    """

    base_styles = """
    body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; margin: 28px; color: #1f2937; background: #f7f9fc; }
    h1, h2, h3 { margin: 0 0 10px; }
    .hero, .card, .problem-card { background: #fff; border: 1px solid #d9e2ec; border-radius: 16px; padding: 20px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05); }
    .hero { margin-bottom: 20px; }
    .hero-grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 20px; align-items: start; }
    .hero-copy p { max-width: 720px; }
    .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; margin-bottom: 20px; }
    .stack { display: grid; gap: 16px; }
    .section-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
    .summary-chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .summary-chip { display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; background: #f3f6fb; color: #334155; font-size: 14px; font-weight: 600; border: 1px solid #dbe5f0; }
    .media-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .media-card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }
    .media-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #344054; }
    .media-card img, .media-card video { width: 100%; display: block; border-radius: 10px; }
    .shot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .shot-card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; }
    .shot-card img { width: 100%; display: block; border-radius: 10px; }
    .shot-meta { margin-top: 8px; font-size: 13px; color: #475467; }
    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .metric-card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; vertical-align: top; }
    ul { margin: 8px 0 0 18px; }
    p { margin: 8px 0; line-height: 1.65; }
    .score { font-size: 34px; font-weight: 700; color: #1d4ed8; }
    .meta { color: #475467; }
    .summary-score-card { margin-top: 20px; background: #f8fafc; border: 1px solid #dbe5f0; border-radius: 14px; padding: 16px; }
    .toggle-wrap { margin-top: 16px; }
    .toggle-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }
    .toggle-btn { display: inline-flex; align-items: center; justify-content: center; min-width: 210px; padding: 12px 18px; border-radius: 999px; background: #1d4ed8; color: #fff; font-weight: 700; border: 1px solid #1d4ed8; cursor: pointer; box-shadow: 0 8px 16px rgba(29, 78, 216, 0.18); }
    .toggle-btn.secondary { background: #e8eefc; color: #1d4ed8; border-color: #c7d7fe; box-shadow: none; }
    .toggle-btn.small { min-width: 160px; padding: 10px 16px; }
    .toggle-panel { display: none; margin-top: 18px; }
    .toggle-panel.is-open { display: block; }
    .report-frame { width: 100%; min-height: 1500px; border: 1px solid #d9e2ec; border-radius: 14px; background: #fff; }
    @media (max-width: 960px) {
      .hero-grid, .grid, .media-grid, .shot-grid, .metric-grid { grid-template-columns: 1fr; }
      .section-head { display: block; }
    }
    """

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(diagnosis['sample_name'])} 动作诊断与训练建议</title>
  <style>{base_styles}</style>
</head>
<body>
  <section class="hero">
    <div class="hero-grid">
      <div class="hero-copy">
        <h1>{escape(diagnosis['sample_name'])} 动作诊断与训练建议</h1>
        <p class="meta">视频：{escape(str(diagnosis['source_video_name']))}</p>
        <div class="score">总评分：{overall_text}</div>
        <p><strong>总体判断：</strong>{escape(diagnosis['overall_summary'])}</p>
        <p><strong>当前最弱阶段：</strong>{escape(str(weakest_stage.get('task_name', weakest_stage.get('stage_name', '暂无'))))}</p>
        <p><strong>当前重点：</strong></p>
        <div class="summary-chips">{top_problem_titles}</div>
        <div class="summary-score-card">
          <h2>评分报告</h2>
          {score_card_media}
          {module6_toggle}
        </div>
      </div>
      <div>
        {_render_video_card("动作视频", module5_video_src) if module5_video_src else "<div class='media-card'><div class='media-title'>动作视频</div><p>当前没有可用的视频文件。</p></div>"}
      </div>
    </div>
  </section>

  <section class="card" style="margin-top: 20px;">
    <h2>详细报告</h2>
    <p class="meta">按内容分类展开查看问题诊断、训练建议、指标、截图和图表。</p>
    {detail_toolbar}
    {detail_sections}
  </section>

  <script>
    function togglePanel(id, button, expandLabel, collapseLabel) {{
      const panel = document.getElementById(id);
      if (!panel) return;
      const isOpen = panel.classList.toggle('is-open');
      button.textContent = isOpen ? collapseLabel : expandLabel;
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
