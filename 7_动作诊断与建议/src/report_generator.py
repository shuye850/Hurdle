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

LEG_LENGTH_KEYS = {
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
    "bar_cross_leading_leg_length_px",
    "bar_cross_leading_leg_length_m",
    "bar_cross_trail_leg_length_px",
    "bar_cross_trail_leg_length_m",
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


def _is_visible_metric_key(key: str) -> bool:
    return key not in LEG_LENGTH_KEYS


def _render_image_card(title: str, src: str, card_class: str = "") -> str:
    if not src:
        return ""
    class_attr = "media-card"
    if card_class:
        class_attr = f"{class_attr} {card_class}"
    return f"""
    <div class="{class_attr}">
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
        if key in used or key in IGNORE_KEYS or value is None or not _is_visible_metric_key(key):
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
        if key in used or key in IGNORE_KEYS or value is None or not _is_visible_metric_key(key):
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


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _source_class(current: Any, original: Any) -> str:
    current_text = _normalize_text(current)
    original_text = _normalize_text(original)
    if not current_text:
        return "text-rule"
    if original_text and current_text != original_text:
        return "text-llm"
    return "text-rule"


def _colored_text(current: Any, original: Any) -> str:
    text = _normalize_text(current) or _normalize_text(original) or "暂无"
    return f"<span class=\"{_source_class(current, original)}\">{escape(text)}</span>"


def _colored_list(items: list[Any], original_items: list[Any] | None = None) -> str:
    original_items = original_items or []
    original_set = {_normalize_text(item) for item in original_items if _normalize_text(item)}
    rows = []
    for item in items:
        text = _normalize_text(item)
        if not text:
            continue
        css = "text-llm" if text not in original_set else "text-rule"
        rows.append(f"<li><span class=\"{css}\">{escape(text)}</span></li>")
    return "".join(rows) or "<li><span class=\"text-rule\">暂无</span></li>"


def render_html_report(diagnosis: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall = diagnosis.get("overall_score")
    overall_text = f"{float(overall['score']):.2f} / 5" if overall and overall.get("score") is not None else "暂无"
    weakest_stage = diagnosis.get("weakest_stage") or {}

    rule_stage_map = {
        str(item.get("stage", "")).strip(): item
        for item in diagnosis.get("rule_based_stage_diagnosis", [])
        if str(item.get("stage", "")).strip()
    }

    stage_rows = []
    for item in diagnosis.get("stage_diagnosis", []):
        stage_label = item.get("task_name") or item.get("stage_name") or item.get("stage")
        rule_item = rule_stage_map.get(str(item.get("stage", "")).strip(), {})
        stage_rows.append(
            f"<tr><td>{escape(str(stage_label))}</td><td>{float(item['score']):.2f}</td><td>{_colored_text(item.get('summary'), rule_item.get('summary'))}</td></tr>"
        )
    stage_table_rows = "".join(stage_rows) or "<tr><td colspan='3'>暂无</td></tr>"

    rule_problem_map = {
        str(item.get("task_id", "")).strip(): item
        for item in diagnosis.get("rule_based_top_problems", [])
        if str(item.get("task_id", "")).strip()
    }

    problem_cards = []
    for idx, problem in enumerate(diagnosis.get("top_problems", []), start=1):
        rule_problem = rule_problem_map.get(str(problem.get("task_id", "")).strip(), {})
        evidence_items = "".join(
            f"<li>{escape(str(line))}</li>" for line in problem.get("evidence", [])
        ) or "<li><span class=\"text-rule\">暂无直接指标证据。</span></li>"
        drill_items = _colored_list(problem.get("recommended_drills", []), rule_problem.get("recommended_drills", []))
        problem_cards.append(
            f"""
            <section class="problem-card">
              <h3>问题 {idx}：{_colored_text(problem.get('problem_title'), rule_problem.get('problem_title'))}</h3>
              <p><strong>对应评分项：</strong>{escape(problem['task_id'])} {escape(problem['task_name'])}（{problem['score']:.2f} 分）</p>
              <p><strong>动作诊断：</strong>{_colored_text(problem.get('diagnosis'), rule_problem.get('diagnosis'))}</p>
              <p><strong>影响：</strong>{_colored_text(problem.get('impact'), rule_problem.get('impact'))}</p>
              <div><strong>证据：</strong></div>
              <ul>{evidence_items}</ul>
              <div><strong>推荐练习：</strong></div>
              <ul>{drill_items}</ul>
              <p><strong>下次关注：</strong>{_colored_text(problem.get('next_focus'), rule_problem.get('next_focus'))}</p>
            </section>
            """
        )
    problem_section = "".join(problem_cards) or "<section class='problem-card'><p>当前没有明显低分问题，可继续保持现有训练节奏。</p></section>"

    drill_list = _colored_list(diagnosis.get("recommended_drills", []), diagnosis.get("rule_based_recommended_drills", []))
    next_focus_list = _colored_list(diagnosis.get("next_focus", []), diagnosis.get("rule_based_next_focus", []))
    top_problem_titles = "".join(
        f"<span class='summary-chip {_source_class(problem.get('problem_title'), rule_problem_map.get(str(problem.get('task_id', '')).strip(), {}).get('problem_title'))}'>{escape(str(problem.get('problem_title', '')))}</span>"
        for problem in diagnosis.get("top_problems", [])[:3]
        if problem.get("problem_title")
    ) or "<span class='summary-chip'>暂无明显突出问题</span>"

    module6_artifacts = diagnosis.get("module6_artifacts") or {}
    chart_src = _embed_image(module6_artifacts.get("chart_png"))
    trajectory_src = _embed_image(diagnosis.get("trajectory_image") or module6_artifacts.get("trajectory_png"))
    speed_src = _embed_image(diagnosis.get("speed_image"))
    phase_timeline_src = _embed_image(diagnosis.get("phase_timeline_image")) or _embed_image(diagnosis.get("module5_summary_image"))
    module5_video_src = _embed_video(diagnosis.get("module5_video_path"))
    module6_report_srcdoc = _embed_html_document(diagnosis.get("score_report_html") or module6_artifacts.get("report_html"))

    score_card_media = _render_image_card("评分总览", chart_src) if chart_src else "<p>暂无评分图。</p>"
    module6_toggle = (
        f"""
        <details class="details-block">
          <summary>查看完整评分明细</summary>
          <iframe class="report-frame" loading="lazy" srcdoc="{module6_report_srcdoc}"></iframe>
        </details>
        """
        if module6_report_srcdoc
        else "<p class='meta'>当前没有可用的完整评分报告。</p>"
    )

    media_top_cards = "".join(
        [
            _render_image_card("重心轨迹图", trajectory_src),
            _render_image_card("重心速度变化图", speed_src),
        ]
    )
    phase_card = _render_image_card("阶段划分与关键事件定位", phase_timeline_src, "phase-card phase-row")
    media_section = (
        f"""
        <section class="card" style="margin-top: 20px;">
          <h2>图表与轨迹</h2>
          <div class="media-grid">{media_top_cards}</div>
          {phase_card}
        </section>
        """
        if media_top_cards or phase_card
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
        key = str(item.get("key", ""))
        if key and not _is_visible_metric_key(key):
            continue
        value = item.get("value")
        if value is None:
            continue
        unit = str(item.get("unit", "")).strip()
        value_text = _format_metric_value(key, value)
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

    detail_sections = f"""
    <section id="diagnosis" class="report-section">
      <div class="section-heading">
        <div><span class="section-kicker">01 / DIAGNOSIS</span><h2>分阶段诊断</h2></div>
        <p>先确认主要短板，再查看具体证据和影响。</p>
      </div>
      <div class="grid">
        <div class="card">
          <h3>阶段评分与结论</h3>
          <table>
            <thead><tr><th>阶段</th><th>得分</th><th>结论</th></tr></thead>
            <tbody>{stage_table_rows}</tbody>
          </table>
        </div>
        <div class="card">
          <h3>下次训练重点</h3>
          <ul>{next_focus_list}</ul>
        </div>
      </div>
      <section class="stack">
        {problem_section}
      </section>
    </section>

    <section id="advice" class="report-section">
      <div class="section-heading">
        <div><span class="section-kicker">02 / TRAINING</span><h2>训练建议</h2></div>
        <p>按优先级执行，下次测试时重点复核。</p>
      </div>
      <section class="card advice-summary">
        <ul>{drill_list}</ul>
      </section>
    </section>

    <section id="metrics" class="report-section">
      <div class="section-heading">
        <div><span class="section-kicker">03 / METRICS</span><h2>技术指标</h2></div>
        <p>关键数值默认显示，全部参数可按需展开。</p>
      </div>
      {metric_section}
      <details class="details-block">
        <summary>展开全部运动学指标</summary>
        {full_metric_section}
      </details>
    </section>

    <section id="shots" class="report-section">
      <div class="section-heading">
        <div><span class="section-kicker">04 / EVIDENCE</span><h2>关键帧证据</h2></div>
        <p>对照起跨、过栏和下栏时刻检查动作。</p>
      </div>
      {screenshot_section}
    </section>

    <section id="charts" class="report-section">
      <div class="section-heading">
        <div><span class="section-kicker">05 / CHARTS</span><h2>图表与轨迹</h2></div>
        <p>用时序图检查重心、速度和阶段节奏。</p>
      </div>
      {media_section}
    </section>
    """

    base_styles = """
    :root { --fg:#1f2328; --muted:#59636e; --canvas:#fff; --subtle:#f6f8fa; --border:#d1d9e0; --blue:#0969da; --green:#1f883d; --orange:#9a6700; --dark:#25292e; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; background: var(--subtle); }
    body { margin:0; color:var(--fg); background:var(--subtle); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Helvetica,Arial,sans-serif; font-size:14px; line-height:1.55; }
    h1,h2,h3,p { margin-top:0; } h1 { font-size:28px; line-height:1.25; letter-spacing:-.025em; } h2 { font-size:20px; } h3 { font-size:14px; }
    .site-header { height:58px; display:flex; align-items:center; justify-content:space-between; padding:0 28px; color:#f0f6fc; background:var(--dark); border-bottom:1px solid #3d444d; }
    .site-brand { display:flex; align-items:center; gap:10px; font-weight:600; }.site-mark { width:30px; height:30px; display:grid; place-items:center; border:1px solid #57606a; border-radius:6px; background:#151b23; font:700 12px var(--mono); }.site-state { color:#7ee787; font:11px var(--mono); }
    .page-layout { width:min(1440px,100%); margin:auto; display:grid; grid-template-columns:235px minmax(0,1fr); gap:24px; padding:24px; }
    .side-nav { position:sticky; top:24px; align-self:start; overflow:hidden; border:1px solid var(--border); border-radius:6px; background:var(--canvas); }
    .side-nav-title { padding:14px 16px; border-bottom:1px solid var(--border); background:var(--subtle); font-weight:600; }.side-nav a { display:block; padding:9px 16px; color:var(--fg); border-left:3px solid transparent; text-decoration:none; font-size:12px; }.side-nav a:hover { color:var(--blue); border-left-color:var(--blue); background:var(--subtle); }.side-meta { margin:10px 16px 14px; padding-top:12px; color:var(--muted); border-top:1px solid var(--border); font:10px/1.6 var(--mono); }
    .report-main { min-width:0; }.hero,.report-section { scroll-margin-top:20px; background:var(--canvas); border:1px solid var(--border); border-radius:6px; }.hero { overflow:hidden; }.hero-heading { padding:20px 22px; border-bottom:1px solid var(--border); }.meta { color:var(--muted); font-size:12px; }.summary-grid { display:grid; grid-template-columns:190px minmax(0,1fr); }.score-panel { padding:22px; color:#f0f6fc; background:var(--dark); }.score-label { color:#9198a1; font:10px var(--mono); }.score { margin:8px 0 16px; font:700 42px/1 var(--mono); }.score small { color:#9198a1; font-size:13px; }.weak-label { color:#b7bdc8; font-size:11px; }.weak-value { display:block; margin-top:4px; color:#fff; font-weight:600; }.summary-copy { padding:22px; }.summary-copy h2 { margin-bottom:8px; }.summary-copy p { margin-bottom:14px; }.summary-chips { display:flex; flex-wrap:wrap; gap:7px; }.summary-chip { display:inline-flex; padding:4px 9px; color:#82071e; border:1px solid #ff8182; border-radius:2em; background:#ffebe9; font-size:11px; font-weight:600; }
    .report-section { margin-top:18px; padding:20px; }.section-heading { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; margin:-20px -20px 18px; padding:16px 20px; border-bottom:1px solid var(--border); background:var(--subtle); }.section-heading h2 { margin:3px 0 0; }.section-heading p { max-width:430px; margin:0; color:var(--muted); text-align:right; font-size:11px; }.section-kicker { color:var(--blue); font:600 10px var(--mono); }
    .grid { display:grid; grid-template-columns:minmax(0,1.3fr) minmax(260px,.7fr); gap:14px; }.stack { display:grid; gap:12px; margin-top:14px; }.card,.problem-card { padding:16px; border:1px solid var(--border); border-radius:6px; background:var(--canvas); }.problem-card { border-left:4px solid #fd8c73; }.problem-card h3 { margin-bottom:10px; }.problem-card strong { color:#343a40; }.problem-card ul,.advice-summary ul { padding-left:20px; }.problem-card li,.advice-summary li { margin:6px 0; }.advice-summary { border-left:4px solid var(--green); }
    table { width:100%; border-collapse:collapse; }.card table { margin:0 -16px -16px; width:calc(100% + 32px); }.card h3+table { margin-top:12px; } th,td { padding:10px 12px; border-top:1px solid var(--border); text-align:left; vertical-align:top; font-size:12px; } th { color:var(--muted); background:var(--subtle); font:600 10px var(--mono); } tbody tr:hover { background:#f6f8fa; }
    .media-grid,.shot-grid,.metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }.media-card,.shot-card,.metric-card { padding:10px; border:1px solid var(--border); border-radius:6px; background:var(--subtle); }.media-card.phase-row { margin-top:12px; }.media-title { margin-bottom:7px; color:var(--muted); font:600 10px var(--mono); }.media-card img,.media-card video,.shot-card img { width:100%; display:block; border-radius:4px; }.shot-meta { margin-top:7px; color:var(--muted); font:10px var(--mono); }.metric-card h3 { margin:2px 2px 10px; }.metric-card table { background:#fff; }.text-rule,.text-llm { color:inherit; }
    .summary-score-card { margin-top:16px; padding:14px; border:1px solid var(--border); border-radius:6px; background:var(--subtle); }.details-block { margin-top:14px; border:1px solid var(--border); border-radius:6px; background:#fff; }.details-block>summary { padding:11px 14px; color:var(--blue); background:var(--subtle); cursor:pointer; font-weight:600; }.details-block[open]>summary { border-bottom:1px solid var(--border); }.details-block .report-section,.details-block .card { margin:12px; }.report-frame { width:calc(100% - 24px); min-height:900px; margin:12px; border:1px solid var(--border); border-radius:4px; background:#fff; }
    @media(max-width:900px){.page-layout{grid-template-columns:1fr;padding:12px}.side-nav{position:static}.side-nav a{display:inline-block;border-left:0;border-bottom:2px solid transparent}.side-nav a:hover{border-bottom-color:var(--blue)}.side-meta{display:none}.summary-grid,.grid,.media-grid,.shot-grid,.metric-grid{grid-template-columns:1fr}.section-heading{display:block}.section-heading p{margin-top:6px;text-align:left}.site-header{padding:0 14px}.site-state{display:none}}
    @media print{.site-header,.side-nav{display:none}.page-layout{display:block;padding:0}.report-section,.hero{break-inside:avoid;border-color:#bbb}.details-block>div{display:block}}
    """

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(diagnosis['sample_name'])} 动作诊断与训练建议</title>
  <style>{base_styles}</style>
</head>
<body>
  <header class="site-header">
    <div class="site-brand"><span class="site-mark">HR</span><span>跨栏动作技术分析</span></div>
    <span class="site-state">RULE-BASED REPORT</span>
  </header>
  <div class="page-layout">
    <aside class="side-nav" aria-label="报告目录">
      <div class="side-nav-title">报告目录</div>
      <a href="#summary">分析摘要</a>
      <a href="#video">动作与评分</a>
      <a href="#diagnosis">分阶段诊断</a>
      <a href="#advice">训练建议</a>
      <a href="#metrics">技术指标</a>
      <a href="#shots">关键帧证据</a>
      <a href="#charts">图表与轨迹</a>
      <div class="side-meta">SYSTEM / HURDLE-08<br>MODE / NO LLM<br>SCALE / 0–5</div>
    </aside>
    <main class="report-main">
      <section class="hero" id="summary">
        <div class="hero-heading">
        <h1>{escape(diagnosis['sample_name'])} 动作诊断与训练建议</h1>
        <p class="meta">视频：{escape(str(diagnosis['source_video_name']))}</p>
        </div>
        <div class="summary-grid">
          <div class="score-panel">
            <span class="score-label">OVERALL SCORE</span>
            <div class="score">{overall_text.replace(' / 5', '<small> / 5</small>')}</div>
            <span class="weak-label">优先改善<span class="weak-value">{escape(str(weakest_stage.get('task_name', weakest_stage.get('stage_name', '暂无'))))}</span></span>
          </div>
          <div class="summary-copy">
            <h2>总体判断</h2>
            <p>{_colored_text(diagnosis.get('overall_summary'), diagnosis.get('rule_based_overall_summary'))}</p>
            <h3>当前重点</h3>
            <div class="summary-chips">{top_problem_titles}</div>
          </div>
        </div>
      </section>

      <section class="report-section" id="video">
        <div class="section-heading">
          <div><span class="section-kicker">00 / OVERVIEW</span><h2>动作与评分</h2></div>
          <p>视频用于复核动作，评分图用于快速定位薄弱项。</p>
        </div>
        <div class="media-grid">
          {_render_video_card("动作视频", module5_video_src) if module5_video_src else "<div class='media-card'><div class='media-title'>动作视频</div><p>当前没有可用的视频文件。</p></div>"}
          <div class="summary-score-card"><h3>评分总览</h3>{score_card_media}{module6_toggle}</div>
        </div>
      </section>

      {detail_sections}
    </main>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
