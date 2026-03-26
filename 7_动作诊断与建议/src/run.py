from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from diagnosis_engine import build_diagnosis, build_reference_profiles, build_summary_frame
from io_utils import (
    DEFAULT_EVENT_CSV_DIR,
    DEFAULT_FRAME_DIR,
    DEFAULT_FUSION_VIDEO_DIR,
    DEFAULT_MODULE5_VIS_DIR,
    DEFAULT_MODULE6_HISTORY_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREPROCESS_VIDEO_DIR,
    DEFAULT_SCORE_JSON,
    DEFAULT_TECHNICAL_DIR,
    build_records_from_legacy_inputs,
    ensure_output_dirs,
    get_metrics_row,
    get_score_payload,
    load_module6_history,
    load_metrics_map,
    load_score_result_directory,
    load_score_results,
    match_video_id,
    save_json,
)
from media_utils import (
    extract_key_event_screenshots,
    load_event_row,
    load_frame_features,
    render_com_trajectory_image,
    render_com_velocity_image,
    resolve_module5_summary_image,
    resolve_module5_video,
    resolve_video_path,
)
from llm_writer import enrich_diagnosis_with_llm
from report_generator import render_html_report


def _record_key(record: dict[str, object]) -> str:
    video_id = str(record.get("video_id", "")).strip()
    if video_id:
        return video_id
    payload = record.get("score_payload")
    if isinstance(payload, dict):
        sample_name = str(payload.get("sample_name", "")).strip()
        if sample_name:
            return sample_name
    sample_name = str(record.get("sample_name", "")).strip()
    return sample_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模块七：动作诊断与建议")
    parser.add_argument("--module6-history-json", default=DEFAULT_MODULE6_HISTORY_JSON)
    parser.add_argument("--score-json", default=DEFAULT_SCORE_JSON)
    parser.add_argument("--technical-dir", default=DEFAULT_TECHNICAL_DIR)
    parser.add_argument("--frame-dir", default=DEFAULT_FRAME_DIR)
    parser.add_argument("--event-dir", default=DEFAULT_EVENT_CSV_DIR)
    parser.add_argument("--fusion-video-dir", default=DEFAULT_FUSION_VIDEO_DIR)
    parser.add_argument("--preprocess-video-dir", default=DEFAULT_PREPROCESS_VIDEO_DIR)
    parser.add_argument("--module5-vis-dir", default=DEFAULT_MODULE5_VIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", default=None, help="只处理指定 video_id")
    parser.add_argument("--enable-llm", action="store_true", help="启用 LLM 生成问题和训练建议文案")
    parser.add_argument("--llm-backend", default="llama_cpp", choices=["llama_cpp", "openai"], help="LLM 后端")
    parser.add_argument("--llm-model", default=None, help="LLM 模型名，默认读取 OPENAI_MODEL")
    parser.add_argument("--llm-base-url", default=None, help="LLM 接口地址，默认读取 OPENAI_BASE_URL")
    parser.add_argument("--llm-api-key", default=None, help="LLM 接口密钥，默认读取 OPENAI_API_KEY")
    parser.add_argument("--llama-python-path", default=None, help="独立 llama.cpp 环境里的 Python 路径")
    parser.add_argument("--llama-model-path", default=None, help="本地 llama.cpp GGUF 模型路径，默认读取 LLAMA_MODEL_PATH")
    return parser.parse_args()


def _cleanup_media_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.output_dir)

    print(f"[模块七] 优先读取模块六历史样本库: {args.module6_history_json}")
    history_records = load_module6_history(Path(args.module6_history_json))
    if history_records:
        print(f"[模块七] 已加载模块六历史样本: {len(history_records)} 条")

    legacy_records = []
    score_json_path = Path(args.score_json)
    technical_dir_path = Path(args.technical_dir)
    if score_json_path.exists() and technical_dir_path.exists():
        print(f"[模块七] 补充读取历史评分结果: {args.score_json}")
        score_payload_map: dict[str, dict[str, object]] = {}
        for payload in load_score_result_directory(score_json_path.parent):
            key = str(payload.get("video_id", "")).strip() or str(payload.get("sample_name", "")).strip()
            if key:
                score_payload_map[key] = payload
        for payload in load_score_results(score_json_path):
            key = str(payload.get("video_id", "")).strip() or str(payload.get("sample_name", "")).strip()
            if key:
                score_payload_map[key] = payload
        print(f"[模块七] 补充读取历史技术指标目录: {args.technical_dir}")
        metrics_map = load_metrics_map(technical_dir_path)
        legacy_records = build_records_from_legacy_inputs(list(score_payload_map.values()), metrics_map)

    merged_records: dict[str, dict[str, object]] = {}
    for record in legacy_records:
        key = _record_key(record)
        if key:
            merged_records[key] = record
    for record in history_records:
        key = _record_key(record)
        if key:
            merged_records[key] = record
    library_records = [merged_records[key] for key in sorted(merged_records)]

    if not library_records:
        print("[模块七] 未找到可用输入数据，结束。")
        return

    print(f"[模块七] 当前可用于诊断的样本总数: {len(library_records)} 条")

    reference_profiles = build_reference_profiles(library_records)

    diagnoses = []
    for record in library_records:
        payload = get_score_payload(record)
        if not payload:
            continue
        if not match_video_id(str(payload.get("video_id", "")), args.video_id):
            continue

        metrics_row = get_metrics_row(record)
        if not metrics_row:
            continue

        diagnosis = build_diagnosis(payload, metrics_row, reference_profiles)
        diagnosis = enrich_diagnosis_with_llm(
            diagnosis,
            enabled=bool(args.enable_llm),
            backend=args.llm_backend,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            llama_python_path=args.llama_python_path,
            llama_model_path=args.llama_model_path,
        )
        if args.enable_llm:
            print(f"[模块七] {diagnosis['sample_name']} LLM 文案状态: {diagnosis.get('llm_status')}")
        artifacts = record.get("artifacts") or {}
        payload_artifacts = payload.get("artifacts") or {}
        merged_artifacts = {}
        if isinstance(artifacts, dict):
            merged_artifacts.update(artifacts)
        if isinstance(payload_artifacts, dict):
            merged_artifacts.update(payload_artifacts)

        diagnosis["module6_artifacts"] = merged_artifacts
        diagnosis["technical_metrics_highlights"] = payload.get("technical_metrics", [])
        diagnosis["event_frames"] = payload.get("event_frames", [])
        diagnosis["score_report_html"] = merged_artifacts.get("report_html")

        stem = diagnosis["sample_name"]
        video_id = str(payload.get("video_id", ""))
        event_row = load_event_row(Path(args.event_dir), stem, video_id)
        frame_df = load_frame_features(Path(args.frame_dir), stem, video_id)
        video_path = resolve_video_path(
            sample_name=stem,
            source_video_name=str(payload.get("source_video_name", "")),
            fusion_video_dir=Path(args.fusion_video_dir),
            preprocess_video_dir=Path(args.preprocess_video_dir),
        )

        screenshot_dir = output_dirs["screenshots"] / stem
        screenshots = extract_key_event_screenshots(
            video_path=video_path,
            event_row=event_row,
            output_dir=screenshot_dir,
            sample_name=stem,
        )
        trajectory_path = render_com_trajectory_image(
            video_path=video_path,
            frame_df=frame_df,
            event_row=event_row,
            output_path=output_dirs["visualization"] / f"{stem}_com_trajectory.png",
            sample_name=stem,
        )
        speed_path = render_com_velocity_image(
            frame_df=frame_df,
            event_row=event_row,
            output_path=output_dirs["visualization"] / f"{stem}_com_speed.png",
            sample_name=stem,
        )
        module5_summary_path = resolve_module5_summary_image(Path(args.module5_vis_dir), stem, video_id)
        module5_video_path = resolve_module5_video(Path(args.fusion_video_dir), stem, video_id)

        diagnosis["screenshots"] = screenshots
        diagnosis["trajectory_image"] = str(trajectory_path) if trajectory_path is not None else ""
        diagnosis["speed_image"] = str(speed_path) if speed_path is not None else ""
        diagnosis["module5_summary_image"] = str(module5_summary_path) if module5_summary_path is not None else ""
        diagnosis["module5_video_path"] = str(module5_video_path) if module5_video_path is not None else ""

        render_html_report(diagnosis, output_dirs["reports"] / f"{stem}_advice.html")

        diagnosis_for_json = dict(diagnosis)
        diagnosis_for_json["screenshots"] = [
            {
                "frame_key": item.get("frame_key"),
                "label": item.get("label"),
                "frame_index": item.get("frame_index"),
            }
            for item in screenshots
        ]
        diagnosis_for_json["trajectory_image"] = ""
        diagnosis_for_json["speed_image"] = ""
        diagnosis_for_json["module5_summary_image"] = ""
        save_json(output_dirs["json"] / f"{stem}_diagnosis.json", diagnosis_for_json)
        diagnoses.append(diagnosis_for_json)

        cleanup_targets = [screenshot_dir]
        if trajectory_path is not None:
            cleanup_targets.append(trajectory_path)
        if speed_path is not None:
            cleanup_targets.append(speed_path)
        _cleanup_media_paths(cleanup_targets)

    summary_df = build_summary_frame(diagnoses)
    summary_df.to_csv(output_dirs["csv"] / "diagnosis_summary.csv", index=False, encoding="utf-8-sig")
    save_json(output_dirs["json"] / "diagnosis_all.json", diagnoses)

    print(f"[模块七] 已完成 {len(diagnoses)} 个样本的诊断输出")
    print(f"[模块七] 输出目录: {output_dirs['root']}")


if __name__ == "__main__":
    main()
