from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_STAGE_DIR = PROJECT_ROOT / "1_视频预处理" / "output"
DEFAULT_PIPELINE_PYTHON = PROJECT_ROOT / ".venv_rtmpose" / "bin" / "python3.11"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

MODULE_OUTPUT_DIRS = [
    PROJECT_ROOT / "1_视频预处理" / "output",
    PROJECT_ROOT / "2_栏架识别" / "outputs",
    PROJECT_ROOT / "3_人体关键点_RTMPOSE" / "output",
    PROJECT_ROOT / "4_阶段划分" / "output",
    PROJECT_ROOT / "5_特征融合" / "output",
    PROJECT_ROOT / "6_技术评价与反馈" / "output",
    PROJECT_ROOT / "7_动作诊断与建议" / "output",
]

MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="总运行脚本：从 input 视频到最终报告")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="输入视频目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="最终报告输出目录")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR, help="模块一统一输入目录")
    parser.add_argument("--python", type=Path, default=DEFAULT_PIPELINE_PYTHON, help="流水线 Python 解释器")
    parser.add_argument("--no-clean", action="store_true", help="不清理已有中间输出")
    parser.add_argument("--enable-llm", action="store_true", help="模块七启用 llama.cpp 文案生成")
    parser.add_argument("--dry-run", action="store_true", help="只打印流程，不实际执行")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_subprocess_env() -> dict[str, str]:
    ensure_dir(MPLCONFIGDIR)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
    return env


def list_input_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"输入目录不存在，已自动创建，请把视频放入后再运行: {input_dir}")
    videos = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(f"输入目录中未找到视频文件: {input_dir}")
    return videos


def reset_output_dirs(clean: bool) -> None:
    for path in MODULE_OUTPUT_DIRS:
        if clean and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)


def stage_input_videos(videos: Iterable[Path], stage_dir: Path) -> list[str]:
    ensure_dir(stage_dir)
    staged_stems: list[str] = []
    for src in videos:
        dst = stage_dir / src.name
        shutil.copy2(src, dst)
        staged_stems.append(src.stem)
    return staged_stems


def _stream_output(pipe, prefix: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(f"[{prefix}] {line.rstrip()}")
    finally:
        pipe.close()


def run_command(command: list[str], *, name: str, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"\n=== {name} ===")
    print(printable)
    if dry_run:
        return

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=build_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    reader = threading.Thread(target=_stream_output, args=(process.stdout, name), daemon=True)
    reader.start()
    return_code = process.wait()
    reader.join()
    if return_code != 0:
        raise RuntimeError(f"{name} 执行失败，退出码 {return_code}")


def run_parallel(commands: list[tuple[str, list[str]]], *, dry_run: bool) -> None:
    print("\n=== 并行阶段：模块2 + 模块3 ===")
    for name, command in commands:
        print(" ".join(command))
    if dry_run:
        return

    processes: list[tuple[str, subprocess.Popen[str], threading.Thread]] = []
    try:
        for name, command in commands:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=build_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            reader = threading.Thread(target=_stream_output, args=(process.stdout, name), daemon=True)
            reader.start()
            processes.append((name, process, reader))

        failures: list[str] = []
        for name, process, reader in processes:
            return_code = process.wait()
            reader.join()
            if return_code != 0:
                failures.append(f"{name}(exit={return_code})")

        if failures:
            raise RuntimeError("并行阶段失败: " + ", ".join(failures))
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.kill()


def collect_final_outputs(stems: Iterable[str], output_dir: Path) -> dict[str, list[str]]:
    report_dir = ensure_dir(output_dir / "reports")
    json_dir = ensure_dir(output_dir / "json")
    csv_dir = ensure_dir(output_dir / "csv")

    copied_reports: list[str] = []
    copied_jsons: list[str] = []
    copied_csvs: list[str] = []
    technical_rows: list[dict[str, str]] = []
    technical_fieldnames: list[str] = []

    for stem in stems:
        diagnosis_report = PROJECT_ROOT / "7_动作诊断与建议" / "output" / "reports" / f"{stem}_advice.html"
        score_report = PROJECT_ROOT / "6_技术评价与反馈" / "output" / "reports" / f"{stem}_report.html"
        diagnosis_json = PROJECT_ROOT / "7_动作诊断与建议" / "output" / "json" / f"{stem}_diagnosis.json"
        score_json = PROJECT_ROOT / "6_技术评价与反馈" / "output" / "json" / f"{stem}_result.json"
        technical_csv = PROJECT_ROOT / "5_特征融合" / "output" / "csv" / "technical" / f"{stem}_technical_metrics.csv"

        if diagnosis_report.exists():
            target = report_dir / diagnosis_report.name
            shutil.copy2(diagnosis_report, target)
            copied_reports.append(str(target))
        if score_report.exists():
            target = report_dir / score_report.name
            shutil.copy2(score_report, target)
            copied_reports.append(str(target))
        if diagnosis_json.exists():
            target = json_dir / diagnosis_json.name
            shutil.copy2(diagnosis_json, target)
            copied_jsons.append(str(target))
        if score_json.exists():
            target = json_dir / score_json.name
            shutil.copy2(score_json, target)
            copied_jsons.append(str(target))
        if technical_csv.exists():
            with technical_csv.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for field in reader.fieldnames:
                        if field not in technical_fieldnames:
                            technical_fieldnames.append(field)
                technical_rows.extend(reader)

    module6_score_csv = PROJECT_ROOT / "6_技术评价与反馈" / "output" / "csv" / "score_results.csv"
    module7_summary_csv = PROJECT_ROOT / "7_动作诊断与建议" / "output" / "csv" / "diagnosis_summary.csv"

    if technical_rows and technical_fieldnames:
        technical_summary_path = csv_dir / "technical_metrics_summary.csv"
        with technical_summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=technical_fieldnames)
            writer.writeheader()
            writer.writerows(technical_rows)
        copied_csvs.append(str(technical_summary_path))

    for source_csv in [module6_score_csv, module7_summary_csv]:
        if source_csv.exists():
            target = csv_dir / source_csv.name
            shutil.copy2(source_csv, target)
            copied_csvs.append(str(target))

    manifest = {
        "reports": copied_reports,
        "json": copied_jsons,
        "csv": copied_csvs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    python_path = str(args.python.expanduser())
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    stage_dir = args.stage_dir.resolve()

    videos = list_input_videos(input_dir)
    print(f"发现输入视频 {len(videos)} 个")
    for video in videos:
        print(f"  - {video.name}")

    reset_output_dirs(clean=not args.no_clean)
    staged_stems = stage_input_videos(videos, stage_dir)
    ensure_dir(output_dir)

    module2_cmd = [python_path, str(PROJECT_ROOT / "2_栏架识别" / "src" / "run.py")]
    module3_cmd = [python_path, str(PROJECT_ROOT / "3_人体关键点_RTMPOSE" / "src" / "run.py")]
    module4_cmd = [python_path, str(PROJECT_ROOT / "4_阶段划分" / "src" / "run.py")]
    module5_cmd = [python_path, str(PROJECT_ROOT / "5_特征融合" / "run.py")]
    module6_cmd = [python_path, str(PROJECT_ROOT / "6_技术评价与反馈" / "src" / "run.py")]
    module7_cmd = [python_path, str(PROJECT_ROOT / "7_动作诊断与建议" / "src" / "run.py")]
    if args.enable_llm:
        module7_cmd.extend(["--enable-llm", "--llm-backend", "llama_cpp"])

    run_parallel(
        [
            ("模块2", module2_cmd),
            ("模块3", module3_cmd),
        ],
        dry_run=args.dry_run,
    )
    run_command(module4_cmd, name="模块4", dry_run=args.dry_run)
    run_command(module5_cmd, name="模块5", dry_run=args.dry_run)
    run_command(module6_cmd, name="模块6", dry_run=args.dry_run)
    run_command(module7_cmd, name="模块7", dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run 完成，未执行实际推理。")
        return

    manifest = collect_final_outputs(staged_stems, output_dir)
    print("\n=== 最终输出 ===")
    for path in manifest["reports"]:
        print(f"报告: {path}")
    for path in manifest["json"]:
        print(f"JSON: {path}")
    for path in manifest["csv"]:
        print(f"CSV: {path}")
    print(f"清单: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
