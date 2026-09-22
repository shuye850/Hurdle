from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_video_name(filename: str) -> str:
    original = Path(filename or "video.mp4").name
    suffix = Path(original).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("仅支持 MP4、MOV、AVI、MKV 和 M4V 视频")
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in Path(original).stem)
    stem = stem.strip("_.") or "video"
    return f"{stem[:80]}{suffix}"


class JobManager:
    """Persist jobs on disk and serialize access to the legacy pipeline outputs."""

    def __init__(
        self,
        project_root: Path,
        jobs_root: Path | None = None,
        python_path: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.jobs_root = (jobs_root or self.project_root / "runtime" / "jobs").resolve()
        self.python_path = (python_path or self.project_root / ".venv_rtmpose" / "bin" / "python3.11").resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._mark_interrupted_jobs()

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or any(char not in "0123456789abcdef-" for char in job_id.lower()):
            raise KeyError(job_id)
        path = (self.jobs_root / job_id).resolve()
        if path.parent != self.jobs_root:
            raise KeyError(job_id)
        return path

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _write_status(self, job_id: str, status: dict[str, Any]) -> None:
        status["updated_at"] = _now()
        path = self._status_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _update_status(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._state_lock:
            status = self.get_job(job_id)
            status.update(changes)
            self._write_status(job_id, status)
            return status

    def _mark_interrupted_jobs(self) -> None:
        for status_path in self.jobs_root.glob("*/status.json"):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("status") in {"queued", "running"}:
                status.update(
                    status="failed",
                    stage="interrupted",
                    error="后端重启，任务执行被中断",
                    updated_at=_now(),
                )
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_job(self, filename: str, source: BinaryIO) -> dict[str, Any]:
        safe_name = _safe_video_name(filename)
        job_id = str(uuid.uuid4())
        job_dir = self._job_dir(job_id)
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True)
        target = input_dir / safe_name

        total = 0
        with target.open("wb") as destination:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    destination.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise ValueError("视频不能超过 500 MB")
                destination.write(chunk)
        if total == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError("上传的视频为空")

        status: dict[str, Any] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "original_filename": Path(filename).name,
            "stored_filename": safe_name,
            "size_bytes": total,
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
            "artifacts": [],
        }
        self._write_status(job_id, status)
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        self._threads[job_id] = thread
        thread.start()
        return status

    def _progress_from_line(self, line: str) -> tuple[str, int] | None:
        markers = (
            ("=== 并行阶段", "detecting", 15),
            ("=== 模块4", "segmenting", 45),
            ("=== 模块5", "extracting_features", 58),
            ("=== 模块6", "scoring", 72),
            ("=== 模块7", "diagnosing", 86),
            ("=== 最终输出", "collecting_results", 96),
        )
        for marker, stage, progress in markers:
            if marker in line:
                return stage, progress
        if "[预处理完成]" in line:
            return "preprocessing", 10
        return None

    def _run_job(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        log_path = job_dir / "pipeline.log"
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"

        try:
            self._update_status(job_id, status="queued", stage="waiting_for_pipeline", progress=1)
            with self._execution_lock:
                self._update_status(job_id, status="running", stage="preparing", progress=3)
                command = [
                    str(self.python_path),
                    "-u",
                    str(self.project_root / "RUN.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ]
                with log_path.open("w", encoding="utf-8") as log_file:
                    process_env = os.environ.copy()
                    hurdle_preview = job_dir / "preview" / "hurdle.jpg"
                    pose_preview = job_dir / "preview" / "pose.jpg"
                    process_env["PIPELINE_PREVIEW_PATH"] = str(hurdle_preview)
                    process_env["PIPELINE_HURDLE_PREVIEW_PATH"] = str(hurdle_preview)
                    process_env["PIPELINE_POSE_PREVIEW_PATH"] = str(pose_preview)
                    process = subprocess.Popen(
                        command,
                        cwd=self.project_root,
                        env=process_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()
                        update = self._progress_from_line(line)
                        if update:
                            stage, progress = update
                            self._update_status(job_id, stage=stage, progress=progress)
                    return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"分析流水线执行失败，退出码 {return_code}")

                artifacts = self._collect_artifacts(job_id)
                self._update_status(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100,
                    artifacts=artifacts,
                    completed_at=_now(),
                )
        except Exception as exc:
            self._update_status(
                job_id,
                status="failed",
                stage="failed",
                error=str(exc),
            )

    def _collect_artifacts(self, job_id: str) -> list[dict[str, str]]:
        output_dir = self._job_dir(job_id) / "output"
        artifacts: list[dict[str, str]] = []
        if not output_dir.exists():
            return artifacts
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output_dir).as_posix()
            category = relative.split("/", 1)[0] if "/" in relative else "manifest"
            artifacts.append(
                {
                    "name": path.name,
                    "category": category,
                    "path": relative,
                    "url": f"/api/jobs/{job_id}/artifacts/{relative}",
                }
            )
        return artifacts

    def get_job(self, job_id: str) -> dict[str, Any]:
        path = self._status_path(job_id)
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in self.jobs_root.glob("*/status.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def delete_job(self, job_id: str) -> None:
        """Delete a finished job and all of its persisted artifacts."""
        with self._state_lock:
            status = self.get_job(job_id)
            if status.get("status") in {"queued", "running"}:
                raise RuntimeError("任务仍在运行，暂时不能删除")
            shutil.rmtree(self._job_dir(job_id))
            self._threads.pop(job_id, None)

    def artifact_path(self, job_id: str, relative_path: str) -> Path:
        output_dir = (self._job_dir(job_id) / "output").resolve()
        target = (output_dir / relative_path).resolve()
        if not target.is_relative_to(output_dir) or not target.is_file():
            raise KeyError(relative_path)
        return target

    def log_path(self, job_id: str) -> Path:
        path = self._job_dir(job_id) / "pipeline.log"
        if not path.is_file():
            raise KeyError(job_id)
        return path

    def preview_path(self, job_id: str, kind: str = "hurdle") -> Path:
        filenames = {"hurdle": "hurdle.jpg", "pose": "pose.jpg"}
        if kind not in filenames:
            raise KeyError(kind)
        path = self._job_dir(job_id) / "preview" / filenames[kind]
        legacy_path = self._job_dir(job_id) / "preview" / "latest.jpg"
        if kind == "hurdle" and not path.is_file() and legacy_path.is_file():
            return legacy_path
        if not path.is_file():
            raise KeyError(job_id)
        return path
