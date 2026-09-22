from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.job_manager import JobManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
manager = JobManager(PROJECT_ROOT)

app = FastAPI(
    title="跨栏动作技术分析 API",
    version="0.1.0",
    description="上传视频并运行无 LLM 的跨栏动作分析流水线。",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm": "disabled"}


@app.get("/api/jobs")
def list_jobs() -> dict[str, object]:
    jobs = manager.list_jobs()
    return {"jobs": jobs, "total": len(jobs)}


@app.post("/api/jobs", status_code=202)
def create_job(video: UploadFile = File(...)) -> dict[str, object]:
    try:
        return manager.create_job(video.filename or "video.mp4", video.file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        return manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    try:
        manager.delete_job(job_id)
        return {"status": "deleted", "id": job_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_job_log(job_id: str) -> FileResponse:
    try:
        return FileResponse(manager.log_path(job_id), media_type="text/plain; charset=utf-8")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务日志不存在") from exc


@app.get("/api/jobs/{job_id}/preview")
def get_job_preview(job_id: str) -> FileResponse:
    try:
        return FileResponse(
            manager.preview_path(job_id),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="实时检测画面尚未生成") from exc


@app.get("/api/jobs/{job_id}/previews/{kind}")
def get_job_preview_by_kind(job_id: str, kind: str) -> FileResponse:
    try:
        return FileResponse(
            manager.preview_path(job_id, kind),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="该路实时画面尚未生成") from exc


@app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
    try:
        return FileResponse(manager.artifact_path(job_id, artifact_path))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="结果文件不存在") from exc


FRONTEND_DIR = PROJECT_ROOT / "8_前端" / "dist"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
