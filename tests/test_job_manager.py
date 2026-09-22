from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from backend.job_manager import JobManager, _safe_video_name


class JobManagerTests(unittest.TestCase):
    def test_safe_video_name_rejects_non_video(self) -> None:
        with self.assertRaises(ValueError):
            _safe_video_name("notes.txt")

    def test_safe_video_name_removes_path_and_unsafe_characters(self) -> None:
        self.assertEqual(_safe_video_name("../动作 视频(1).MP4"), "动作_视频_1.mp4")

    def test_create_job_persists_upload_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = JobManager(root, jobs_root=root / "jobs", python_path=root / "python")
            manager._run_job = lambda job_id: None  # type: ignore[method-assign]
            status = manager.create_job("sample.mp4", io.BytesIO(b"video"))
            stored = manager.get_job(status["id"])
            self.assertEqual(stored["status"], "queued")
            self.assertEqual(stored["size_bytes"], 5)
            self.assertTrue((root / "jobs" / status["id"] / "input" / "sample.mp4").is_file())

    def test_interrupted_job_is_marked_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "jobs" / "abcd"
            job_dir.mkdir(parents=True)
            (job_dir / "status.json").write_text(
                json.dumps({"id": "abcd", "status": "running"}), encoding="utf-8"
            )
            manager = JobManager(root, jobs_root=root / "jobs", python_path=root / "python")
            status = manager.get_job("abcd")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stage"], "interrupted")


if __name__ == "__main__":
    unittest.main()
