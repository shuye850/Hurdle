from __future__ import annotations

import json
import os
import pickle
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

try:
    from tabpfn.model_loading import load_fitted_tabpfn_model
except Exception:
    load_fitted_tabpfn_model = None

try:
    from tabpfn.settings import settings as tabpfn_settings
except Exception:
    tabpfn_settings = None


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "deployable_models"


class HurdleScorePredictor:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.cache_dir = self.model_dir / ".tabpfn_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(self.cache_dir)
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        if tabpfn_settings is not None:
            tabpfn_settings.tabpfn.model_cache_dir = self.cache_dir

        manifest_path = self.model_dir / "deployment_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = [entry for entry in manifest if entry.get("enabled", True)]
        self.entry_map: dict[str, dict[str, Any]] = {}
        self.bundles: dict[str, dict[str, Any]] = {}

        for entry in self.entries:
            task_id = str(entry["task_id"])
            self.entry_map[task_id] = entry
            self.bundles[task_id] = self._load_bundle(entry)

    def _resolve_local_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path

        local_candidate = self.model_dir / PureWindowsPath(str(raw_path)).name
        if local_candidate.exists():
            return local_candidate

        raise FileNotFoundError(f"Artifact not found: {raw_path}")

    def _load_bundle(self, entry: dict[str, Any]) -> dict[str, Any]:
        artifact_format = str(entry.get("artifact_format", "pickle"))
        if artifact_format == "tabpfn_fit":
            if load_fitted_tabpfn_model is None:
                raise ImportError("tabpfn 未安装，无法加载 .tabpfn_fit 模型。")

            model_path = self._resolve_local_path(entry["model_path"])
            metadata_path = self._resolve_local_path(entry["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model = load_fitted_tabpfn_model(model_path, device="cpu")

            return {
                "model": model,
                "feature_names": metadata["feature_names"],
                "preprocess": metadata["preprocess"],
                "score_clip_min": metadata["score_clip_min"],
                "score_clip_max": metadata["score_clip_max"],
            }

        model_path = self._resolve_local_path(entry["model_path"])
        with model_path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _to_frame(metrics: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(metrics, pd.DataFrame):
            return metrics.copy()
        if isinstance(metrics, pd.Series):
            return pd.DataFrame([metrics.to_dict()])
        return pd.DataFrame([dict(metrics)])

    @staticmethod
    def _apply_preprocess(X: np.ndarray, preprocess: dict[str, Any]) -> np.ndarray:
        medians = np.asarray(preprocess["medians"], dtype=float)
        X = np.where(np.isnan(X), medians, X)
        if preprocess["mode"] == "scale":
            mean = np.asarray(preprocess["mean"], dtype=float)
            std = np.asarray(preprocess["std"], dtype=float)
            std[std == 0] = 1.0
            X = (X - mean) / std
        return X

    def predict(self, metrics: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        frame = self._to_frame(metrics)
        results = pd.DataFrame(index=frame.index)
        total_tasks = len(self.bundles)

        print(f"[评分] 开始批量预测：{len(frame)} 个样本，{total_tasks} 个任务")
        for task_idx, (task_id, bundle) in enumerate(self.bundles.items(), start=1):
            entry = self.entry_map.get(task_id, {})
            task_name = entry.get("task_name", task_id)
            print(f"[评分] ({task_idx}/{total_tasks}) {task_id} {task_name}")

            features = bundle["feature_names"]
            X = frame.reindex(columns=features).to_numpy(dtype=float, copy=True)
            X_ready = self._apply_preprocess(X, bundle["preprocess"])
            preds = np.asarray(bundle["model"].predict(X_ready), dtype=float)
            preds = np.clip(preds, bundle["score_clip_min"], bundle["score_clip_max"])

            results[task_id] = np.round(preds, 4)
            results[f"{task_id}_round"] = np.rint(preds).astype(int)

        return results.reset_index(drop=True)

    def predict_from_metrics_csv(self, metrics_csv_path: Path | str) -> pd.DataFrame:
        return self.predict(pd.read_csv(metrics_csv_path))

    def get_enabled_entries(self) -> list[dict[str, Any]]:
        return list(self.entries)


if __name__ == "__main__":
    predictor = HurdleScorePredictor()
    print("Loaded tasks:", ", ".join(sorted(predictor.bundles)))
