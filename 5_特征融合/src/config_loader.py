# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = config_path.parent.parent.resolve()
    upstream = cfg["paths"]["upstream"]
    for key, value in upstream.items():
        upstream[key] = str((base_dir / value).resolve())

    cfg["paths"]["current_output_root"] = str((base_dir / cfg["paths"]["current_output_root"]).resolve())
    return cfg