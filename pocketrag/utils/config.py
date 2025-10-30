from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json, yaml

DEFAULTS = {
    "data_dir": "./data",
    "results_dir": "./results",
    "seed": 42,
    "device": "auto"
}

def load_config(cfg_path: str = "pocketrag.yaml") -> Dict[str, Any]:
    p = Path(cfg_path)
    if not p.exists():
        return DEFAULTS.copy()
    text = p.read_text()
    try:
        data = yaml.safe_load(text)
        if data is None:
            data = {}
    except Exception:
        data = json.loads(text)
    cfg = DEFAULTS.copy()
    cfg.update({k: v for k, v in data.items() if v is not None})
    return cfg

def data_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["data_dir"]).resolve()

def results_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["results_dir"]).resolve()