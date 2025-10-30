from __future__ import annotations
import platform, sys, json
from typing import Dict, Any
import psutil

def get_system_report() -> Dict[str, Any]:
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "framework_versions": {},
        "device": "cpu",
        "accelerator_name": None,
    }
    try:
        import torch
        info["framework_versions"]["torch"] = torch.__version__
        mps = getattr(torch.backends, "mps", None)
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["accelerator_name"] = torch.cuda.get_device_name(0)
        elif mps and mps.is_available():
            info["device"] = "mps"
            info["accelerator_name"] = "Apple Silicon (MPS)"
    except Exception:
        pass
    try:
        import transformers
        info["framework_versions"]["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["framework_versions"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    return info

if __name__ == "__main__":
    print(json.dumps(get_system_report(), indent=2))