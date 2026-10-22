"""Machine-readable evaluation output helpers."""

import json
from pathlib import Path


def write_json(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return destination