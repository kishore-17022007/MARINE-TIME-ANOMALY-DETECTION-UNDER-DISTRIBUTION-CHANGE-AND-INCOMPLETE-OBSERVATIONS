from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    output_root = Path(os.environ.get("MARITIME_OUTPUT_DIR", ROOT))
    return output_root / candidate


def ensure_output_dirs() -> None:
    paths = (
        "data/raw", "data/interim", "data/processed", "data/manifests",
        "artifacts/models", "artifacts/metrics", "artifacts/logs",
        "reports/tables", "reports/generated",
        "reports/figures/eda", "reports/figures/descriptive",
        "reports/figures/diagnostic", "reports/figures/models",
        "reports/figures/drift", "reports/figures/simulation",
        "reports/figures/explainability",
    )
    for relative in paths:
        project_path(relative).mkdir(parents=True, exist_ok=True)