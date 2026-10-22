from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.utils.paths import project_path


def save_figure(figure, path: str | Path) -> Path:
    """Save a static report figure as a readable 300-DPI PNG and close it."""
    destination = project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_plotly(figure: go.Figure, stem: str | Path) -> dict[str, str | None]:
    """Always save interactive HTML; record PNG export failures explicitly."""
    destination = project_path(stem)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html_path = destination.with_suffix(".html")
    figure.write_html(html_path, include_plotlyjs="cdn")
    png_path = destination.with_suffix(".png")
    try:
        figure.write_image(png_path, width=1600, height=1000, scale=2)
        png_result: str | None = str(png_path.relative_to(project_path(".")))
        export_error = None
    except (ImportError, ValueError, RuntimeError) as error:
        png_result = None
        export_error = str(error)
    export_record = {
        "html": str(html_path.relative_to(project_path("."))),
        "png": png_result,
        "png_export_limitation": export_error,
    }
    html_path.with_suffix(".export.json").write_text(json.dumps(export_record, indent=2), encoding="utf-8")
    return export_record