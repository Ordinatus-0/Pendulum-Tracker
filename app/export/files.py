"""Export analysis artefacts without mixing UI concerns into the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_csv(data: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def write_summary(summary: dict, fit: dict, warnings: list[str], destination: str | Path) -> Path:
    path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
    clean = lambda item: None if isinstance(item, (float, np.floating)) and not np.isfinite(item) else item
    payload = {"tracking_summary": {key: clean(value) for key, value in summary.items()},
               "spiral_fit": {key: clean(value) for key, value in fit.items() if key not in {"theta", "radius", "predicted"}}, "warnings": warnings}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_figures(figures: dict[str, plt.Figure], directory: str | Path) -> list[Path]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        for ext in ("png", "svg"):
            path = directory / f"{name}.{ext}"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            paths.append(path)
    return paths
