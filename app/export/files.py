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


def write_fit_tables(fit: dict, directory: str | Path) -> dict[str, Path]:
    """Write the scored model comparison and the two requested fitted curves."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(fit.get("candidates", []))
    if not comparison.empty:
        comparison["parameters"] = comparison["parameters"].map(lambda values: json.dumps(values))
        comparison = comparison.sort_values("aicc", kind="stable")
    comparison_path = directory / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    paths = {"model_comparison": comparison_path}
    for name, model in (("best_fit_curve", fit), ("spiral_fit_curve", fit.get("spiral_fit", {}))):
        if "theta" not in model:
            continue
        curve = pd.DataFrame({"theta_rad_from_start": model["theta"], "measured_r_px": model["radius"], "fitted_r_px": model["predicted"]})
        path = directory / f"{name}.csv"
        curve.to_csv(path, index=False)
        paths[name] = path
    return paths


def write_summary(summary: dict, fit: dict, warnings: list[str], destination: str | Path) -> Path:
    path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
    def clean(item):
        if isinstance(item, (float, np.floating)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items() if key not in {"theta", "radius", "predicted"}}
        if isinstance(item, list):
            return [clean(value) for value in item]
        return item
    payload = {"tracking_summary": {key: clean(value) for key, value in summary.items()},
               "model_fit": clean(fit), "warnings": warnings}
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
