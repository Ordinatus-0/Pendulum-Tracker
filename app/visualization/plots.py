"""Plot measured and modelled pendulum motion separately."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=.3)
    ax.legend()


def motion_figures(data: pd.DataFrame, smoothed: bool = True) -> dict[str, plt.Figure]:
    x_col, y_col = ("x_smooth_px", "y_smooth_px") if smoothed and "x_smooth_px" in data else ("x_rel_px", "y_rel_px")
    label = "smoothed measurement" if x_col.startswith("x_smooth") else "raw measurement"
    figures = {}
    for name, value, ylabel in (("x_time", x_col, "Relative x (px)"), ("y_time", y_col, "Relative y (px)")):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.plot(data.timestamp_s, data[value], label=label, color="#1769aa")
        _style(ax, "Time (s)", ylabel, f"{ylabel.split(' (')[0]} vs time")
        figures[name] = fig
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.plot(data[x_col], data[y_col], label=label, color="#1769aa")
    ax.scatter(data[x_col].iloc[0], data[y_col].iloc[0], label="start", color="#d32f2f", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    _style(ax, "Relative x (px)", "Relative y (px)", "Bob trajectory")
    figures["trajectory"] = fig
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    r_col, t_col = ("r_smooth_px", "theta_smooth_rad") if smoothed and "r_smooth_px" in data else ("r_px", "theta_rad")
    ax.plot(data[t_col], data[r_col], label=label, color="#1769aa")
    _style(ax, "Angle θ (rad)", "Radial distance r (px)", "Radial distance vs angle")
    figures["r_theta"] = fig
    return figures


def spiral_figure(fit: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if "theta" not in fit:
        ax.text(.5, .5, fit.get("reason", "No spiral fit available."), ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    ax.scatter(fit["theta"], fit["radius"], s=12, alpha=.55, label="measured trajectory")
    ax.plot(fit["theta"], fit["predicted"], color="#d32f2f", lw=2, label=f"model: A exp(−kθ), R²={fit['r_squared']:.3f}")
    _style(ax, "Angle θ − θ₀ (rad)", "Radial distance r (px)", "Exponential spiral fit")
    return fig


def best_model_figure(model: dict) -> plt.Figure:
    """Show the winning radial model without implying that it is a spiral."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if "theta" not in model:
        ax.text(.5, .5, model.get("reason", "No model fit available."), ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    ax.scatter(model["theta"], model["radius"], s=12, alpha=.55, label="measured trajectory")
    ax.plot(model["theta"], model["predicted"], color="#2e7d32", lw=2, label=f"best: {model['label']}, R²={model['r_squared']:.3f}")
    _style(ax, "Angle θ − θ₀ (rad)", "Radial distance r (px)", "Best-supported radial model")
    return fig
