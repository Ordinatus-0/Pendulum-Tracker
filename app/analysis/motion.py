"""Coordinate transforms, smoothing, damping estimates, and spiral fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


def transform_coordinates(data: pd.DataFrame, pivot_x: float, pivot_y: float, invert_y: bool = True) -> pd.DataFrame:
    """Set the pivot as origin; positive x is right and positive y is upward."""
    result = data.copy()
    result["x_rel_px"] = result.x_px - pivot_x
    result["y_rel_px"] = (pivot_y - result.y_px) if invert_y else (result.y_px - pivot_y)
    result["r_px"] = np.hypot(result.x_rel_px, result.y_rel_px)
    result["theta_rad"] = np.unwrap(np.arctan2(result.y_rel_px, result.x_rel_px))
    return result


def smooth_coordinates(data: pd.DataFrame, window: int = 21, polyorder: int = 3) -> pd.DataFrame:
    result = data.copy()
    window = min(window if window % 2 else window - 1, len(result) if len(result) % 2 else len(result) - 1)
    if window < polyorder + 2 or window < 3:
        result["x_smooth_px"], result["y_smooth_px"] = result.x_rel_px, result.y_rel_px
        return result
    result["x_smooth_px"] = savgol_filter(result.x_rel_px, window, polyorder)
    result["y_smooth_px"] = savgol_filter(result.y_rel_px, window, polyorder)
    result["r_smooth_px"] = np.hypot(result.x_smooth_px, result.y_smooth_px)
    result["theta_smooth_rad"] = np.unwrap(np.arctan2(result.y_smooth_px, result.x_smooth_px))
    return result


def exponential_spiral(theta: np.ndarray, amplitude: float, decay: float) -> np.ndarray:
    return amplitude * np.exp(-decay * theta)


def fit_spiral(data: pd.DataFrame, use_smoothed: bool = True) -> dict:
    r_col = "r_smooth_px" if use_smoothed and "r_smooth_px" in data else "r_px"
    theta_col = "theta_smooth_rad" if use_smoothed and "theta_smooth_rad" in data else "theta_rad"
    subset = data[[r_col, theta_col]].dropna()
    if len(subset) < 8 or subset[r_col].max() <= 0:
        return {"reliable": False, "reason": "Too few valid points for a spiral fit."}
    theta = subset[theta_col].to_numpy()
    radius = subset[r_col].to_numpy()
    theta = theta - theta.min()
    try:
        params, _ = curve_fit(exponential_spiral, theta, radius, p0=(radius[0], .01), bounds=([0, -2], [np.inf, 2]), maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return {"reliable": False, "reason": f"Spiral optimization failed: {exc}"}
    predicted = exponential_spiral(theta, *params)
    ss_res, ss_tot = np.sum((radius - predicted) ** 2), np.sum((radius - radius.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"reliable": bool(r_squared >= .5 and params[1] >= 0), "A": float(params[0]), "k": float(params[1]), "r_squared": float(r_squared),
            "theta": theta, "radius": radius, "predicted": predicted,
            "reason": "Fit R² is below 0.50 or the fitted decay is non-physical." if r_squared < .5 or params[1] < 0 else ""}


def motion_summary(data: pd.DataFrame) -> dict:
    detected = data.tracking_status.eq("detected")
    x = data.get("x_smooth_px", data.get("x_rel_px", pd.Series(dtype=float)))
    peaks, _ = find_peaks(np.abs(x.to_numpy()), distance=max(1, len(data) // 30)) if len(x) else (np.array([]), {})
    amplitudes = np.abs(x.iloc[peaks]) if len(peaks) else pd.Series(dtype=float)
    damping = np.nan
    if len(amplitudes) >= 2 and (amplitudes > 0).all():
        damping = float(-np.polyfit(data.timestamp_s.iloc[peaks], np.log(amplitudes), 1)[0])
    return {"tracked_frames": int(detected.sum()), "total_frames": int(len(data)), "interpolated_frames": int(data.tracking_status.eq("interpolated").sum()),
            "mean_confidence": float(data.loc[detected, "confidence"].mean()) if detected.any() else np.nan,
            "maximum_displacement_px": float(np.abs(x).max()) if len(x) else np.nan, "decay_per_second": damping,
            "amplitude_peak_count": int(len(amplitudes))}
