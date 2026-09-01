"""Coordinate transforms, smoothing, damping estimates, and model fitting."""
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


def _fit_quality(observed: np.ndarray, predicted: np.ndarray, parameter_count: int) -> tuple[float, float]:
    """Return R² and AICc; the latter prevents complex models winning by default."""
    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - observed.mean()) ** 2))
    r_squared = 1 - residual_sum / total_sum if total_sum else 0.0
    sample_count = len(observed)
    # A small floor makes the score well-defined for a perfect synthetic fit.
    aic = sample_count * np.log(max(residual_sum / sample_count, np.finfo(float).eps)) + 2 * parameter_count
    correction = (2 * parameter_count * (parameter_count + 1)) / (sample_count - parameter_count - 1)
    return float(r_squared), float(aic + correction)


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


def fit_best_radial_model(data: pd.DataFrame, use_smoothed: bool = True) -> dict:
    """Compare plausible radial models and return the best-supported one.

    A pendulum trajectory is not assumed to be a spiral.  Models are scored with
    AICc (not just R²), which penalises the quadratic curve for its extra degree
    of freedom.  The exponential spiral remains a candidate and is returned as
    ``spiral_fit`` so callers can always display that requested comparison.
    """
    r_col = "r_smooth_px" if use_smoothed and "r_smooth_px" in data else "r_px"
    theta_col = "theta_smooth_rad" if use_smoothed and "theta_smooth_rad" in data else "theta_rad"
    subset = data[[r_col, theta_col]].dropna()
    if len(subset) < 8 or subset[r_col].max() <= 0:
        return {"reliable": False, "reason": "Too few valid points for model fitting.", "spiral_fit": fit_spiral(data, use_smoothed)}

    theta = subset[theta_col].to_numpy(dtype=float)
    theta = theta - theta.min()
    radius = subset[r_col].to_numpy(dtype=float)
    candidates: list[dict] = []
    for name, label, degree in (("constant", "Constant radius", 0), ("linear", "Linear radius", 1), ("quadratic", "Quadratic radius", 2)):
        coefficients = np.polyfit(theta, radius, degree)
        predicted = np.polyval(coefficients, theta)
        r_squared, aicc = _fit_quality(radius, predicted, degree + 1)
        candidates.append({"name": name, "label": label, "parameters": coefficients.tolist(), "r_squared": r_squared, "aicc": aicc, "theta": theta, "radius": radius, "predicted": predicted})

    spiral = fit_spiral(data, use_smoothed)
    if "predicted" in spiral:
        r_squared, aicc = _fit_quality(radius, spiral["predicted"], 2)
        candidates.append({"name": "exponential_spiral", "label": "Exponential spiral", "parameters": [spiral["A"], spiral["k"]], "r_squared": r_squared, "aicc": aicc, "theta": theta, "radius": radius, "predicted": spiral["predicted"]})

    best = min(candidates, key=lambda candidate: candidate["aicc"])
    return {**best, "reliable": bool(best["r_squared"] >= .5), "spiral_fit": spiral,
            "candidates": [{key: value for key, value in candidate.items() if key not in {"theta", "radius", "predicted"}} for candidate in candidates],
            "reason": "Best model R² is below 0.50." if best["r_squared"] < .5 else ""}


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
