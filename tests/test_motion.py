import numpy as np
import pandas as pd

from app.analysis.motion import exponential_spiral, fit_best_radial_model, fit_spiral, smooth_coordinates, transform_coordinates
from app.export.files import write_fit_tables


def test_coordinate_transform_uses_upward_positive_y():
    data = pd.DataFrame({"x_px": [110.0], "y_px": [90.0]})
    result = transform_coordinates(data, 100, 100)
    assert result.x_rel_px.iloc[0] == 10
    assert result.y_rel_px.iloc[0] == 10
    assert np.isclose(result.r_px.iloc[0], np.sqrt(200))


def test_smoothing_preserves_series_length():
    data = pd.DataFrame({"x_px": np.linspace(0, 10, 31), "y_px": np.linspace(10, 0, 31)})
    result = smooth_coordinates(transform_coordinates(data, 0, 0), window=9)
    assert len(result.x_smooth_px) == 31


def test_spiral_fit_recovers_synthetic_parameters():
    theta = np.linspace(0, 4, 100)
    data = pd.DataFrame({"r_px": exponential_spiral(theta, 20, .15), "theta_rad": theta})
    result = fit_spiral(data, use_smoothed=False)
    assert result["reliable"]
    assert np.isclose(result["A"], 20, rtol=.01)
    assert np.isclose(result["k"], .15, rtol=.01)


def test_best_model_can_select_a_non_spiral_curve():
    theta = np.linspace(0, 4, 100)
    data = pd.DataFrame({"r_px": 10 + 2 * theta, "theta_rad": theta})
    result = fit_best_radial_model(data, use_smoothed=False)
    assert result["name"] == "linear"
    assert result["reliable"]
    assert "spiral_fit" in result


def test_model_search_includes_high_order_polynomial_and_sine_cosine_candidates(tmp_path):
    theta = np.linspace(0, 6 * np.pi, 150)
    radius = 20 + 3 * np.sin(theta) - 1.5 * np.cos(2 * theta)
    result = fit_best_radial_model(pd.DataFrame({"r_px": radius, "theta_rad": theta}), use_smoothed=False)
    names = {candidate["name"] for candidate in result["candidates"]}
    assert "polynomial_6" in names
    assert "sin_cos_3" in names
    paths = write_fit_tables(result, tmp_path)
    assert set(paths) == {"model_comparison", "best_fit_curve", "spiral_fit_curve"}
    assert pd.read_csv(paths["model_comparison"]).shape[0] == len(result["candidates"])
