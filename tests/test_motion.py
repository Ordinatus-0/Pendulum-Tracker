import numpy as np
import pandas as pd

from app.analysis.motion import exponential_spiral, fit_spiral, smooth_coordinates, transform_coordinates


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
