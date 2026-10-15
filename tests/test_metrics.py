"""
Unit tests for forecasting metrics and calculations.
"""

import pytest
import numpy as np
from src.models.metrics import calculate_metrics

def test_calculate_metrics_accuracy():
    y_true = np.array([100.0, 105.0, 110.0, 115.0])
    y_pred = np.array([101.0, 104.0, 112.0, 114.0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] == 1.25
    assert metrics["sample_count"] == 4
    assert metrics["mape"] > 0
    assert metrics["rmse"] > 0
    assert metrics["directional_accuracy"] == 100.0

def test_metrics_handles_nans():
    y_true = np.array([100.0, np.nan, 110.0])
    y_pred = np.array([102.0, 105.0, np.nan])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["sample_count"] == 1
    assert metrics["mae"] == 2.0
