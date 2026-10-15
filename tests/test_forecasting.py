"""
Unit tests for forecasting models and walk-forward validation.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import build_feature_dataset
from src.models.forecasting import MultiHorizonForecaster

def test_walk_forward_expanding_window_strictly_temporal():
    # Verify that in expanding-window validation, training indices strictly precede test index
    dates = pd.date_range("2026-01-01", periods=80, freq="D", tz="UTC")
    prices = np.linspace(50000, 60000, 80) + np.random.normal(0, 100, 80)
    df = pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": 100000
    }, index=dates)

    feat_df = build_feature_dataset(df)
    forecaster = MultiHorizonForecaster(n_estimators=10, max_depth=3, min_train_size=35, step_size=3)
    results = forecaster.run_walk_forward_validation(feat_df)

    assert "horizon_results" in results
    assert "summary_metrics_df" in results

    # Verify predictions dataframe structure for horizon n+1
    n1_res = results["horizon_results"]["n+1"]
    p_df = n1_res["predictions_df"]
    assert not p_df.empty
    assert "actual" in p_df.columns
    assert "rf_predicted" in p_df.columns
    assert "naive_predicted" in p_df.columns
    assert n1_res["rf_metrics"]["mae"] > 0
