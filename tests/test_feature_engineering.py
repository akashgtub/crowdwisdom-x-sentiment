"""
Unit tests for feature engineering, horizon targets, and zero-lookahead alignment.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import (
    aggregate_sentiment_daily,
    build_feature_dataset,
    get_feature_columns_for_horizon
)

def test_daily_sentiment_aggregation_dispersion():
    posts = [
        {"id": "1", "date": "2026-06-01", "sentiment_score": 0.8, "likes": 10, "reposts": 2, "views": 100, "author_followers": 500},
        {"id": "2", "date": "2026-06-01", "sentiment_score": -0.4, "likes": 20, "reposts": 4, "views": 200, "author_followers": 1500},
        {"id": "3", "date": "2026-06-02", "sentiment_score": 0.5, "likes": 5, "reposts": 1, "views": 50, "author_followers": 200}
    ]

    daily = aggregate_sentiment_daily(posts)
    assert len(daily) == 2

    # Check June 1st mean and dispersion
    june1 = daily.loc[pd.Timestamp("2026-06-01", tz="UTC")]
    assert june1["post_count"] == 2
    assert pytest.approx(june1["sentiment_mean"], 0.01) == 0.2
    assert june1["sentiment_std"] > 0.0
    assert june1["likes_sum"] == 30
    assert june1["has_sentiment_data"] == 1
    assert june1["is_imputed"] == 0

    # Single post day dispersion should be 0.0, not NaN
    june2 = daily.loc[pd.Timestamp("2026-06-02", tz="UTC")]
    assert june2["post_count"] == 1
    assert june2["sentiment_std"] == 0.0

def test_zero_lookahead_and_target_alignment():
    # Construct 100 days of price data
    dates = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
    prices = np.linspace(100, 200, 100)
    price_df = pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": 10000
    }, index=dates)

    feat_df = build_feature_dataset(price_df)

    # Test Target Alignment across horizons
    # For day t:
    # target_n0 == close_t
    # target_n1 == close_{t+1}
    # target_n5 == close_{t+5}
    sample_idx = feat_df.index[10]
    next_idx = feat_df.index[11]
    plus5_idx = feat_df.index[15]

    assert feat_df.loc[sample_idx, "target_n0"] == feat_df.loc[sample_idx, "close"]
    assert feat_df.loc[sample_idx, "target_n1"] == feat_df.loc[next_idx, "close"]
    assert feat_df.loc[sample_idx, "target_n5"] == feat_df.loc[plus5_idx, "close"]

def test_distinction_missing_data_from_neutral():
    dates = pd.date_range("2026-01-01", periods=50, freq="D", tz="UTC")
    price_df = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000
    }, index=dates)

    # Create sentiment for days after 20-day SMA warmup (e.g. Jan 25 and Jan 26)
    sent_dates = [pd.Timestamp("2026-01-25", tz="UTC"), pd.Timestamp("2026-01-26", tz="UTC")]
    sent_df = pd.DataFrame({
        "post_count": [10, 15],
        "sentiment_mean": [0.0, 0.5],  # Jan 25 is genuine neutral (posts present, mean 0.0)
        "sentiment_std": [0.1, 0.2],
        "has_sentiment_data": [1, 1],
        "is_imputed": [0, 0]
    }, index=sent_dates)

    feat_df = build_feature_dataset(price_df, sent_df)

    # Jan 25 has genuine neutral data: has_sentiment_data == 1, is_imputed == 0
    # Jan 40 (Feb 09) has missing data: has_sentiment_data == 0, is_imputed == 1
    assert feat_df.loc[pd.Timestamp("2026-01-25", tz="UTC"), "has_sentiment_data"] == 1
    assert feat_df.loc[pd.Timestamp("2026-01-25", tz="UTC"), "is_imputed"] == 0

    assert feat_df.loc[pd.Timestamp("2026-02-09", tz="UTC"), "has_sentiment_data"] == 0
    assert feat_df.loc[pd.Timestamp("2026-02-09", tz="UTC"), "is_imputed"] == 1

def test_feature_columns_n0_nowcast_leakage_prevention():
    # For n+0 nowcasting, close_t and any day t close-derived features must NOT be in the feature set (to prevent target leakage)
    n0_features = get_feature_columns_for_horizon("n+0")
    assert "close" not in n0_features
    assert "high" not in n0_features
    assert "low" not in n0_features
    assert "volume" not in n0_features
    assert "return_1d" not in n0_features
    assert "dist_sma_7" not in n0_features
    assert "volatility_7d" not in n0_features
    assert "hl_spread" not in n0_features
    assert "open" in n0_features
    assert "overnight_gap" in n0_features
    assert "return_1d_lag1" in n0_features
    assert "dist_sma_7_lag1" in n0_features

    # For forward horizons n+1..n+5, close of day t is permitted
    n1_features = get_feature_columns_for_horizon("n+1")
    assert "close" in n1_features
    assert "return_1d" in n1_features
    assert "dist_sma_7" in n1_features

