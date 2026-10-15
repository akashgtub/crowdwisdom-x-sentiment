"""
Unit tests for price ingestion and cleaning.
"""

import pytest
import pandas as pd
import numpy as np
from src.ingestion.price_ingestion import clean_price_data

def test_clean_price_data_column_normalization():
    # Test standard lowercase column normalization
    data = {
        "Open": [100.0, 102.0],
        "High": [105.0, 107.0],
        "Low": [99.0, 101.0],
        "Close": [104.0, 106.0],
        "Volume": [1000, 1500]
    }
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    df = pd.DataFrame(data, index=idx)
    cleaned = clean_price_data(df)

    assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]
    assert cleaned.index.tz is not None
    assert str(cleaned.index.tz) == "UTC"

def test_clean_price_data_multiindex_flattening():
    # yfinance often returns MultiIndex columns (Attribute, Ticker)
    tuples = [
        ("Open", "BTC-USD"),
        ("High", "BTC-USD"),
        ("Low", "BTC-USD"),
        ("Close", "BTC-USD"),
        ("Volume", "BTC-USD")
    ]
    cols = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame(np.random.uniform(50000, 60000, (3, 5)), index=idx, columns=cols)
    cleaned = clean_price_data(df)

    assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]
    assert len(cleaned) == 3

def test_clean_price_data_deduplication():
    # Duplicate index entries should keep the latest
    idx = [
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC")
    ]
    df = pd.DataFrame({
        "open": [100.0, 105.0, 110.0],
        "high": [102.0, 108.0, 112.0],
        "low": [98.0, 103.0, 109.0],
        "close": [101.0, 107.0, 111.0],
        "volume": [500, 600, 700]
    }, index=idx)

    cleaned = clean_price_data(df)
    assert len(cleaned) == 2
    assert cleaned.loc[pd.Timestamp("2026-01-01", tz="UTC"), "close"] == 107.0
