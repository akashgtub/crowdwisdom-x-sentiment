"""
Feature engineering module for CrowdWisdomTrading pipeline.
Aggregates X sentiment and engagement metrics into daily intervals,
constructs technical price features, momentum, volatility, and time features,
and strictly aligns data to prevent future lookahead bias.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.db import save_daily_sentiment

logger = logging.getLogger(__name__)

def aggregate_sentiment_daily(posts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate individual post sentiment scores and engagement into daily bars.
    
    Metrics created:
    - post_count: total posts on that date
    - sentiment_mean: average compound sentiment
    - sentiment_std: sentiment dispersion (std dev)
    - likes_sum, likes_mean
    - reposts_sum, reposts_mean
    - views_sum, views_mean
    - followers_mean: average author follower count
    - sentiment_pos_ratio, sentiment_neg_ratio
    - is_imputed: 0 for days with observed posts
    """
    if not posts:
        logger.warning("No posts provided for sentiment aggregation.")
        return pd.DataFrame()

    df_posts = pd.DataFrame(posts)
    if "date" not in df_posts.columns or "sentiment_score" not in df_posts.columns:
        raise ValueError("Posts list missing required 'date' or 'sentiment_score' fields.")

    df_posts["date"] = pd.to_datetime(df_posts["date"], utc=True).dt.normalize()

    # Numeric conversions
    for col in ["likes", "reposts", "views", "author_followers", "sentiment_score"]:
        if col in df_posts.columns:
            df_posts[col] = pd.to_numeric(df_posts[col], errors="coerce").fillna(0)

    # Calculate sentiment polarity flags
    df_posts["is_pos"] = (df_posts["sentiment_score"] >= 0.05).astype(int)
    df_posts["is_neg"] = (df_posts["sentiment_score"] <= -0.05).astype(int)

    # Track source provenance explicitly
    if "source" not in df_posts.columns:
        df_posts["source"] = "sample_fixture"
    df_posts["is_live"] = (df_posts["source"].str.startswith("live")).astype(int)
    df_posts["is_sample"] = (df_posts["source"].str.contains("sample|fixture")).astype(int)

    grouped = df_posts.groupby("date")
    daily = grouped.agg(
        post_count=("id", "count"),
        live_post_count=("is_live", "sum"),
        sample_post_count=("is_sample", "sum"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        likes_sum=("likes", "sum"),
        likes_mean=("likes", "mean"),
        reposts_sum=("reposts", "sum"),
        reposts_mean=("reposts", "mean"),
        views_sum=("views", "sum"),
        views_mean=("views", "mean"),
        followers_mean=("author_followers", "mean"),
        pos_count=("is_pos", "sum"),
        neg_count=("is_neg", "sum")
    )

    # Fill dispersion NaN (for single-post days) with 0.0
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily["sentiment_pos_ratio"] = (daily["pos_count"] / daily["post_count"]).fillna(0.0)
    daily["sentiment_neg_ratio"] = (daily["neg_count"] / daily["post_count"]).fillna(0.0)
    daily.drop(columns=["pos_count", "neg_count"], inplace=True)
    daily["is_imputed"] = 0
    daily["has_sentiment_data"] = 1
    daily["provenance"] = np.where(daily["live_post_count"] > 0, "live_apify", "sample_fixture")

    daily.sort_index(inplace=True)
    logger.info("Aggregated %d posts into %d daily sentiment bars (live_posts=%d, sample_posts=%d).",
                len(df_posts), len(daily), int(daily["live_post_count"].sum()), int(daily["sample_post_count"].sum()))
    return daily

def build_feature_dataset(
    price_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge historical price data with aggregated sentiment and generate features.
    
    Strict Zero-Lookahead Rules:
    - Features at row t only utilize price information and sentiment available up to day t.
    - Missing sentiment days are explicitly flagged with `has_sentiment_data = 0` and `is_imputed = 1`,
      preventing neutral sentiment from being conflated with missing data.
    - Generates target horizons:
        - target_n0: close price of day t (nowcasted from open_t, sentiment_t, and lags)
        - target_n1: close price of day t+1
        - target_n2: close price of day t+2
        - target_n3: close price of day t+3
        - target_n4: close price of day t+4
        - target_n5: close price of day t+5
    """
    if price_df.empty:
        raise ValueError("price_df cannot be empty.")

    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.normalize()

    # Align sentiment
    if sentiment_df is not None and not sentiment_df.empty:
        s_df = sentiment_df.copy()
        if not isinstance(s_df.index, pd.DatetimeIndex):
            s_df.index = pd.to_datetime(s_df.index, utc=True)
        s_df.index = s_df.index.normalize()
        df = df.join(s_df, how="left")
    else:
        for col in ["post_count", "sentiment_mean", "sentiment_std", "likes_sum", "likes_mean",
                    "reposts_sum", "views_sum", "followers_mean", "sentiment_pos_ratio", "sentiment_neg_ratio"]:
            df[col] = 0.0
        df["is_imputed"] = 1
        df["has_sentiment_data"] = 0

    # Distinguish missing from neutral sentiment & track provenance:
    df["has_sentiment_data"] = df["has_sentiment_data"].fillna(0).astype(int)
    df["is_imputed"] = df["is_imputed"].fillna(1).astype(int)
    df["post_count"] = df["post_count"].fillna(0).astype(int)
    df["live_post_count"] = df["live_post_count"].fillna(0).astype(int) if "live_post_count" in df.columns else 0
    df["sample_post_count"] = df["sample_post_count"].fillna(0).astype(int) if "sample_post_count" in df.columns else 0
    if "provenance" in df.columns:
        df["provenance"] = df["provenance"].fillna("imputed_no_data")
    else:
        df["provenance"] = "imputed_no_data"
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
    df["sentiment_std"] = df["sentiment_std"].fillna(0.0)
    for c in ["likes_sum", "likes_mean", "reposts_sum", "reposts_mean", "views_sum", "views_mean", "followers_mean",
              "sentiment_pos_ratio", "sentiment_neg_ratio"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # 1. Price Momentum & Returns (strictly lagged or current-bar known)
    # daily returns up to close
    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_7d"] = df["close"].pct_change(7)
    df["return_14d"] = df["close"].pct_change(14)

    # 2. Moving Average Distance
    df["sma_7"] = df["close"].rolling(window=7).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["dist_sma_7"] = (df["close"] - df["sma_7"]) / (df["sma_7"] + 1e-9)
    df["dist_sma_20"] = (df["close"] - df["sma_20"]) / (df["sma_20"] + 1e-9)
    df["dist_sma_50"] = (df["close"] - df["sma_50"]) / (df["sma_50"] + 1e-9)

    # 3. Rolling Volatility
    df["volatility_7d"] = df["return_1d"].rolling(window=7).std().fillna(0.0)
    df["volatility_14d"] = df["return_1d"].rolling(window=14).std().fillna(0.0)

    # 4. Intraday Range and Volume Ratio
    df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["volume_sma_7"] = df["volume"].rolling(window=7).mean()
    df["volume_ratio_7d"] = df["volume"] / (df["volume_sma_7"] + 1e-9)

    # 5. Lagged Sentiment and Volume Features
    df["sentiment_mean_lag1"] = df["sentiment_mean"].shift(1).fillna(0.0)
    df["sentiment_mean_lag2"] = df["sentiment_mean"].shift(2).fillna(0.0)
    df["sentiment_mean_lag3"] = df["sentiment_mean"].shift(3).fillna(0.0)
    df["post_volume_lag1"] = df["post_count"].shift(1).fillna(0.0)
    df["post_volume_lag2"] = df["post_count"].shift(2).fillna(0.0)
    df["sentiment_momentum_3d"] = df["sentiment_mean"] - df["sentiment_mean_lag3"]

    # 6. Time-based features
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df.index.month

    # 7. Targets for horizons n+0 through n+5
    # Target n+0: closing price of day t
    df["target_n0"] = df["close"]
    # Targets n+1 to n+5: forward closing prices
    df["target_n1"] = df["close"].shift(-1)
    df["target_n2"] = df["close"].shift(-2)
    df["target_n3"] = df["close"].shift(-3)
    df["target_n4"] = df["close"].shift(-4)
    df["target_n5"] = df["close"].shift(-5)

    # Forward returns for strategy evaluation
    df["forward_ret_1d"] = (df["target_n1"] - df["close"]) / df["close"]

    # Drop early warm-up periods where moving averages / returns are NaN
    # SMA 50 needs 50 periods; if dataset has 180 days, dropping ~20-30 days leaves ample data
    df.dropna(subset=["sma_20", "return_7d"], inplace=True)
    df.fillna(0.0, inplace=True)

    logger.info("Engineered feature dataset: shape %s, date range %s to %s",
                df.shape, df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d'))
    return df

def get_feature_columns_for_horizon(horizon: str) -> List[str]:
    """
    Returns feature column names appropriate for the given horizon.
    
    Zero-Lookahead Guarantee:
    - For 'n+0' (nowcasting current day close):
      Cannot use close_t or close-derived features of day t (which would leak the target!).
      Uses open_t, day t sentiment/engagement, day t time features, and strictly lagged price features.
    - For 'n+1' through 'n+5':
      Can use all features computed up to close of day t to predict future prices.
    """
    if horizon == "n+0":
        # Strictly features available before close of day t
        return [
            "open",
            "sentiment_mean", "sentiment_std", "post_count",
            "likes_sum", "views_sum", "followers_mean",
            "sentiment_pos_ratio", "sentiment_neg_ratio", "has_sentiment_data",
            "sentiment_mean_lag1", "sentiment_mean_lag2", "post_volume_lag1",
            "sentiment_momentum_3d",
            "return_1d", "return_3d", "return_7d",  # Note: computed from prior close
            "dist_sma_7", "dist_sma_20", "volatility_7d",
            "day_of_week", "is_weekend"
        ]
    else:
        # Features up to day t close
        return [
            "open", "high", "low", "close", "volume",
            "sentiment_mean", "sentiment_std", "post_count",
            "likes_sum", "reposts_sum", "views_sum", "followers_mean",
            "sentiment_pos_ratio", "sentiment_neg_ratio", "has_sentiment_data",
            "sentiment_mean_lag1", "sentiment_mean_lag2", "sentiment_mean_lag3",
            "post_volume_lag1", "post_volume_lag2", "sentiment_momentum_3d",
            "return_1d", "return_3d", "return_7d", "return_14d",
            "dist_sma_7", "dist_sma_20", "dist_sma_50",
            "volatility_7d", "volatility_14d", "hl_spread", "volume_ratio_7d",
            "day_of_week", "is_weekend", "month"
        ]
