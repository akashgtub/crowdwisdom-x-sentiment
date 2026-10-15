"""
Historical price ingestion module using yfinance.
Fetches OHLCV price bars, normalizes timestamps to UTC, cleans the data,
and persists to both CSV (raw/cleaned) and SQLite database.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf
from src.config import (
    DEFAULT_ASSET,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_INTERVAL,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DB_PATH
)
from src.db import save_prices_df, load_prices_df

logger = logging.getLogger(__name__)

def fetch_historical_prices(
    ticker: str = DEFAULT_ASSET,
    history_days: int = DEFAULT_HISTORY_DAYS,
    interval: str = DEFAULT_INTERVAL,
    save_to_db: bool = True,
    db_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch OHLCV price data for the specified asset and duration.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (raw_df, cleaned_df)
    """
    logger.info("Fetching %d days of price data for %s (interval=%s)...", history_days, ticker, interval)
    period_str = f"{history_days}d"
    
    try:
        raw_df = yf.download(ticker, period=period_str, interval=interval, progress=False)
    except Exception as e:
        logger.error("Failed to download price data from yfinance for %s: %s", ticker, e)
        # Attempt fallback to local database if previously cached
        cached_df = load_prices_df(ticker, db_path=db_path)
        if not cached_df.empty:
            logger.warning("Using %d previously cached price records from database.", len(cached_df))
            return cached_df, cached_df
        raise

    if raw_df.empty:
        logger.warning("yfinance returned empty dataset for %s. Checking local DB...", ticker)
        cached_df = load_prices_df(ticker, db_path=db_path)
        if not cached_df.empty:
            return cached_df, cached_df
        raise ValueError(f"No price data available for ticker {ticker}.")

    # Save raw DataFrame to CSV before manipulation
    raw_csv_path = RAW_DATA_DIR / f"{ticker.replace('-', '_').lower()}_raw.csv"
    raw_df.to_csv(raw_csv_path)
    logger.info("Saved raw price data to %s (shape: %s)", raw_csv_path, raw_df.shape)

    # Process and normalize
    cleaned_df = clean_price_data(raw_df)

    # Save cleaned DataFrame to CSV
    clean_csv_path = PROCESSED_DATA_DIR / f"{ticker.replace('-', '_').lower()}_clean.csv"
    cleaned_df.to_csv(clean_csv_path)
    logger.info("Saved cleaned price data to %s (shape: %s)", clean_csv_path, cleaned_df.shape)

    # Persist to SQLite
    if save_to_db:
        rows_saved = save_prices_df(cleaned_df, ticker=ticker, db_path=db_path)
        logger.info("Persisted %d price records to SQLite database.", rows_saved)

    return raw_df, cleaned_df

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize price DataFrame:
    - Flatten MultiIndex columns if present.
    - Standardize column names to lowercase ['open', 'high', 'low', 'close', 'volume'].
    - Ensure index is UTC DatetimeIndex and deduplicated.
    - Drop invalid or negative prices, forward-fill small gaps.
    """
    cleaned = df.copy()

    # Flatten MultiIndex columns if yfinance returns (Attribute, Ticker)
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [col[0].lower() for col in cleaned.columns]
    else:
        cleaned.columns = [str(c).lower() for c in cleaned.columns]

    # Required price columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Cleaned price data missing required columns: {missing}")

    cleaned = cleaned[required_cols]

    # Ensure index is datetime with UTC timezone
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, utc=True)
    elif cleaned.index.tz is None:
        cleaned.index = cleaned.index.tz_localize("UTC")
    else:
        cleaned.index = cleaned.index.tz_convert("UTC")

    # Normalize to midnight UTC for clean daily alignment
    cleaned.index = cleaned.index.normalize()
    cleaned.index.name = "timestamp"

    # Deduplicate index keeping the last record
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned.sort_index(inplace=True)

    # Handle missing values
    cleaned.ffill(inplace=True)
    cleaned.bfill(inplace=True)

    # Verify no negative or zero close prices
    cleaned = cleaned[cleaned["close"] > 0]

    return cleaned

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.db import init_db
    init_db()
    raw, clean = fetch_historical_prices()
    print("Cleaned prices head:")
    print(clean.head())
    print("Cleaned prices tail:")
    print(clean.tail())
