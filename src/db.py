"""
SQLite persistence module for CrowdWisdomTrading pipeline.
Provides structured storage for prices, raw posts, normalized posts, daily sentiment, news, and metrics.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import DB_PATH

logger = logging.getLogger(__name__)

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection with row factory enabled."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        PRIMARY KEY (ticker, timestamp)
    );

    CREATE TABLE IF NOT EXISTS raw_posts (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        raw_json TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS normalized_posts (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        date TEXT NOT NULL,
        text TEXT NOT NULL,
        likes INTEGER DEFAULT 0,
        reposts INTEGER DEFAULT 0,
        views INTEGER DEFAULT 0,
        author_followers INTEGER DEFAULT 0,
        sentiment_score REAL DEFAULT 0.0,
        sentiment_label TEXT DEFAULT 'neutral',
        source TEXT DEFAULT 'apify'
    );

    CREATE TABLE IF NOT EXISTS daily_sentiment (
        date TEXT PRIMARY KEY,
        post_count INTEGER DEFAULT 0,
        live_post_count INTEGER DEFAULT 0,
        sample_post_count INTEGER DEFAULT 0,
        sentiment_mean REAL DEFAULT 0.0,
        sentiment_std REAL DEFAULT 0.0,
        likes_sum INTEGER DEFAULT 0,
        likes_mean REAL DEFAULT 0.0,
        reposts_sum INTEGER DEFAULT 0,
        views_sum INTEGER DEFAULT 0,
        followers_mean REAL DEFAULT 0.0,
        is_imputed INTEGER DEFAULT 0,
        provenance TEXT DEFAULT 'sample_fixture'
    );

    CREATE TABLE IF NOT EXISTS news_articles (
        url TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        source TEXT NOT NULL,
        published_date TEXT,
        snippet TEXT,
        ingested_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        horizon TEXT NOT NULL,
        mae REAL NOT NULL,
        rmse REAL NOT NULL,
        mape REAL NOT NULL,
        directional_accuracy REAL,
        evaluated_at TEXT NOT NULL
    );
    """)

    # Safe migration: ensure new columns exist if table was created previously
    cursor.execute("PRAGMA table_info(daily_sentiment)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "live_post_count" not in existing_cols:
        cursor.execute("ALTER TABLE daily_sentiment ADD COLUMN live_post_count INTEGER DEFAULT 0")
    if "sample_post_count" not in existing_cols:
        cursor.execute("ALTER TABLE daily_sentiment ADD COLUMN sample_post_count INTEGER DEFAULT 0")
    if "provenance" not in existing_cols:
        cursor.execute("ALTER TABLE daily_sentiment ADD COLUMN provenance TEXT DEFAULT 'sample_fixture'")

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully at %s", db_path or DB_PATH)

def save_prices_df(df: pd.DataFrame, ticker: str, db_path: Optional[Path] = None) -> int:
    """
    Save or replace historical price records from a DataFrame.
    DataFrame must have index as datetime or a 'timestamp' column.
    """
    if df.empty:
        return 0

    conn = get_connection(db_path)
    cursor = conn.cursor()

    records = []
    for idx, row in df.iterrows():
        ts = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(row.get("timestamp", idx))[:10]
        records.append((
            ticker,
            ts,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row.get("volume", 0.0))
        ))

    cursor.executemany("""
    INSERT OR REPLACE INTO prices (ticker, timestamp, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    count = cursor.rowcount
    conn.close()
    return len(records)

def load_prices_df(ticker: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load historical prices from database sorted by timestamp."""
    conn = get_connection(db_path)
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM prices
    WHERE ticker = ?
    ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    return df

def save_raw_post(post_id: str, source: str, raw_data: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """Store raw post json payload separately from normalized data."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
    INSERT OR REPLACE INTO raw_posts (id, source, raw_json, ingested_at)
    VALUES (?, ?, ?, ?)
    """, (post_id, source, json.dumps(raw_data), now_iso))
    conn.commit()
    conn.close()

def save_normalized_posts(posts: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """Save normalized posts, handling deduplication."""
    if not posts:
        return 0
    conn = get_connection(db_path)
    cursor = conn.cursor()
    records = []
    for p in posts:
        records.append((
            str(p["id"]),
            str(p["created_at"]),
            str(p["date"]),
            str(p["text"]),
            int(p.get("likes", 0)),
            int(p.get("reposts", 0)),
            int(p.get("views", 0)),
            int(p.get("author_followers", 0)),
            float(p.get("sentiment_score", 0.0)),
            str(p.get("sentiment_label", "neutral")),
            str(p.get("source", "apify"))
        ))
    cursor.executemany("""
    INSERT OR REPLACE INTO normalized_posts 
    (id, created_at, date, text, likes, reposts, views, author_followers, sentiment_score, sentiment_label, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
    return len(records)

def save_daily_sentiment(df: pd.DataFrame, db_path: Optional[Path] = None) -> int:
    """Save daily aggregated sentiment DataFrame to SQLite."""
    if df.empty:
        return 0
    conn = get_connection(db_path)
    cursor = conn.cursor()
    records = []
    for idx, row in df.iterrows():
        dt_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        records.append((
            dt_str,
            int(row.get("post_count", 0)),
            int(row.get("live_post_count", 0)),
            int(row.get("sample_post_count", 0)),
            float(row.get("sentiment_mean", 0.0)),
            float(row.get("sentiment_std", 0.0)),
            int(row.get("likes_sum", 0)),
            float(row.get("likes_mean", 0.0)),
            int(row.get("reposts_sum", 0)),
            int(row.get("views_sum", 0)),
            float(row.get("followers_mean", 0.0)),
            int(row.get("is_imputed", 0)),
            str(row.get("provenance", "sample_fixture"))
        ))
    cursor.executemany("""
    INSERT OR REPLACE INTO daily_sentiment 
    (date, post_count, live_post_count, sample_post_count, sentiment_mean, sentiment_std, likes_sum, likes_mean, reposts_sum, views_sum, followers_mean, is_imputed, provenance)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
    return len(records)

def save_news_articles(articles: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """Save news articles to SQLite."""
    if not articles:
        return 0
    conn = get_connection(db_path)
    cursor = conn.cursor()
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    records = []
    for a in articles:
        records.append((
            str(a.get("url", "")),
            str(a.get("title", "Untitled")),
            str(a.get("source", "tavily")),
            str(a.get("published_date", "")),
            str(a.get("snippet", "")),
            now_iso
        ))
    cursor.executemany("""
    INSERT OR REPLACE INTO news_articles (url, title, source, published_date, snippet, ingested_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
    return len(records)

def save_metrics(metrics_list: List[Dict[str, Any]], db_path: Optional[Path] = None) -> None:
    """Save model validation metrics."""
    if not metrics_list:
        return
    conn = get_connection(db_path)
    cursor = conn.cursor()
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    records = []
    for m in metrics_list:
        records.append((
            str(m["model_name"]),
            str(m["horizon"]),
            float(m["mae"]),
            float(m["rmse"]),
            float(m["mape"]),
            float(m.get("directional_accuracy", 0.0)),
            now_iso
        ))
    cursor.executemany("""
    INSERT INTO model_metrics (model_name, horizon, mae, rmse, mape, directional_accuracy, evaluated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
