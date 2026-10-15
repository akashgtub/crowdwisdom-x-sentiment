"""
Configuration module for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Loads environment variables safely and defines pipeline constants.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file if it exists, without raising error if missing
load_dotenv()

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DB_PATH = DATA_DIR / "crowdwisdom.db"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# API Secrets (Read from environment, never hardcoded or printed)
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "apidojo/tweet-scraper").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()

# Pipeline Defaults
DEFAULT_ASSET = os.getenv("ASSET_TICKER", "BTC-USD").strip()
DEFAULT_HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "180"))
DEFAULT_INTERVAL = os.getenv("INTERVAL", "1d").strip()
USE_SAMPLE_DATA_FALLBACK = os.getenv("USE_SAMPLE_DATA_FALLBACK", "True").lower() in ("true", "1", "yes")

# Forecasting Horizons
HORIZONS = ["n+0", "n+1", "n+2", "n+3", "n+4", "n+5"]

def is_valid_token(token_str: str) -> bool:
    """Check if token is provided and is not a placeholder."""
    return bool(token_str and not token_str.startswith("your_") and len(token_str) > 5)

def get_api_status() -> Dict[str, Any]:
    """
    Returns boolean status and metadata for configured APIs without exposing keys.
    """
    return {
        "apify_token_configured": is_valid_token(APIFY_TOKEN),
        "apify_actor_id": APIFY_ACTOR_ID,
        "tavily_key_configured": is_valid_token(TAVILY_API_KEY),
        "exa_key_configured": is_valid_token(EXA_API_KEY),
        "asset": DEFAULT_ASSET,
        "history_days": DEFAULT_HISTORY_DAYS,
        "database_path": str(DB_PATH),
    }
