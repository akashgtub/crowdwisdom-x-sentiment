"""
Apify X/Twitter ingestion module.
Uses Apify REST API to collect posts related to crypto/asset sentiment.
Handles actor input schemas, rate limits, timeouts, raw/normalized separation,
schema normalization, deduplication, and honest reporting of collected counts.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

from src.config import (
    APIFY_TOKEN,
    APIFY_ACTOR_ID,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    USE_SAMPLE_DATA_FALLBACK,
    DB_PATH
)
from src.db import save_raw_post, save_normalized_posts

logger = logging.getLogger(__name__)

class ApifyTwitterIngestion:
    """Client for triggering and ingesting X/Twitter data from Apify actors."""

    BASE_URL = "https://api.apify.com/v2"

    def __init__(self, token: Optional[str] = None, actor_id: Optional[str] = None):
        self.token = (token or APIFY_TOKEN).strip()
        self.actor_id = (actor_id or APIFY_ACTOR_ID).strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CrowdWisdomTrading-Agent/1.0"})

    def is_configured(self) -> bool:
        """Return True if an API token is present and not a placeholder."""
        return bool(self.token and not self.token.startswith("your_") and len(self.token) > 5)

    def build_actor_input(self, search_query: str = "Bitcoin OR #BTC OR $BTC", max_items: int = 100) -> Dict[str, Any]:
        """
        Build actor input schema adapted to known Twitter scraping actors.
        Supports apidojo/tweet-scraper, quacker/twitter-scraper, and generic fallback.
        """
        actor = self.actor_id.lower()
        if "apidojo" in actor:
            return {
                "searchTerms": [search_query],
                "maxItems": max_items,
                "sort": "Latest",
                "tweetLanguage": "en"
            }
        elif "quacker" in actor:
            return {
                "searchQueries": [search_query],
                "maxTweets": max_items,
                "onlyVerifiedUsers": False
            }
        else:
            # Generic multi-key payload compatible with multiple Apify actors
            return {
                "searchTerms": [search_query],
                "searchQueries": [search_query],
                "queries": [search_query],
                "query": search_query,
                "maxItems": max_items,
                "maxTweets": max_items,
                "language": "en"
            }

    def fetch_posts_live(
        self,
        search_query: str = "Bitcoin OR #BTC OR $BTC",
        max_items: int = 100,
        timeout_seconds: int = 180
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute an Apify actor run and fetch resulting dataset items.
        
        Returns:
            Tuple[List[Dict], List[Dict]]: (raw_items, normalized_posts)
        """
        if not self.is_configured():
            logger.warning("APIFY_TOKEN is not configured. Live scraping cannot proceed.")
            return [], []

        # Replace slash with tilde for Apify URL format (username~actor-name)
        formatted_actor_id = self.actor_id.replace("/", "~")
        run_endpoint = f"{self.BASE_URL}/acts/{formatted_actor_id}/runs?token={self.token}"
        actor_input = self.build_actor_input(search_query=search_query, max_items=max_items)

        logger.info("Starting Apify actor '%s' run (query='%s', max_items=%d)...", self.actor_id, search_query, max_items)
        try:
            resp = self.session.post(run_endpoint, json=actor_input, timeout=30)
            if resp.status_code == 401:
                logger.error("Apify authentication failed (HTTP 401). Invalid APIFY_TOKEN.")
                return [], []
            elif resp.status_code == 404:
                logger.error("Apify actor '%s' not found (HTTP 404). Check APIFY_ACTOR_ID.", self.actor_id)
                return [], []
            elif not resp.ok:
                logger.error("Apify run request failed with HTTP %d: %s", resp.status_code, resp.text)
                return [], []

            run_data = resp.json().get("data", {})
            run_id = run_data.get("id")
            dataset_id = run_data.get("defaultDatasetId")
            logger.info("Apify run started with run_id=%s, dataset_id=%s", run_id, dataset_id)

            # Poll for completion
            start_time = time.time()
            status = run_data.get("status")
            while status not in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                if time.time() - start_time > timeout_seconds:
                    logger.warning("Apify actor run %s timed out after %d seconds.", run_id, timeout_seconds)
                    break
                time.sleep(5)
                poll_url = f"{self.BASE_URL}/acts/{formatted_actor_id}/runs/{run_id}?token={self.token}"
                poll_resp = self.session.get(poll_url, timeout=15)
                if poll_resp.ok:
                    status = poll_resp.json().get("data", {}).get("status")
                    logger.debug("Apify run %s status: %s", run_id, status)

            if status != "SUCCEEDED":
                logger.warning("Apify actor run did not succeed (final status: %s).", status)
                return [], []

            # Fetch dataset items
            dataset_url = f"{self.BASE_URL}/datasets/{dataset_id}/items?token={self.token}&format=json"
            items_resp = self.session.get(dataset_url, timeout=30)
            if not items_resp.ok:
                logger.error("Failed to retrieve dataset items: HTTP %d", items_resp.status_code)
                return [], []

            raw_items = items_resp.json()
            if not isinstance(raw_items, list):
                raw_items = []

            logger.info("Successfully fetched %d raw items from Apify dataset %s", len(raw_items), dataset_id)

            # Save raw payload
            raw_path = RAW_DATA_DIR / f"apify_raw_{run_id}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, indent=2)

            # Persist raw items to DB
            for idx, item in enumerate(raw_items):
                item_id = str(item.get("id") or item.get("id_str") or f"{run_id}_{idx}")
                save_raw_post(post_id=item_id, source="apify", raw_data=item)

            normalized = self.normalize_posts(raw_items, source="live_apify")
            save_normalized_posts(normalized)
            return raw_items, normalized

        except requests.exceptions.RequestException as e:
            logger.error("Network or connection error during Apify call: %s", e)
            return [], []

    def normalize_posts(self, raw_items: List[Dict[str, Any]], source: str = "live_apify") -> List[Dict[str, Any]]:
        """
        Normalize raw posts from various actor schemas into a consistent schema:
        - id
        - created_at (ISO UTC)
        - date (YYYY-MM-DD)
        - text
        - likes
        - reposts
        - views
        - author_followers
        - source
        Deduplicates by post id.
        """
        normalized: List[Dict[str, Any]] = []
        seen_ids = set()

        for idx, item in enumerate(raw_items):
            # Extract ID
            raw_id = item.get("id") or item.get("id_str") or item.get("tweet_id") or item.get("url")
            post_id = str(raw_id) if raw_id else f"{source}_{idx}_{hash(str(item.get('text', '')))}"
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            # Extract Timestamp
            raw_ts = (
                item.get("createdAt")
                or item.get("created_at")
                or item.get("timestamp")
                or item.get("date")
                or item.get("timeParsed")
            )
            dt_obj = self._parse_timestamp(raw_ts)
            iso_utc = dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
            date_str = dt_obj.strftime("%Y-%m-%d")

            # Extract Text
            text = (
                item.get("text")
                or item.get("full_text")
                or item.get("tweetText")
                or item.get("content")
                or ""
            ).strip()

            # Extract Engagement Metrics
            likes = int(
                item.get("likeCount")
                or item.get("likes")
                or item.get("favorite_count")
                or item.get("favoriteCount")
                or 0
            )
            reposts = int(
                item.get("retweetCount")
                or item.get("retweets")
                or item.get("repostCount")
                or item.get("retweet_count")
                or 0
            )
            views = int(
                item.get("viewCount")
                or item.get("views")
                or item.get("impressions_count")
                or 0
            )

            # Extract Author Followers
            author_info = item.get("author") or item.get("user") or {}
            if isinstance(author_info, dict):
                followers = int(
                    author_info.get("followers")
                    or author_info.get("followers_count")
                    or author_info.get("subscribers")
                    or 0
                )
            else:
                followers = int(item.get("authorFollowersCount") or 0)

            normalized.append({
                "id": post_id,
                "created_at": iso_utc,
                "date": date_str,
                "text": text,
                "likes": max(0, likes),
                "reposts": max(0, reposts),
                "views": max(0, views),
                "author_followers": max(0, followers),
                "source": source
            })

        logger.info("Normalized %d posts (from %d raw items).", len(normalized), len(raw_items))
        return normalized

    @staticmethod
    def _parse_timestamp(raw_ts: Any) -> datetime:
        """Parse arbitrary timestamp string or numeric timestamp to UTC datetime."""
        if not raw_ts:
            return datetime.now(timezone.utc)
        if isinstance(raw_ts, (int, float)):
            # Check if seconds or milliseconds
            if raw_ts > 1e11:
                return datetime.fromtimestamp(raw_ts / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        
        ts_str = str(raw_ts).strip()
        # Common twitter / ISO formats
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%a %b %d %H:%M:%S %z %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ):
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue

        return datetime.now(timezone.utc)

    def get_or_create_sample_dataset(self, days: int = 180) -> List[Dict[str, Any]]:
        """
        Generate or load a realistic sample dataset of historical X/Twitter posts
        covering the 180-day price window for reproducible testing when API keys are pending.
        Clearly tagged as 'sample_fixture'.
        """
        sample_path = DATA_DIR / "sample_posts_fixture.json"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("Generating realistic sample X/Twitter posts fixture covering %d days...", days)
        import random
        from datetime import timedelta
        
        # Seed for full reproducibility
        rng = random.Random(42)
        end_date = datetime.now(timezone.utc).date()
        
        sample_tweets_templates = [
            ("Bitcoin showing tremendous strength as institutional inflows accelerate! Bullish on $BTC breakout.", 0.75),
            ("Massive sell-off on crypto exchanges, fear sentiment high, support level broken.", -0.65),
            ("Bitcoin consolidating around key moving average, waiting for next directional move.", 0.05),
            ("Bitcoin hash rate hits new all-time high! Fundamentals remain strong regardless of short-term price chop.", 0.60),
            ("Regulatory concerns loom as SEC announces investigation into crypto trading desks.", -0.55),
            ("BTC ETF trading volume sets new daily record! Adoption curve is steep.", 0.70),
            ("Market looks overheated, RSI divergence signaling potential short-term pullback in Bitcoin.", -0.40),
            ("Sideways volatility compression usually precedes an explosive move. Accumulating BTC on dips.", 0.35),
            ("Whale alerts: 5,000 BTC moved from cold storage to Binance. Watch out for dump.", -0.50),
            ("Macro liquidity turning favorable. Global rate cuts will fuel next Bitcoin bull leg.", 0.65),
        ]

        sample_posts = []
        for d in range(days):
            current_date = end_date - timedelta(days=(days - 1 - d))
            # 5 to 15 posts per day
            daily_post_count = rng.randint(5, 15)
            for p_idx in range(daily_post_count):
                template_text, expected_sentiment = rng.choice(sample_tweets_templates)
                hour = rng.randint(0, 23)
                minute = rng.randint(0, 59)
                post_time = datetime(current_date.year, current_date.month, current_date.day, hour, minute, tzinfo=timezone.utc)
                
                likes = int(rng.expovariate(1/120.0))
                reposts = int(likes * rng.uniform(0.1, 0.4))
                views = int(likes * rng.uniform(20.0, 100.0)) + rng.randint(500, 5000)
                followers = int(rng.expovariate(1/4000.0)) + 150

                sample_posts.append({
                    "id": f"sample_{current_date.strftime('%Y%m%d')}_{p_idx}",
                    "created_at": post_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "date": current_date.strftime("%Y-%m-%d"),
                    "text": template_text,
                    "likes": likes,
                    "reposts": reposts,
                    "views": views,
                    "author_followers": followers,
                    "source": "sample_fixture"
                })

        # Save to disk
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_posts, f, indent=2)
            
        logger.info("Saved %d sample posts to %s", len(sample_posts), sample_path)
        return sample_posts

    def collect(self, search_query: str = "Bitcoin OR #BTC OR $BTC", max_items: int = 100) -> Dict[str, Any]:
        """
        Main entry point for X post collection:
        - Attempts live Apify scraping if APIFY_TOKEN is set.
        - Reports actual API collected count honestly.
        - Uses sample fixture if enabled and no live token is provided.
        """
        actual_collected = 0
        raw_items = []
        normalized_posts = []
        source_mode = "none"

        if self.is_configured():
            logger.info("APIFY_TOKEN found. Executing live Apify collection...")
            raw_items, normalized_posts = self.fetch_posts_live(search_query=search_query, max_items=max_items)
            actual_collected = len(raw_items)
            source_mode = "live_apify"
        else:
            logger.info("No APIFY_TOKEN provided in environment. Live Apify collection skipped.")

        if actual_collected == 0 and USE_SAMPLE_DATA_FALLBACK:
            logger.warning("Using offline reproducible sample posts fixture for pipeline continuity.")
            normalized_posts = self.get_or_create_sample_dataset(days=180)
            source_mode = "sample_fixture"

        # Persist normalized posts to DB
        if normalized_posts:
            save_normalized_posts(normalized_posts)

        return {
            "token_configured": self.is_configured(),
            "actor_id": self.actor_id,
            "actual_api_collected_count": actual_collected,
            "total_normalized_count": len(normalized_posts),
            "source_mode": source_mode,
            "posts": normalized_posts
        }
