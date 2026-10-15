"""
Live API verification script.
Tests real live requests against Apify, Tavily, and Exa APIs.
Verifies retrieved items, schemas, engagement fields, timestamps, and normalization.
Never prints or logs secret token values.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load local .env without printing secrets
load_dotenv()

from src.config import APIFY_TOKEN, APIFY_ACTOR_ID, TAVILY_API_KEY, EXA_API_KEY
from src.ingestion.apify_ingestion import ApifyTwitterIngestion
from src.ingestion.news_ingestion import NewsIngestion

import sys
# Set utf-8 encoding on stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_apify():
    print("=" * 70)
    print(" 1. TESTING APIFY (X/TWITTER) LIVE INGESTION")
    print("=" * 70)
    if not APIFY_TOKEN or APIFY_TOKEN.startswith("your_"):
        print("[FAIL] APIFY_TOKEN is NOT configured or is currently a placeholder in .env.")
        print("   Required action: Replace the placeholder in .env with your real Apify API token.")
        print(f"   Target Actor ID: {APIFY_ACTOR_ID}")
        print("   Documented Input Schema:")
        print("     {")
        print('       "searchTerms": ["#Bitcoin OR $BTC OR Bitcoin"],')
        print('       "maxItems": 20,')
        print('       "sort": "Latest",')
        print('       "tweetLanguage": "en"')
        print("     }")
        return False

    print(f"[OK] APIFY_TOKEN is detected (length: {len(APIFY_TOKEN)} chars).")
    print(f"[OK] Using configured actor: {APIFY_ACTOR_ID}")
    client = ApifyTwitterIngestion()
    print("  Triggering live Apify actor run (requesting up to 20 live tweets)...")

    raw_items, normalized = client.fetch_posts_live(
        search_query="#Bitcoin OR $BTC",
        max_items=20,
        timeout_seconds=180
    )

    print(f"  Raw items returned: {len(raw_items)}")
    print(f"  Normalized posts:   {len(normalized)}")

    if normalized:
        print("\n  Sample Verified Live Post:")
        p0 = normalized[0]
        print(f"    - ID:        {p0['id']}")
        print(f"    - Timestamp: {p0['created_at']} (Date: {p0['date']})")
        print(f"    - Text:      {p0['text'][:90]}...")
        print(f"    - Likes:     {p0['likes']}")
        print(f"    - Reposts:   {p0['reposts']}")
        print(f"    - Views:     {p0['views']}")
        print(f"    - Followers: {p0['author_followers']}")
        print(f"    - Source:    {p0['source']}")
        print("[OK] Apify Live Ingestion: VERIFIED SUCCESSFUL")
        return True
    else:
        print("[FAIL] Apify live run did not return items or failed.")
        return False

def test_tavily():
    print("\n" + "=" * 70)
    print(" 2. TESTING TAVILY (NEWS) LIVE INGESTION")
    print("=" * 70)
    if not TAVILY_API_KEY or TAVILY_API_KEY.startswith("your_"):
        print("[FAIL] TAVILY_API_KEY is NOT configured or is currently a placeholder in .env.")
        print("   Required action: Replace the placeholder in .env with your real Tavily API key.")
        return False

    print(f"[OK] TAVILY_API_KEY is detected (length: {len(TAVILY_API_KEY)} chars).")
    client = NewsIngestion()
    articles = client.fetch_tavily_news(max_results=5)
    print(f"  Articles retrieved: {len(articles)}")
    if articles:
        for idx, art in enumerate(articles[:2], 1):
            print(f"  [{idx}] {art['title']}")
            print(f"      URL:     {art['url']}")
            print(f"      Date:    {art['published_date']}")
            print(f"      Snippet: {art['snippet'][:90]}...")
        print("[OK] Tavily Live Ingestion: VERIFIED SUCCESSFUL")
        return True
    else:
        print("[FAIL] Tavily search returned 0 items or encountered error.")
        return False

def test_exa():
    print("\n" + "=" * 70)
    print(" 3. TESTING EXA (NEWS) LIVE INGESTION (OPTIONAL)")
    print("=" * 70)
    if not EXA_API_KEY or EXA_API_KEY.startswith("your_"):
        print("[INFO] EXA_API_KEY is not configured or is placeholder (optional).")
        return None

    print(f"[OK] EXA_API_KEY is detected (length: {len(EXA_API_KEY)} chars).")
    client = NewsIngestion()
    articles = client.fetch_exa_news(num_results=3)
    print(f"  Articles retrieved: {len(articles)}")
    if articles:
        for idx, art in enumerate(articles[:2], 1):
            print(f"  [{idx}] {art['title']} ({art['url']})")
        print("[OK] Exa Live Ingestion: VERIFIED SUCCESSFUL")
        return True
    else:
        print("[FAIL] Exa search returned 0 items.")
        return False

def main():
    apify_ok = test_apify()
    tavily_ok = test_tavily()
    exa_ok = test_exa()

    print("\n" + "=" * 70)
    print(" SUMMARY OF LIVE CREDENTIALS AUDIT")
    print("=" * 70)
    print(f"  Apify Live X/Twitter: {'READY / LIVE' if apify_ok else 'NOT CONFIGURED / PENDING'}")
    print(f"  Tavily Live News:     {'READY / LIVE' if tavily_ok else 'NOT CONFIGURED / PENDING'}")
    print(f"  Exa Live News:        {'READY / LIVE' if exa_ok else 'OPTIONAL / PENDING'}")
    print("=" * 70 + "\n")

    if not apify_ok or not tavily_ok:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
