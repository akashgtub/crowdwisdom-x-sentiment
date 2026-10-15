"""
Unit tests for Apify X/Twitter ingestion and normalization.
"""

import pytest
from src.ingestion.apify_ingestion import ApifyTwitterIngestion

def test_apify_normalization_multiple_schemas():
    client = ApifyTwitterIngestion()
    # Test items mimicking different actor schemas (apidojo vs quacker vs generic)
    raw_items = [
        {
            "id": "tweet_1",
            "createdAt": "2026-05-01T12:00:00Z",
            "full_text": "Bitcoin is pumping! Bullish market structure.",
            "likeCount": 150,
            "retweetCount": 45,
            "viewCount": 5000,
            "author": {"followers": 12000}
        },
        {
            "id_str": "tweet_2",
            "date": "2026-05-01 14:30:00",
            "text": "Market selling off heavily, support broken.",
            "likes": 80,
            "retweets": 20,
            "views": 2500,
            "user": {"followers_count": 3000}
        },
        # Duplicate of tweet_1
        {
            "id": "tweet_1",
            "createdAt": "2026-05-01T12:00:00Z",
            "text": "Duplicate tweet"
        }
    ]

    normalized = client.normalize_posts(raw_items, source="test")
    assert len(normalized) == 2  # Deduplicated from 3 to 2

    p1 = normalized[0]
    assert p1["id"] == "tweet_1"
    assert p1["date"] == "2026-05-01"
    assert p1["likes"] == 150
    assert p1["reposts"] == 45
    assert p1["views"] == 5000
    assert p1["author_followers"] == 12000

    p2 = normalized[1]
    assert p2["id"] == "tweet_2"
    assert p2["likes"] == 80
    assert p2["author_followers"] == 3000

def test_missing_fields_graceful_handling():
    client = ApifyTwitterIngestion()
    raw_items = [{}]  # completely empty dictionary
    normalized = client.normalize_posts(raw_items, source="test")
    assert len(normalized) == 1
    p = normalized[0]
    assert p["likes"] == 0
    assert p["reposts"] == 0
    assert p["views"] == 0
    assert p["author_followers"] == 0
    assert p["text"] == ""
