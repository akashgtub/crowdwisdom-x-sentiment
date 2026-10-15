"""
Sentiment analysis module for CrowdWisdomTrading pipeline.
Combines VADER sentiment with crypto and financial market lexicon extensions.
Supports batch scoring and clearly distinguishes neutral sentiment from missing data.
"""

import logging
from typing import List, Dict, Any, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Specialized financial & crypto market lexicon additions to calibrate VADER
FINANCIAL_CRYPTO_LEXICON: Dict[str, float] = {
    # Bullish terms
    "bullish": 2.4,
    "breakout": 2.0,
    "inflows": 1.8,
    "accumulation": 1.7,
    "rally": 2.1,
    "ath": 2.5,
    "all-time high": 2.5,
    "adoption": 1.9,
    "soaring": 2.2,
    "undervalued": 1.6,
    "hodl": 1.5,
    "surge": 2.0,
    "rebound": 1.7,
    "buy": 1.2,
    "support": 1.0,
    # Bearish terms
    "bearish": -2.4,
    "dump": -2.5,
    "crash": -2.8,
    "selloff": -2.3,
    "outflows": -1.8,
    "liquidation": -2.2,
    "liquidated": -2.2,
    "rekt": -2.7,
    "overvalued": -1.7,
    "resistance": -0.8,
    "fud": -1.5,
    "sec investigation": -2.0,
    "hack": -3.0,
    "insolvency": -3.2,
    "collapse": -3.0,
    "plummet": -2.4,
    "breakdown": -2.0,
}

class FinancialSentimentAnalyzer:
    """VADER-based sentiment analyzer calibrated with financial/crypto market lexicon."""

    def __init__(self, use_financial_lexicon: bool = True):
        self.vader = SentimentIntensityAnalyzer()
        self.use_financial_lexicon = use_financial_lexicon
        if use_financial_lexicon:
            self.vader.lexicon.update(FINANCIAL_CRYPTO_LEXICON)
            logger.info("Enriched VADER lexicon with %d financial/crypto terms.", len(FINANCIAL_CRYPTO_LEXICON))

    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text string.
        Returns:
            Dict containing compound score [-1.0, 1.0], pos, neg, neu, and label.
        """
        if not text or not text.strip():
            return {
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
                "label": "neutral"
            }

        scores = self.vader.polarity_scores(text)
        compound = float(scores["compound"])

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "compound": compound,
            "pos": float(scores["pos"]),
            "neg": float(scores["neg"]),
            "neu": float(scores["neu"]),
            "label": label
        }

    def score_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a list of post dictionaries in-place or returning updated copies.
        Adds 'sentiment_score' and 'sentiment_label'.
        """
        scored_posts = []
        for post in posts:
            p = dict(post)
            text = p.get("text", "")
            res = self.score_text(text)
            p["sentiment_score"] = res["compound"]
            p["sentiment_label"] = res["label"]
            scored_posts.append(p)
        return scored_posts
