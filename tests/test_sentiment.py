"""
Unit tests for sentiment analysis and financial lexicon extensions.
"""

import pytest
from src.sentiment.sentiment_analyzer import FinancialSentimentAnalyzer

def test_financial_sentiment_lexicon_enrichment():
    analyzer = FinancialSentimentAnalyzer(use_financial_lexicon=True)

    bullish_text = "Massive rally with bullish institutional inflows and accumulation at support!"
    res_bull = analyzer.score_text(bullish_text)
    assert res_bull["compound"] > 0.4
    assert res_bull["label"] == "positive"

    bearish_text = "Sudden crash and selloff! Long positions liquidated and rekt in massive dump."
    res_bear = analyzer.score_text(bearish_text)
    assert res_bear["compound"] < -0.4
    assert res_bear["label"] == "negative"

    neutral_text = "Bitcoin traded between 81,000 and 81,500 during the session."
    res_neu = analyzer.score_text(neutral_text)
    assert -0.05 < res_neu["compound"] < 0.05
    assert res_neu["label"] == "neutral"

def test_empty_text_scoring():
    analyzer = FinancialSentimentAnalyzer()
    res = analyzer.score_text("")
    assert res["compound"] == 0.0
    assert res["label"] == "neutral"
