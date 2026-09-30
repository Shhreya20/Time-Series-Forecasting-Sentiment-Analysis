"""Live recent-window news sentiment (VADER over Google News RSS headlines).

Validated separately against a real ~90-day GDELT+FinBERT sample (n=51 matched
trading days: Pearson r=-0.13, p=0.36) to show NO statistically significant
correlation with next-day return for GOOGL. This is included in the app as a live
signal to look at, not a proven predictor — see the Combined Outlook section.
"""
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import feedparser
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def load_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def fetch_recent_sentiment(query: str, analyzer: SentimentIntensityAnalyzer, period_days: int = 14):
    """Fetches recent headlines for `query` via Google News RSS and scores them with VADER.
    Returns None if the query is empty or no dated headlines could be fetched/found."""
    if not query:
        return None

    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = urllib.request.urlopen(req, timeout=15).read()
    except Exception:
        return None
    feed = feedparser.parse(data)

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=period_days)
    scores = []
    for entry in feed.entries:
        if not getattr(entry, "published_parsed", None):
            continue
        published = datetime(*entry.published_parsed[:6])
        if published >= cutoff:
            scores.append(analyzer.polarity_scores(entry.title)["compound"])

    if not scores:
        return None
    return {"avg_sentiment": float(np.mean(scores)), "n_headlines": len(scores)}
