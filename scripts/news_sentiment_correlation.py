"""
Standalone check: does recent daily news sentiment (VADER) correlate with next-day
stock return? This is intentionally NOT folded into the main walk-forward LSTM app —
free news sources only give reliable dated headlines for the last few weeks, nowhere
near the years of history the main model trains on. This script answers a narrower,
honest question: over the recent window we CAN get data for, is there any hint of a
relationship at all?

Uses VADER instead of a transformer model (e.g. FinBERT) because this machine's
~3.8GB RAM can't reliably run PyTorch + a BERT-sized model without crashing.

Run with: streamlit run news_sentiment_correlation.py
"""
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import feedparser
import nltk
import pandas as pd
import streamlit as st
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr

st.title("📰 News Sentiment vs Next-Day Return (Recent-Window Correlation Check)")
st.caption("Free news sources only give reliably dated headlines for the last few weeks — "
           "this is a small correlational check, not a trained model, and not part of the main app.")

ticker = st.text_input("Ticker Symbol (for price data):", "GOOGL")
query = st.text_input("Company name to search news for (often works better than the ticker):", "Google")
period_days = st.slider("News lookback window (days)", min_value=7, max_value=60, value=30)


@st.cache_resource(show_spinner="Loading VADER sentiment lexicon...")
def load_sentiment_analyzer():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def fetch_headlines(query: str, period_days: int) -> pd.DataFrame:
    """Uses Google News' public RSS feed — lighter-weight and far less rate-limited
    than scraping the Google News search HTML page."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=15).read()
    feed = feedparser.parse(data)

    cutoff = datetime.utcnow() - timedelta(days=period_days)
    rows = []
    for entry in feed.entries:
        if not getattr(entry, "published_parsed", None):
            continue
        published = datetime(*entry.published_parsed[:6])
        if published >= cutoff:
            rows.append({"date": published.date(), "title": entry.title})
    return pd.DataFrame(rows)


def score_headlines(headlines: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    headlines = headlines.copy()
    headlines["sentiment"] = headlines["title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    return headlines


if st.button("Run Correlation Check"):
    with st.spinner(f"Fetching headlines for '{query}' (last {period_days} days)..."):
        headlines = fetch_headlines(query, period_days)

    if headlines.empty:
        st.error("No dated headlines found. Try a different/broader company name or a longer window.")
        st.stop()

    st.write(f"Found {len(headlines)} dated headlines.")

    analyzer = load_sentiment_analyzer()
    with st.spinner("Scoring headlines with VADER..."):
        scored = score_headlines(headlines, analyzer)

    daily_sentiment = scored.groupby("date")["sentiment"].mean().reset_index()
    daily_sentiment.columns = ["date", "avg_sentiment"]

    price_start = daily_sentiment["date"].min() - pd.Timedelta(days=2)
    price_end = daily_sentiment["date"].max() + pd.Timedelta(days=5)
    prices = yf.download(ticker, start=price_start, end=price_end)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices = prices[["Close"]].reset_index()
    prices["date"] = prices["Date"].dt.date
    prices["next_day_return"] = prices["Close"].pct_change().shift(-1)

    merged = daily_sentiment.merge(prices[["date", "next_day_return"]], on="date", how="inner").dropna()

    st.subheader("📊 Daily Sentiment (with headlines used)")
    st.dataframe(scored[["date", "title", "sentiment"]].sort_values("date"), use_container_width=True)

    if len(merged) < 5:
        st.warning(f"Only {len(merged)} matched (headline date, trading day) pairs — "
                   "too few to draw any conclusion. Try a wider news window.")
        st.stop()

    r, p_value = pearsonr(merged["avg_sentiment"], merged["next_day_return"])

    st.subheader("📈 Correlation: Daily Sentiment vs Next-Day Return")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pearson r", f"{r:.3f}")
    col2.metric("p-value", f"{p_value:.3f}")
    col3.metric("Sample size (days)", len(merged))

    if p_value > 0.05:
        st.warning(f"Not statistically significant (p={p_value:.3f}, n={len(merged)}) — "
                   "with this few data points, we cannot conclude sentiment predicts next-day return.")
    else:
        st.info(f"Correlation is statistically significant (p={p_value:.3f}), but with n={len(merged)} "
                "days this should still be treated as a preliminary signal, not proof.")

    st.scatter_chart(merged, x="avg_sentiment", y="next_day_return")
