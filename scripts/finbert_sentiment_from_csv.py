"""
Takes the finbert_sentiment.csv produced by finbert_colab_notebook.ipynb (run on
Google Colab, since this machine can't reliably run PyTorch + FinBERT locally) and
checks whether real FinBERT sentiment correlates with next-day stock return.

Run with: streamlit run finbert_sentiment_from_csv.py
"""
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import pearsonr

st.title("📰 FinBERT News Sentiment vs Next-Day Return")
st.caption("Uses real FinBERT scores from finbert_colab_notebook.ipynb (run on Colab), "
           "not the local VADER fallback.")

ticker = st.text_input("Ticker Symbol (for price data):", "GOOGL")
csv_file = st.file_uploader("Upload finbert_sentiment.csv from Colab", type="csv")

if csv_file is not None:
    headlines = pd.read_csv(csv_file, parse_dates=["date"])
    headlines["date"] = headlines["date"].dt.date

    st.subheader("📊 Scored Headlines")
    st.dataframe(headlines.sort_values("date"), use_container_width=True)

    daily_sentiment = headlines.groupby("date")["sentiment"].mean().reset_index()
    daily_sentiment.columns = ["date", "avg_sentiment"]

    if st.button("Run Correlation Check"):
        price_start = daily_sentiment["date"].min() - pd.Timedelta(days=2)
        price_end = daily_sentiment["date"].max() + pd.Timedelta(days=5)
        prices = yf.download(ticker, start=price_start, end=price_end)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

        prices = prices[["Close"]].reset_index()
        prices["date"] = prices["Date"].dt.date
        prices["next_day_return"] = prices["Close"].pct_change().shift(-1)

        merged = daily_sentiment.merge(prices[["date", "next_day_return"]], on="date", how="inner").dropna()

        if len(merged) < 5:
            st.warning(f"Only {len(merged)} matched (headline date, trading day) pairs — "
                       "too few to draw any conclusion.")
            st.stop()

        r, p_value = pearsonr(merged["avg_sentiment"], merged["next_day_return"])

        st.subheader("📈 Correlation: FinBERT Sentiment vs Next-Day Return")
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
