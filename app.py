"""Streamlit UI. Business logic lives in features.py, model.py, evaluation.py and
sentiment.py — this file only wires up inputs, calls those modules, and displays
results.
"""
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from evaluation import append_run_to_log, train_and_evaluate
from sentiment import fetch_recent_sentiment, load_sentiment_analyzer

st.title("📈 Stock Price Prediction (LSTM + Technical Indicators, Walk-Forward Validated)")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL):", "GOOGL")
start_date = st.date_input("Start Date", value=date.today() - timedelta(days=5 * 365))
end_date = st.date_input("End Date", value=date.today())
epochs = st.slider("Max training epochs per fold", min_value=10, max_value=150, value=100, step=10)
time_step = st.slider("Lookback window (days)", min_value=30, max_value=150, value=100, step=10)
horizon = st.slider("Forecast horizon (trading days ahead)", min_value=1, max_value=10, value=3)
n_splits = st.slider("Walk-forward folds", min_value=2, max_value=5, value=3,
                      help="More folds = a more trustworthy estimate, but trains the model that many more times.")
company_name = st.text_input("Company name (for news sentiment, e.g. 'Google' works better than 'GOOGL'):", "")
run_note = st.text_input("Note for this run (optional, e.g. what you changed):", "")

cached_train_and_evaluate = st.cache_resource(
    show_spinner="Running walk-forward validation (trains the model several times)..."
)(train_and_evaluate)

cached_load_analyzer = st.cache_resource(show_spinner="Loading VADER sentiment lexicon...")(load_sentiment_analyzer)
cached_fetch_sentiment = st.cache_data(ttl=3600, show_spinner="Fetching recent news sentiment...")(
    fetch_recent_sentiment
)

if st.button("Predict"):
    result = cached_train_and_evaluate(ticker, start_date, end_date, epochs, time_step, horizon, n_splits)

    if result is None:
        st.error("No data found. Please try a different ticker or date range.")
    elif result["error"] == "not_enough_data":
        st.error("Not enough data for this many walk-forward folds + lookback window. "
                 "Try a wider date range, fewer folds, or a smaller lookback window.")
    elif result["error"] == "not_enough_data_for_folds":
        st.error("Every walk-forward fold ended up too small to train on. "
                 "Try a wider date range, fewer folds, or a smaller lookback window.")
    else:
        featured = result["featured"]
        st.subheader("📊 Raw Stock Data (with indicators)")
        st.write(featured.tail())

        st.subheader(f"📊 Walk-Forward Validation ({result['n_folds_used']} folds)")
        st.caption("Each fold trains on an earlier chronological chunk and tests on the next unseen chunk — "
                    "this is a far more trustworthy estimate than a single train/test split.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Test RMSE", f"{result['mean_test_rmse']:.2f} ± {result['std_test_rmse']:.2f}")
            st.metric("Mean Naive Baseline RMSE", f"{result['mean_naive_rmse']:.2f}")
        with col2:
            st.metric("Mean R² Score", f"{result['mean_r2']:.2f}")
            st.metric("Mean Directional Accuracy", f"{result['mean_dir_acc']:.1f}% ± {result['std_dir_acc']:.1f}%",
                       help="Random guessing scores ~50%.")

        st.write("Per-fold test RMSE:", result["fold_test_rmse"])
        st.caption(f"Average epochs actually run per fold (early stopping): {result['mean_epochs_run']:.0f}")

        if result["mean_test_rmse"] >= result["mean_naive_rmse"] or result["mean_dir_acc"] <= 50:
            st.warning("Across folds, the LSTM did not reliably beat the naive baseline / random-chance "
                       "direction guessing — treat its predictions with caution.")

        append_run_to_log({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "epochs_requested": epochs,
            "time_step": time_step,
            "horizon": horizon,
            "n_splits": n_splits,
            "n_folds_used": result["n_folds_used"],
            "mean_test_rmse": round(result["mean_test_rmse"], 4),
            "std_test_rmse": round(result["std_test_rmse"], 4),
            "mean_naive_rmse": round(result["mean_naive_rmse"], 4),
            "mean_r2": round(result["mean_r2"], 4),
            "mean_dir_acc": round(result["mean_dir_acc"], 2),
            "std_dir_acc": round(result["std_dir_acc"], 2),
            "note": run_note,
        })

        st.subheader("📈 Actual vs Predicted (final model, trained on full history)")
        st.caption("Shown for visual context only — the metrics above (walk-forward) are the real evaluation.")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
        ax_pred.plot(result["plot_actual_price"], label="Actual Price")
        offset = len(result["plot_actual_price"]) - len(result["plot_predicted_price"])
        ax_pred.plot(np.arange(offset, len(result["plot_actual_price"])), result["plot_predicted_price"],
                     label=f"Predicted Price ({horizon}d ahead, in-sample)", color="green", alpha=0.7)
        ax_pred.set_title(f"{ticker} Price Prediction")
        ax_pred.set_xlabel("Time")
        ax_pred.set_ylabel("Price")
        ax_pred.legend()
        st.pyplot(fig_pred)

        st.subheader(f"📈 Forecast: {horizon} Trading Days Ahead")
        change_pct = (result["forecast_price"] / result["last_close"] - 1) * 100
        st.metric(f"Predicted price ({horizon}d from last close)", f"{result['forecast_price']:.2f}",
                   delta=f"{change_pct:+.2f}%")
        st.caption(f"Last observed close: {result['last_close']:.2f}")

        st.subheader("🧭 Combined Outlook (price signal + news sentiment)")
        st.warning(
            "**Read this before the numbers below**: separate validation testing found neither "
            "the price/technical-indicator signal (walk-forward directional accuracy ~48-53%, no "
            "better than chance) nor news sentiment (Pearson r=-0.13, p=0.36, n=51 real trading days) "
            "to be statistically significant predictors of this stock's short-term return. This section "
            "is an exploratory view of what both signals currently show, not a validated trading signal."
        )

        sentiment_query = company_name.strip() or ticker
        analyzer = cached_load_analyzer()
        sentiment_result = cached_fetch_sentiment(sentiment_query, analyzer)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Price signal (LSTM forecast)", f"{change_pct:+.2f}%")
        with col2:
            if sentiment_result is None:
                st.metric("News sentiment (last 14 days)", "unavailable")
            else:
                st.metric("News sentiment (last 14 days)", f"{sentiment_result['avg_sentiment']:+.3f}",
                           help=f"VADER compound score averaged over {sentiment_result['n_headlines']} headlines "
                                f"for '{sentiment_query}'. Range -1 (very negative) to +1 (very positive).")

        price_lean = "bullish" if change_pct > 0.1 else "bearish" if change_pct < -0.1 else "flat"
        if sentiment_result is None:
            combined = "No sentiment data available — showing price signal only."
        else:
            sentiment_lean = ("bullish" if sentiment_result["avg_sentiment"] > 0.05
                               else "bearish" if sentiment_result["avg_sentiment"] < -0.05 else "flat")
            if price_lean == sentiment_lean and price_lean != "flat":
                combined = f"Both signals lean **{price_lean}** (they agree)."
            elif "flat" in (price_lean, sentiment_lean):
                combined = "At least one signal is flat/neutral — no clear combined lean."
            else:
                combined = f"Signals **disagree** (price: {price_lean}, sentiment: {sentiment_lean}) — mixed."

        st.write(combined)
