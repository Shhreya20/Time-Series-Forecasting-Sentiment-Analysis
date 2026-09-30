import math
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import feedparser
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

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

# All features are stationary (bounded ratios/oscillators), not raw price levels,
# so the model isn't asked to extrapolate outside the price range it trained on.
FEATURE_COLUMNS = ["Return", "Price_SMA10", "Price_SMA50", "RSI_14", "MACD_norm", "MACD_signal_norm", "Volume_z"]
TARGET_COLUMN = "Target_Return"  # forward-looking, NEVER included in FEATURE_COLUMNS (would leak the answer)

LOG_PATH = os.path.join(os.path.dirname(__file__), "experiment_log.csv")
LOG_COLUMNS = [
    "timestamp", "ticker", "start_date", "end_date", "epochs_requested", "time_step", "horizon", "n_splits",
    "n_folds_used", "mean_test_rmse", "std_test_rmse", "mean_naive_rmse", "mean_r2",
    "mean_dir_acc", "std_dir_acc", "note",
]


def append_run_to_log(row: dict) -> None:
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


def add_technical_indicators(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Builds stationary input features plus a forward-looking N-day target return."""
    out = df.copy()
    sma_10 = out["Close"].rolling(10).mean()
    sma_50 = out["Close"].rolling(50).mean()

    out["Return"] = out["Close"].pct_change()
    out["Price_SMA10"] = out["Close"] / sma_10 - 1
    out["Price_SMA50"] = out["Close"] / sma_50 - 1

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI_14"] = (100 - (100 / (1 + rs))) / 100  # scaled to 0-1

    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["MACD_norm"] = macd / out["Close"]
    out["MACD_signal_norm"] = macd_signal / out["Close"]

    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std()
    out["Volume_z"] = (out["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # Forward-looking target: return from today's close to the close `horizon` days ahead.
    out["Target_Return"] = out["Close"].shift(-horizon) / out["Close"] - 1

    return out.dropna().reset_index(drop=True)


def create_dataset(feature_arr: np.ndarray, target_arr: np.ndarray, time_step: int):
    """X: (samples, time_step, num_features) window ending at row i (inclusive).
    Y: target_arr[i], which already represents the horizon-forward return from day i."""
    X, Y = [], []
    for i in range(time_step - 1, len(feature_arr)):
        X.append(feature_arr[i - time_step + 1: i + 1, :])
        Y.append(target_arr[i])
    return np.array(X), np.array(Y)


def build_model(time_step: int, num_features: int) -> Sequential:
    # A pipeline sanity check (planted-signal test, run separately on Colab) showed
    # the previous 3-stacked-LSTM+dropout architecture was collapsing to predicting
    # close to the mean regardless of input — too heavy for ~500-600 training samples
    # and tiny return-scale targets. This lighter architecture correctly distinguishes
    # a planted signal from noise (83% directional accuracy vs 53% for a no-signal
    # control, in that same test), so it's what should actually be trusted here.
    model = Sequential([
        LSTM(32, input_shape=(time_step, num_features)),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def fit_with_early_stopping(model: Sequential, X: np.ndarray, Y: np.ndarray, epochs: int):
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    return model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=64,
                      callbacks=[early_stop], verbose=0)


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int, time_step: int):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    train_feat_scaled = feature_scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    test_feat_scaled = feature_scaler.transform(test_df[FEATURE_COLUMNS].values)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_target_scaled = target_scaler.fit_transform(train_df[[TARGET_COLUMN]].values).flatten()
    test_target_scaled = target_scaler.transform(test_df[[TARGET_COLUMN]].values).flatten()

    X_train, Y_train = create_dataset(train_feat_scaled, train_target_scaled, time_step)
    X_test, Y_test = create_dataset(test_feat_scaled, test_target_scaled, time_step)
    if X_train.size == 0 or X_test.size == 0:
        return None

    model = build_model(time_step, len(FEATURE_COLUMNS))
    history = fit_with_early_stopping(model, X_train, Y_train, epochs)

    pred_return_scaled = model.predict(X_test, verbose=0).flatten()
    pred_return = target_scaler.inverse_transform(pred_return_scaled.reshape(-1, 1)).flatten()
    actual_return = test_df[TARGET_COLUMN].values[time_step - 1:]

    base_close = test_df["Close"].values[time_step - 1:]
    predicted_price = base_close * (1 + pred_return)
    actual_price = base_close * (1 + actual_return)
    naive_price = base_close  # naive baseline: predict no change over the horizon

    return {
        "test_rmse": math.sqrt(mean_squared_error(actual_price, predicted_price)),
        "naive_rmse": math.sqrt(mean_squared_error(actual_price, naive_price)),
        "r2": r2_score(actual_price, predicted_price),
        "dir_acc": float(np.mean(np.sign(pred_return) == np.sign(actual_return)) * 100),
        "epochs_run": len(history.history["loss"]),
    }


@st.cache_resource(show_spinner="Running walk-forward validation (trains the model several times)...")
def train_and_evaluate(ticker: str, start_date, end_date, epochs: int, time_step: int, horizon: int, n_splits: int):
    raw = yf.download(ticker, start=start_date, end=end_date)
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    featured = add_technical_indicators(raw[["Close", "Volume"]], horizon)
    if len(featured) < (n_splits + 1) * (time_step + 5):
        return {"error": "not_enough_data"}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    for train_idx, test_idx in tscv.split(np.arange(len(featured))):
        train_df = featured.iloc[train_idx].reset_index(drop=True)
        test_df = featured.iloc[test_idx].reset_index(drop=True)
        if len(train_df) <= time_step + 1 or len(test_df) <= time_step + 1:
            continue
        result = run_fold(train_df, test_df, epochs, time_step)
        if result is not None:
            fold_results.append(result)

    if not fold_results:
        return {"error": "not_enough_data_for_folds"}

    def mean_of(key):
        return float(np.mean([r[key] for r in fold_results]))

    def std_of(key):
        return float(np.std([r[key] for r in fold_results]))

    # Final production model: trained on ALL available history, used only for the
    # display chart and the live forward forecast (walk-forward folds above are what
    # give the honest generalization estimate; this model is not itself evaluated).
    final_feature_scaler = MinMaxScaler(feature_range=(0, 1))
    final_feat_scaled = final_feature_scaler.fit_transform(featured[FEATURE_COLUMNS].values)
    final_target_scaler = MinMaxScaler(feature_range=(0, 1))
    final_target_scaled = final_target_scaler.fit_transform(featured[[TARGET_COLUMN]].values).flatten()

    X_all, Y_all = create_dataset(final_feat_scaled, final_target_scaled, time_step)
    final_model = build_model(time_step, len(FEATURE_COLUMNS))
    fit_with_early_stopping(final_model, X_all, Y_all, epochs)

    pred_return_scaled = final_model.predict(X_all, verbose=0).flatten()
    pred_return = final_target_scaler.inverse_transform(pred_return_scaled.reshape(-1, 1)).flatten()
    base_close = featured["Close"].values[time_step - 1:]
    plot_predicted_price = base_close * (1 + pred_return)
    plot_actual_price = featured["Close"].values

    # Single forward pass predicting the return `horizon` days ahead from today.
    last_window = final_feat_scaled[-time_step:, :].reshape(1, time_step, len(FEATURE_COLUMNS))
    forecast_return_scaled = final_model.predict(last_window, verbose=0)[0][0]
    forecast_return = final_target_scaler.inverse_transform([[forecast_return_scaled]])[0][0]
    forecast_price = float(featured["Close"].values[-1]) * (1 + forecast_return)

    return {
        "error": None,
        "n_folds_used": len(fold_results),
        "fold_test_rmse": [r["test_rmse"] for r in fold_results],
        "mean_test_rmse": mean_of("test_rmse"), "std_test_rmse": std_of("test_rmse"),
        "mean_naive_rmse": mean_of("naive_rmse"),
        "mean_r2": mean_of("r2"),
        "mean_dir_acc": mean_of("dir_acc"), "std_dir_acc": std_of("dir_acc"),
        "mean_epochs_run": mean_of("epochs_run"),
        "featured": featured,
        "plot_actual_price": plot_actual_price,
        "plot_predicted_price": plot_predicted_price,
        "forecast_price": forecast_price,
        "last_close": float(featured["Close"].values[-1]),
    }


@st.cache_resource(show_spinner="Loading VADER sentiment lexicon...")
def load_sentiment_analyzer():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_data(ttl=3600, show_spinner="Fetching recent news sentiment...")
def fetch_recent_sentiment(query: str, period_days: int = 14):
    """Live recent-window sentiment via Google News RSS + VADER. Validated separately
    (against a real ~90-day GDELT+FinBERT sample, n=51) to show NO statistically
    significant correlation with next-day return for GOOGL — so this is included as
    a live signal to look at, not a proven predictor."""
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
    analyzer = load_sentiment_analyzer()
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


if st.button("Predict"):
    result = train_and_evaluate(ticker, start_date, end_date, epochs, time_step, horizon, n_splits)

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
        sentiment_result = fetch_recent_sentiment(sentiment_query)

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
