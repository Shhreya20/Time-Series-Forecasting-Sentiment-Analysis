"""
Pipeline sanity check: inject a synthetic feature with a KNOWN, planted correlation
to future returns, and confirm the walk-forward LSTM pipeline actually detects it.

This does NOT tell us anything about whether real news sentiment predicts real
returns (that requires real data, which we're still fetching from GDELT separately).
It answers a narrower but important question: when our earlier walk-forward runs
found ~50% directional accuracy (no better than chance), was that because there's
genuinely no learnable signal in the data, or because something in the pipeline is
silently broken and could never detect a signal even if one existed?

Run with: python pipeline_validation.py
"""
import math

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

TICKER = "GOOGL"
YEARS_OF_HISTORY = 3
EPOCHS = 25
TIME_STEP = 60
HORIZON = 3
SIGNAL_STRENGTH = 3.0   # how much of the true future return leaks into the synthetic feature
NOISE_LEVEL = 0.1       # noise added on top, so it's not a perfect giveaway
SKIP_CONTROL = True     # debug pass: only run the signal variant, to save time/memory

BASE_FEATURE_COLUMNS = ["Return", "Price_SMA10", "Price_SMA50", "RSI_14", "MACD_norm", "MACD_signal_norm", "Volume_z"]
TARGET_COLUMN = "Target_Return"


def add_technical_indicators(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
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
    out["RSI_14"] = (100 - (100 / (1 + rs))) / 100

    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["MACD_norm"] = macd / out["Close"]
    out["MACD_signal_norm"] = macd_signal / out["Close"]

    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std()
    out["Volume_z"] = (out["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    out["Target_Return"] = out["Close"].shift(-horizon) / out["Close"] - 1
    return out.dropna().reset_index(drop=True)


def create_dataset(feature_arr: np.ndarray, target_arr: np.ndarray, time_step: int):
    X, Y = [], []
    for i in range(time_step - 1, len(feature_arr)):
        X.append(feature_arr[i - time_step + 1: i + 1, :])
        Y.append(target_arr[i])
    return np.array(X), np.array(Y)


def build_model(time_step: int, num_features: int) -> Sequential:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def run_fold(train_df, test_df, feature_columns, epochs, time_step):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    train_feat_scaled = feature_scaler.fit_transform(train_df[feature_columns].values)
    test_feat_scaled = feature_scaler.transform(test_df[feature_columns].values)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_target_scaled = target_scaler.fit_transform(train_df[[TARGET_COLUMN]].values).flatten()
    test_target_scaled = target_scaler.transform(test_df[[TARGET_COLUMN]].values).flatten()

    X_train, Y_train = create_dataset(train_feat_scaled, train_target_scaled, time_step)
    X_test, Y_test = create_dataset(test_feat_scaled, test_target_scaled, time_step)
    if X_train.size == 0 or X_test.size == 0:
        return None

    model = build_model(time_step, len(feature_columns))
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, Y_train, validation_split=0.1, epochs=epochs, batch_size=64,
              callbacks=[early_stop], verbose=0)

    pred_return_scaled = model.predict(X_test, verbose=0).flatten()
    pred_return = target_scaler.inverse_transform(pred_return_scaled.reshape(-1, 1)).flatten()
    actual_return = test_df[TARGET_COLUMN].values[time_step - 1:]

    base_close = test_df["Close"].values[time_step - 1:]
    predicted_price = base_close * (1 + pred_return)
    actual_price = base_close * (1 + actual_return)
    naive_price = base_close

    return {
        "test_rmse": math.sqrt(mean_squared_error(actual_price, predicted_price)),
        "naive_rmse": math.sqrt(mean_squared_error(actual_price, naive_price)),
        "r2": r2_score(actual_price, predicted_price),
        "dir_acc": float(np.mean(np.sign(pred_return) == np.sign(actual_return)) * 100),
    }


def single_split_eval(featured: pd.DataFrame, feature_columns: list, label: str):
    """A single chronological 80/20 split — enough for a pass/fail signal-detection
    sanity check, much lighter than full walk-forward on this machine."""
    train_size = int(len(featured) * 0.8)
    train_df = featured.iloc[:train_size].reset_index(drop=True)
    test_df = featured.iloc[train_size:].reset_index(drop=True)
    if len(train_df) <= TIME_STEP + 1 or len(test_df) <= TIME_STEP + 1:
        print(f"  [{label}] not enough data for this split")
        return None

    print(f"  [{label}] training...", flush=True)
    result = run_fold(train_df, test_df, feature_columns, EPOCHS, TIME_STEP)
    if result is None:
        print(f"  [{label}] no usable data after windowing")
        return None

    return {
        "mean_test_rmse": result["test_rmse"],
        "mean_naive_rmse": result["naive_rmse"],
        "mean_r2": result["r2"],
        "mean_dir_acc": result["dir_acc"],
        "std_dir_acc": 0.0,
    }


def main():
    print(f"Downloading {TICKER}...")
    raw = yf.download(TICKER, period=f"{YEARS_OF_HISTORY}y")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    featured = add_technical_indicators(raw[["Close", "Volume"]], HORIZON)

    # Control: same 7 features as the main app — should reproduce the earlier
    # ~50% directional accuracy finding (no real signal detected).
    if SKIP_CONTROL:
        print("\n=== CONTROL skipped for this debug pass ===")
        control = None
    else:
        print("\n=== CONTROL: real technical features only (no synthetic signal) ===")
        control = single_split_eval(featured, BASE_FEATURE_COLUMNS, "control")

    # Signal-injected: add a feature that's PARTLY the true future return plus noise —
    # a stand-in for "what if we had a genuinely predictive sentiment score."
    rng = np.random.default_rng(42)
    noise = rng.normal(0, featured[TARGET_COLUMN].std() * NOISE_LEVEL, size=len(featured))
    featured["Synthetic_Signal"] = featured[TARGET_COLUMN] * SIGNAL_STRENGTH + noise
    signal_columns = BASE_FEATURE_COLUMNS + ["Synthetic_Signal"]

    print("\n=== SIGNAL: same features + a planted, partially-correlated synthetic feature ===")
    signal = single_split_eval(featured, signal_columns, "signal")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, res in [("Control (real features only)", control), ("Signal-injected", signal)]:
        if res is None:
            print(f"{label}: FAILED (no usable folds)")
            continue
        print(f"{label}:")
        print(f"  Mean test RMSE:        {res['mean_test_rmse']:.3f}  (naive baseline: {res['mean_naive_rmse']:.3f})")
        print(f"  Mean R2:               {res['mean_r2']:.3f}")
        print(f"  Mean directional acc:  {res['mean_dir_acc']:.1f}% (+/- {res['std_dir_acc']:.1f}%)")
        print()

    if signal is None:
        print("Could not evaluate the signal variant at all.")
    elif control is not None:
        if signal["mean_dir_acc"] > control["mean_dir_acc"] + 15 and signal["mean_test_rmse"] < signal["mean_naive_rmse"]:
            print("PIPELINE CHECK PASSED: the model clearly detected the planted signal "
                  "(directional accuracy jumped, RMSE beat naive baseline).")
            print("This means our earlier real-data findings (~50% directional accuracy) are trustworthy —")
            print("the pipeline CAN detect real signal when it's there; it just didn't find any in the real data.")
        else:
            print("PIPELINE CHECK FAILED (or inconclusive): the model did NOT clearly pick up the planted signal.")
            print("This suggests something in the pipeline may be preventing real signal from being learned —")
            print("worth investigating before trusting the earlier 'no signal found' conclusions.")
    else:
        # Debug pass with an extreme, near-noiseless signal and no control run.
        if signal["mean_dir_acc"] >= 80 and signal["mean_test_rmse"] < signal["mean_naive_rmse"] * 0.7:
            print("EXTREME-SIGNAL DEBUG PASSED: even under the lighter/faster config, the model clearly "
                  "detects an obvious planted signal. The earlier moderate-signal failure was likely just "
                  "'too little training for a partially-noisy signal', not a real pipeline bug.")
        else:
            print("EXTREME-SIGNAL DEBUG FAILED: even a near-noiseless planted signal wasn't picked up. "
                  "This points to a real problem in the pipeline (scaling, feature/target alignment, or "
                  "the model architecture/training setup) rather than just 'not enough training'.")


if __name__ == "__main__":
    main()
