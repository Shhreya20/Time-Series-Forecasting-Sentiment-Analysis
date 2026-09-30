"""Feature engineering: turns raw OHLCV data into the stationary inputs the model
trains on, plus the forward-looking target it predicts.
"""
import numpy as np
import pandas as pd

# All features are stationary (bounded ratios/oscillators), not raw price levels,
# so the model isn't asked to extrapolate outside the price range it trained on.
FEATURE_COLUMNS = ["Return", "Price_SMA10", "Price_SMA50", "RSI_14", "MACD_norm", "MACD_signal_norm", "Volume_z"]
TARGET_COLUMN = "Target_Return"  # forward-looking, NEVER included in FEATURE_COLUMNS (would leak the answer)


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
