"""The best validated LSTM configuration for volatility prediction.

Tuning history (see experiment_log_volatility_lstm.csv): the original LSTM setup
(32 units, 20-day window, all 10 features) gave a non-significant result (r=0.06,
p=0.06) -- worse than a plain feedforward net (r=0.20) or RandomForest (r=0.34) on
the same data. Two changes fixed it:
  1. Shorter lookback (5 days, not 20) -- the informative features here (VIX,
     days-to-earnings) are short-memory signals; a long window just adds noise.
  2. A leaner feature set (TrailingVol, VIX, DaysToEarnings, Volume_z only, not
     all 10 technical indicators) -- fewer irrelevant inputs for a small LSTM to
     have to learn to ignore.
  3. Pooling 5 tickers (GOOGL/AAPL/MSFT/AMZN/META) for training -- LSTMs are more
     data-hungry than RandomForest, and pooling gave the single biggest jump
     (single-ticker R² was often near zero; pooled R²=0.127).

Result (3-seed average, walk-forward, n=4895): R2=0.127 (std 0.015),
correlation=0.367 (std 0.014), p<1e-140 on every seed -- the strongest, most
reproducible result in this project, comparable to or better than RandomForest's
per-ticker results.
"""
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

LEAN_FEATURES = ["TrailingVol", "VIX", "DaysToEarnings", "Volume_z"]
TIME_STEP = 5
UNITS = 16


def build_pooled_lstm_model(time_step: int = TIME_STEP, num_features: int = len(LEAN_FEATURES)) -> Sequential:
    model = Sequential([
        LSTM(UNITS, input_shape=(time_step, num_features)),
        Dense(8, activation="relu"),
        Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
