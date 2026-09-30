"""Walk-forward training/evaluation orchestration, plus the experiment log.

Walk-forward validation (train on an early chronological chunk, test on the next
unseen chunk, repeat across multiple folds, average the results) is what makes the
reported metrics trustworthy instead of dependent on one lucky/unlucky split.
"""
import math
import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from features import FEATURE_COLUMNS, TARGET_COLUMN, add_technical_indicators, create_dataset
from model import build_model, fit_with_early_stopping

LOG_PATH = os.path.join(os.path.dirname(__file__), "experiment_log.csv")
LOG_COLUMNS = [
    "timestamp", "ticker", "start_date", "end_date", "epochs_requested", "time_step", "horizon", "n_splits",
    "n_folds_used", "mean_test_rmse", "std_test_rmse", "mean_naive_rmse", "mean_r2",
    "mean_dir_acc", "std_dir_acc", "note",
]


def append_run_to_log(row: dict) -> None:
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


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
