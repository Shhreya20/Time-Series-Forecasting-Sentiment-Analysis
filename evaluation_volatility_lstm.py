"""Walk-forward evaluation for the pooled multi-ticker LSTM volatility model
(see model_volatility_lstm.py for why this config was chosen). Pools several
tickers so the LSTM -- more data-hungry than the RandomForest in
evaluation_volatility.py -- has enough samples to learn from; windows are built
per-ticker (never across a ticker boundary) and then concatenated per fold.
"""
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from model_volatility_lstm import LEAN_FEATURES, TIME_STEP, build_pooled_lstm_model
from volatility_features import TARGET_VOL_COLUMN, build_volatility_dataset

LOG_PATH = os.path.join(os.path.dirname(__file__), "experiment_log_volatility_lstm.csv")
LOG_COLUMNS = [
    "timestamp", "tickers", "start_date", "end_date", "horizon", "time_step",
    "n_splits", "seed", "rmse", "r2", "corr", "pvalue", "n_samples", "note",
]


def append_run_to_log(row: dict) -> None:
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


def _create_windows(feature_arr: np.ndarray, target_arr: np.ndarray, time_step: int):
    X, Y = [], []
    for i in range(time_step - 1, len(feature_arr)):
        X.append(feature_arr[i - time_step + 1: i + 1, :])
        Y.append(target_arr[i])
    return np.array(X), np.array(Y)


def evaluate_pooled_lstm(tickers: list[str], start_date, end_date, horizon: int,
                          n_splits: int, time_step: int = TIME_STEP, epochs: int = 60, seed: int = 42):
    datasets = {}
    for ticker in tickers:
        df = build_volatility_dataset(ticker, start_date, end_date, horizon)
        if df is not None and len(df) > (n_splits + 1) * (time_step + 5):
            datasets[ticker] = df
    if len(datasets) < 2:
        return {"error": "not_enough_tickers"}

    tf.keras.utils.set_random_seed(seed)
    folds = {t: list(TimeSeriesSplit(n_splits=n_splits).split(datasets[t])) for t in datasets}

    all_pred, all_actual = [], []
    for fold_i in range(n_splits):
        train_parts = [datasets[t].iloc[folds[t][fold_i][0]] for t in datasets]
        test_parts = [datasets[t].iloc[folds[t][fold_i][1]] for t in datasets]
        train_concat = pd.concat(train_parts, ignore_index=True)

        feature_scaler = StandardScaler().fit(train_concat[LEAN_FEATURES].values)
        target_scaler = StandardScaler().fit(train_concat[[TARGET_VOL_COLUMN]].values)

        X_train_parts, Y_train_parts, X_test_parts, Y_test_parts = [], [], [], []
        for part in train_parts:
            feat = feature_scaler.transform(part[LEAN_FEATURES].values)
            targ = target_scaler.transform(part[[TARGET_VOL_COLUMN]].values).flatten()
            X, Y = _create_windows(feat, targ, time_step)
            if X.size:
                X_train_parts.append(X); Y_train_parts.append(Y)
        for part in test_parts:
            feat = feature_scaler.transform(part[LEAN_FEATURES].values)
            targ = target_scaler.transform(part[[TARGET_VOL_COLUMN]].values).flatten()
            X, Y = _create_windows(feat, targ, time_step)
            if X.size:
                X_test_parts.append(X); Y_test_parts.append(Y)
        if not X_train_parts or not X_test_parts:
            continue

        X_train, Y_train = np.concatenate(X_train_parts), np.concatenate(Y_train_parts)
        X_test, Y_test = np.concatenate(X_test_parts), np.concatenate(Y_test_parts)

        model = build_pooled_lstm_model(time_step, len(LEAN_FEATURES))
        early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(X_train, Y_train, validation_split=0.1, epochs=epochs, batch_size=64,
                  callbacks=[early_stop], verbose=0)

        pred = target_scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).flatten()
        actual = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
        all_pred.extend(pred.tolist()); all_actual.extend(actual.tolist())
        del model
        tf.keras.backend.clear_session()

    if not all_pred:
        return {"error": "not_enough_data_for_folds"}

    all_pred, all_actual = np.array(all_pred), np.array(all_actual)
    corr, pvalue = pearsonr(all_pred, all_actual) if np.std(all_pred) > 1e-12 else (0.0, 1.0)
    return {
        "error": None,
        "n_samples": len(all_actual),
        "rmse": math.sqrt(mean_squared_error(all_actual, all_pred)),
        "r2": r2_score(all_actual, all_pred),
        "corr": corr,
        "pvalue": pvalue,
    }
   