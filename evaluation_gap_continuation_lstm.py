"""Walk-forward evaluation for the pooled multi-ticker gap-continuation LSTM
(see model_gap_continuation_lstm.py for why this config was chosen). Reports
both overall accuracy and confidence-bucketed accuracy, since the confidence
calibration is this model's most defensible, validated result.
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from gap_continuation_features import FEATURE_COLUMNS, TARGET_COLUMN, _broad_peer_gap, build_gap_dataset
from model_gap_continuation_lstm import TIME_STEP, build_gap_continuation_model

LOG_PATH = os.path.join(os.path.dirname(__file__), "experiment_log_gap_continuation.csv")
LOG_COLUMNS = [
    "timestamp", "tickers", "start_date", "end_date", "time_step", "n_splits", "seed",
    "accuracy", "naive_accuracy", "improvement_pp", "n_samples",
    "top25pct_confidence_acc", "top25pct_confidence_naive_acc", "note",
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


def evaluate_pooled_gap_model(tickers: list[str], start_date, end_date, n_splits: int,
                               time_step: int = TIME_STEP, epochs: int = 40, seed: int = 42):
    broad_peer_gap = _broad_peer_gap(start_date, end_date)
    datasets = {}
    for ticker in tickers:
        df = build_gap_dataset(ticker, start_date, end_date, broad_peer_gap)
        if df is not None and len(df) > (n_splits + 1) * (time_step + 5):
            datasets[ticker] = df
    if len(datasets) < 2:
        return {"error": "not_enough_tickers"}

    tf.keras.utils.set_random_seed(seed)
    folds = {t: list(TimeSeriesSplit(n_splits=n_splits).split(datasets[t])) for t in datasets}

    all_proba, all_actual, all_naive = [], [], []
    for fold_i in range(n_splits):
        train_parts = [datasets[t].iloc[folds[t][fold_i][0]] for t in datasets]
        test_parts = [datasets[t].iloc[folds[t][fold_i][1]] for t in datasets]
        train_concat = pd.concat(train_parts, ignore_index=True)
        scaler = StandardScaler().fit(train_concat[FEATURE_COLUMNS].values)

        X_train_parts, Y_train_parts, X_test_parts, Y_test_parts, naive_parts = [], [], [], [], []
        for part in train_parts:
            feat = scaler.transform(part[FEATURE_COLUMNS].values)
            X, Y = _create_windows(feat, part[TARGET_COLUMN].values, time_step)
            if X.size:
                X_train_parts.append(X); Y_train_parts.append(Y)
        for part in test_parts:
            feat = scaler.transform(part[FEATURE_COLUMNS].values)
            X, Y = _create_windows(feat, part[TARGET_COLUMN].values, time_step)
            if X.size:
                X_test_parts.append(X); Y_test_parts.append(Y)
                naive_parts.append((part["OvernightGap"].values[time_step - 1:] > 0).astype(int))
        if not X_train_parts or not X_test_parts:
            continue

        X_train, Y_train = np.concatenate(X_train_parts), np.concatenate(Y_train_parts)
        X_test, Y_test = np.concatenate(X_test_parts), np.concatenate(Y_test_parts)
        naive = np.concatenate(naive_parts)

        model = build_gap_continuation_model(time_step, len(FEATURE_COLUMNS))
        early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(X_train, Y_train, validation_split=0.1, epochs=epochs, batch_size=64,
                  callbacks=[early_stop], verbose=0)

        proba = model.predict(X_test, verbose=0).flatten()
        all_proba.extend(proba.tolist()); all_actual.extend(Y_test.tolist()); all_naive.extend(naive.tolist())
        del model
        tf.keras.backend.clear_session()

    if not all_proba:
        return {"error": "not_enough_data_for_folds"}

    all_proba, all_actual, all_naive = np.array(all_proba), np.array(all_actual), np.array(all_naive)
    pred = (all_proba > 0.5).astype(int)
    accuracy = float((pred == all_actual).mean())
    naive_accuracy = float((all_naive == all_actual).mean())

    confidence = np.abs(all_proba - 0.5)
    top25_cut = np.percentile(confidence, 75)
    top25_mask = confidence >= top25_cut

    return {
        "error": None,
        "n_samples": len(all_actual),
        "accuracy": accuracy,
        "naive_accuracy": naive_accuracy,
        "improvement_pp": 100 * (accuracy - naive_accuracy),
        "top25pct_confidence_acc": float((pred[top25_mask] == all_actual[top25_mask]).mean()),
        "top25pct_confidence_naive_acc": float((all_naive[top25_mask] == all_actual[top25_mask]).mean()),
    }
