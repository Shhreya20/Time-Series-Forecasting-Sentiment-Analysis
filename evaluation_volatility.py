"""Walk-forward evaluation for the volatility-prediction pivot, with the same
rigor as evaluation.py: walk-forward splits, an explicit naive baseline, and a
logged experiment history -- plus a mean-prediction baseline and a significance
test, since "beats RMSE" alone was shown (in the direction-prediction work) to be
misleading on its own.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from model_volatility import build_volatility_model
from volatility_features import TARGET_VOL_COLUMN, VOLATILITY_FEATURE_COLUMNS, build_volatility_dataset

LOG_PATH = os.path.join(os.path.dirname(__file__), "experiment_log_volatility.csv")
LOG_COLUMNS = [
    "timestamp", "ticker", "start_date", "end_date", "horizon", "n_splits", "n_folds_used",
    "rmse_mean_pred", "rmse_persistence", "rmse_model",
    "r2_mean_pred", "r2_persistence", "r2_model",
    "corr_model", "pvalue_model", "note",
]


def append_run_to_log(row: dict) -> None:
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[VOLATILITY_FEATURE_COLUMNS])
    X_test = scaler.transform(test_df[VOLATILITY_FEATURE_COLUMNS])
    y_train = train_df[TARGET_VOL_COLUMN].values
    y_test = test_df[TARGET_VOL_COLUMN].values

    model = build_volatility_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    persistence_pred = test_df["TrailingVol"].values       # naive: tomorrow's vol = today's trailing vol
    mean_pred = np.full(len(y_test), y_train.mean())        # naive: always predict the training-period average

    return {
        "y_test": y_test, "pred": pred,
        "persistence_pred": persistence_pred, "mean_pred": mean_pred,
    }


def evaluate_volatility_model(ticker: str, start_date, end_date, horizon: int, n_splits: int):
    df = build_volatility_dataset(ticker, start_date, end_date, horizon)
    if df is None or len(df) < (n_splits + 1) * 10:
        return {"error": "not_enough_data"}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_y, all_pred, all_persist, all_mean = [], [], [], []
    n_folds_used = 0
    for train_idx, test_idx in tscv.split(np.arange(len(df))):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        if len(train_df) < 20 or len(test_df) < 5:
            continue
        fold = run_fold(train_df, test_df)
        all_y.extend(fold["y_test"].tolist())
        all_pred.extend(fold["pred"].tolist())
        all_persist.extend(fold["persistence_pred"].tolist())
        all_mean.extend(fold["mean_pred"].tolist())
        n_folds_used += 1

    if n_folds_used == 0:
        return {"error": "not_enough_data_for_folds"}

    all_y, all_pred, all_persist, all_mean = map(np.array, [all_y, all_pred, all_persist, all_mean])
    corr, pvalue = pearsonr(all_pred, all_y) if np.std(all_pred) > 1e-12 else (0.0, 1.0)

    return {
        "error": None,
        "n_folds_used": n_folds_used,
        "n_samples": len(all_y),
        "rmse_mean_pred": math.sqrt(mean_squared_error(all_y, all_mean)),
        "rmse_persistence": math.sqrt(mean_squared_error(all_y, all_persist)),
        "rmse_model": math.sqrt(mean_squared_error(all_y, all_pred)),
        "r2_mean_pred": r2_score(all_y, all_mean),
        "r2_persistence": r2_score(all_y, all_persist),
        "r2_model": r2_score(all_y, all_pred),
        "corr_model": corr,
        "pvalue_model": pvalue,
    }
