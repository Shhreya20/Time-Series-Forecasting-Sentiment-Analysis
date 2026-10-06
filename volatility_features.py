"""Feature/target construction for the volatility-prediction pivot.

Direction prediction (see features.py/evaluation.py) was tested extensively --
12 experiments across architectures, feature sets, and data pooling -- and found
no statistically significant signal (see experiment_log.csv and README). Realized
volatility is a different, better-established target: it's known to cluster/persist,
and two features available in advance (market-wide VIX regime, proximity to a
scheduled earnings date) are plausible, economically grounded predictors of it.

Target_Vol[i] = std of daily returns over days (i+1, i+2, i+3) for horizon=3 --
strictly AFTER day i, non-overlapping with TrailingVol[i] (days i-2, i-1, i).
Verified via scripts/validate_volatility_target.py.
"""
import numpy as np
import pandas as pd
import yfinance as yf

from features import FEATURE_COLUMNS, add_technical_indicators

VOLATILITY_FEATURE_COLUMNS = FEATURE_COLUMNS + ["TrailingVol", "VIX", "DaysToEarnings"]
TARGET_VOL_COLUMN = "Target_Vol"


def _days_to_next_earnings(dates: pd.DatetimeIndex, earnings_dates: pd.DatetimeIndex) -> list:
    return [
        (future.min() - d).days if len(future := earnings_dates[earnings_dates >= d]) else np.nan
        for d in dates
    ]


def build_volatility_dataset(ticker: str, start_date, end_date, horizon: int) -> pd.DataFrame | None:
    """Downloads price/VIX/earnings data and returns a feature+target frame for
    volatility prediction, reusing add_technical_indicators' own dropna() so every
    extra column stays aligned to it (the single dropna() call is what keeps this
    safe -- see the leakage bug this approach was written to avoid)."""
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    earnings = yf.Ticker(ticker).earnings_dates
    earnings_dates = (
        pd.to_datetime(earnings.index.tz_localize(None)) if earnings is not None and len(earnings) else pd.DatetimeIndex([])
    )

    src = raw[["Close", "Volume"]].copy()
    src["Date"] = raw.index
    src["VIX"] = vix["Close"].reindex(raw.index)

    daily_return = raw["Close"].pct_change()
    src["TrailingVol"] = daily_return.rolling(horizon).std().reindex(raw.index)
    src[TARGET_VOL_COLUMN] = daily_return.rolling(horizon).std().shift(-horizon).reindex(raw.index)
    src["DaysToEarnings"] = _days_to_next_earnings(raw.index, earnings_dates)

    return add_technical_indicators(src, horizon)
