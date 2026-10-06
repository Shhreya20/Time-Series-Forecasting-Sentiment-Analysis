"""Feature/target construction for the overnight-gap-continuation model.

After 12 next-day-direction experiments and a volatility pivot (see
evaluation.py / evaluation_volatility.py / README), this tests a different,
well-documented real-world setup: does TODAY's close beat YESTERDAY's close,
using TODAY's overnight gap (known at market open, before the close) and a
small set of same-day/contemporaneous features. This is NOT blind next-day
forecasting -- it uses information available at the open to predict that same
day's close, which is what real gap-trading strategies actually do.

Important, disclosed finding: a trivial rule ("predict close-up if the stock
gapped up") already gets ~66-67% accuracy on its own, because the overnight
gap is a large, correlated component of the full day's move (not leakage --
just two parts of one sum). The model's job is to add value ON TOP of that
rule, not to replace a mystery with magic.
"""
import numpy as np
import pandas as pd
import yfinance as yf

FEATURE_COLUMNS = ["OvernightGap", "PrevDayReturn", "RangePct", "VolumeChg", "VIX", "BroadPeerGap"]
TARGET_COLUMN = "Target"

# Diverse-sector reference basket for BroadPeerGap: distinguishes "the whole
# market gapped" from "this stock has its own news." A broader, less-correlated
# basket (finance/retail/healthcare/energy/market) outperformed averaging the
# other 4 tech tickers against each other (see experiment log).
DIVERSE_REFERENCE_TICKERS = ["JPM", "WMT", "JNJ", "XOM", "SPY"]


def _overnight_gap_series(ticker: str, start_date, end_date) -> pd.Series | None:
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prev_close = raw["Close"].shift(1)
    gap = (raw["Open"] - prev_close) / prev_close
    gap.index = raw.index
    return gap


def _broad_peer_gap(start_date, end_date) -> pd.Series:
    gaps = {t: _overnight_gap_series(t, start_date, end_date) for t in DIVERSE_REFERENCE_TICKERS}
    gaps = {t: g for t, g in gaps.items() if g is not None}
    return pd.DataFrame(gaps).mean(axis=1)


def build_gap_dataset(ticker: str, start_date, end_date, broad_peer_gap: pd.Series | None = None) -> pd.DataFrame | None:
    """Builds the feature+target frame for one ticker. Pass a precomputed
    `broad_peer_gap` (from `_broad_peer_gap`) when building several tickers in
    one run, so the reference basket is only downloaded once."""
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    vix_raw = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw["Close"]

    if broad_peer_gap is None:
        broad_peer_gap = _broad_peer_gap(start_date, end_date)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["PrevClose"] = df["Close"].shift(1)
    df["PrevClose2"] = df["Close"].shift(2)
    df["OvernightGap"] = (df["Open"] - df["PrevClose"]) / df["PrevClose"]
    df["PrevDayReturn"] = (df["PrevClose"] - df["PrevClose2"]) / df["PrevClose2"]
    df["RangePct"] = (df["High"] - df["Low"]) / df["Open"]
    df["VolumeChg"] = df["Volume"].pct_change()
    df["VIX"] = vix.reindex(raw.index).values
    df["BroadPeerGap"] = broad_peer_gap.reindex(raw.index).values
    df[TARGET_COLUMN] = (df["Close"] > df["PrevClose"]).astype(int)

    return df.dropna().reset_index(drop=True)
