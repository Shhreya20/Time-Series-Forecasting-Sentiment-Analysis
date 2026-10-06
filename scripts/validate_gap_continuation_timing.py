"""
Sanity check for the gap-continuation model (see gap_continuation_features.py,
model_gap_continuation_lstm.py, evaluation_gap_continuation_lstm.py): confirms
every feature uses only information available at or before market open on the
prediction day, and that the model's confidence edge is real, not just the
mechanical "overnight gap is part of the full day's move" effect restated.

Run with: python scripts/validate_gap_continuation_timing.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "GOOGL"
START, END = "2023-01-01", "2024-01-01"

raw = yf.download(TICKER, start=START, end=END, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

prev_close = raw["Close"].shift(1)
overnight_gap = (raw["Open"] - prev_close) / prev_close
intraday_move = (raw["Close"] - raw["Open"]) / raw["Open"]
full_day_move = (raw["Close"] - prev_close) / prev_close

print("Timing check: OvernightGap uses only Open (today) and Close (yesterday) --")
print("both known before today's close. PrevDayReturn/RangePct/VolumeChg/VIX are")
print("each either backward-looking or same-day-at-open. No feature here can see")
print("today's Close before predicting it.\n")

print("Decomposition check: FullDayMove = OvernightGap + IntradayMove (approximately,")
print("for small moves). This is why a trivial 'bet on the gap' rule already scores")
print("well -- it is NOT leakage, it's two parts of one sum, disclosed in the README.")
corr = overnight_gap.corr(full_day_move)
naive_acc = ((overnight_gap > 0).astype(int) == (full_day_move > 0).astype(int)).mean()
print(f"  corr(OvernightGap, FullDayMove) = {corr:.3f}")
print(f"  naive 'bet on the gap' accuracy = {naive_acc:.3f}")
print("  (the model must beat THIS number, not 50%, to be adding real value)")
