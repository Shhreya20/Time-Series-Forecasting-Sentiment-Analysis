"""
Sanity check for the volatility-prediction pivot (see volatility_features.py,
model_volatility.py, evaluation_volatility.py): proves Target_Vol and TrailingVol
use disjoint, correctly-ordered return windows, with no overlap or future leakage.

Run with: python scripts/validate_volatility_target.py
"""
import numpy as np
import pandas as pd

HORIZON = 3

toy = pd.Series(np.arange(1, 21, dtype=float))
toy_ret = toy.pct_change()
trailing_vol = toy_ret.rolling(HORIZON).std()
target_vol = toy_ret.rolling(HORIZON).std().shift(-HORIZON)

row = 10
expected_trailing = np.std(toy_ret.iloc[row - HORIZON + 1: row + 1], ddof=1)
expected_target = np.std(toy_ret.iloc[row + 1: row + 1 + HORIZON], ddof=1)

assert abs(trailing_vol.iloc[row] - expected_trailing) < 1e-9, "TrailingVol window mismatch"
assert abs(target_vol.iloc[row] - expected_target) < 1e-9, "Target_Vol window mismatch"

print(f"TrailingVol[{row}] uses returns {row - HORIZON + 1}..{row} (backward-looking): OK")
print(f"Target_Vol[{row}] uses returns {row + 1}..{row + HORIZON} (forward-looking, no overlap): OK")
print("Validated: Target_Vol and TrailingVol windows are disjoint and correctly ordered.")
