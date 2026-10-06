"""Model for the volatility-prediction pivot.

RandomForest was chosen over the project's LSTM/feedforward alternatives because
it was the only one that generalized: on the same walk-forward setup and features,
LSTM(32-unit, 20-day window) gave a non-significant correlation (r=0.06, p=0.06),
a plain feedforward net got r=0.20 (p=1.4e-10), and this RandomForest got r=0.34
(p=1.2e-28) on GOOGL, replicated (r=0.23-0.49, all p<1e-12) across AAPL/MSFT/AMZN/META.
The likely reason: the two strongest features here (VIX level, days-to-earnings) are
informative as a same-day snapshot, not as a sequence -- so a recurrent window adds
noise rather than signal, and a tree-based model's ability to split on thresholds
(e.g. "earnings within 3 days") fits this kind of feature better than a dense/LSTM net.
"""
from sklearn.ensemble import RandomForestRegressor


def build_volatility_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
