"""The validated LSTM for same-day gap-continuation prediction.

Tuning history (see experiment_log_gap_continuation.csv): a hyperparameter
sweep (units 16/32, dropout, 1 vs 2 layers, lookback 3/5/10 days) found no
configuration meaningfully beats this simple one -- the real gains came from
feature engineering, not architecture:
  - Base features (gap, prior return, range, volume change) alone: only
    +0.8pp over the naive "bet on the gap direction" rule.
  - Adding VIX + a diverse-sector peer gap (BroadPeerGap): jumped to +5.0pp.
  - A 10-day lookback was clearly worse (+4.3pp) than 5 or 3 days -- this
    signal is short-memory, consistent with VIX/peer-gap being same-day,
    not sequential, information.

This model's overall accuracy (~71-72%) is real but modest. Its most useful,
validated property is that it's well-calibrated: restricting to its most
confident quartile of predictions gives ~92% accuracy (vs. ~84% for the naive
rule on that same subset) -- a genuine, statistically significant edge on top
of the naive rule, not just a repackaged version of it (see
scripts/validate_gap_continuation_confidence.py).
"""
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

TIME_STEP = 5
UNITS = 16


def build_gap_continuation_model(time_step: int = TIME_STEP, num_features: int = 6) -> Sequential:
    model = Sequential([
        LSTM(UNITS, input_shape=(time_step, num_features)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model
