# Stock Price Prediction — LSTM + Technical Indicators + News Sentiment

A stock price/direction prediction project built around one core principle: **every
result is validated before it's trusted.** The project doesn't claim to beat the
market — it demonstrates a properly validated pipeline, catches its own mistakes
along the way, and reports honest findings (including negative ones) instead of
overselling results.

## What this project actually shows

The original next-day direction LSTM (walk-forward evaluation, naive-baseline
comparison, a pipeline sanity check to rule out silent bugs) **and** a real
historical news-sentiment correlation test both came back negative: neither
price/technical data nor news sentiment showed statistically significant
predictive power for short-horizon GOOGL returns. That's a legitimate,
evidence-backed finding, consistent with weak-form market efficiency — and a
stronger result than most tutorial LSTM projects produce (most of which are fooled
by an inflated R², or by predicting raw price levels instead of returns — see
below).

Two follow-up pivots, tested with the same walk-forward rigor across 5 tickers
(GOOGL/AAPL/MSFT/AMZN/META), found real, statistically significant signal by
changing *what* is predicted rather than forcing the same question to work:

- **Volatility prediction** (`model_volatility_lstm.py`): predicting near-term
  realized volatility (which clusters, unlike direction) using VIX and
  days-to-earnings as features. **R²=0.127±0.015, correlation=0.367±0.014,
  p<10⁻¹⁴⁰**, stable across 3 random seeds.
- **Same-day gap-continuation** (`model_gap_continuation_lstm.py`): predicting
  whether today's close beats yesterday's close, using the overnight gap, VIX, and
  a diverse-sector peer-gap feature. **71.9% accuracy, +5.4 percentage points over
  a naive "bet on the gap" baseline**, and — the most defensible part — **91.6%
  accuracy on the model's most confident quartile of predictions** (vs. 84-86% for
  the naive rule on that same subset), confirmed not to be a restated tautology
  (see `scripts/validate_gap_continuation_timing.py`).

## Project structure

```
stock_app/
├── app.py                  # Streamlit UI — thin, wires up the modules below
├── features.py             # Technical indicators + dataset windowing (direction model)
├── model.py                # Direction-prediction LSTM architecture + training helper
├── evaluation.py           # Walk-forward training/evaluation orchestration + run log
├── sentiment.py            # Live recent-window news sentiment (VADER)
├── experiment_log.csv      # Auto-appended log of every "Predict" run (not shown in UI)
│
├── volatility_features.py          # Feature/target construction for volatility prediction
├── model_volatility.py             # RandomForest baseline for volatility prediction
├── model_volatility_lstm.py        # Validated LSTM for volatility prediction (R²=0.127)
├── evaluation_volatility.py        # Walk-forward eval (RandomForest) + run log
├── evaluation_volatility_lstm.py   # Walk-forward eval (LSTM) + run log
├── experiment_log_volatility.csv       # RandomForest run log
├── experiment_log_volatility_lstm.csv  # LSTM run log
│
├── gap_continuation_features.py       # Feature/target construction for gap-continuation
├── model_gap_continuation_lstm.py     # Validated LSTM (71.9% acc, +5.4pp over naive)
├── evaluation_gap_continuation_lstm.py # Walk-forward eval + confidence-bucket reporting
├── experiment_log_gap_continuation.csv # Run log
│
├── requirements.txt        # Deps for app.py
├── requirements-sentiment.txt  # Extra deps for scripts/ (feedparser, nltk)
├── scripts/                # Standalone one-off analyses (not part of the main app)
│   ├── pipeline_validation.py                 # Local synthetic-signal sanity check
│   ├── news_sentiment_correlation.py          # Recent-window VADER correlation check
│   ├── finbert_sentiment_from_csv.py          # Correlation check using a FinBERT-scored CSV
│   ├── validate_volatility_target.py          # Proves the volatility target has no lookahead leakage
│   └── validate_gap_continuation_timing.py    # Proves gap-continuation features have no lookahead leakage
├── notebooks/               # Colab notebooks (for anything too heavy for this machine)
│   ├── pipeline_validation_colab.ipynb  # Full-config synthetic-signal sanity check
│   ├── finbert_colab_notebook.ipynb     # FinBERT scoring (recent headlines)
│   └── gdelt_colab_fetch.ipynb          # ~90-day historical GDELT fetch + FinBERT scoring
├── data/
│   └── gdelt_correlation_merged.csv     # The real sentiment/return dataset used in the finding above
└── legacy/                  # Original pre-refactor files, kept for history
    ├── stock_price_prediction_streamlit.py     # The very first version (untouched)
    └── stock_price_prediction_streamlit_v2.py  # The monolithic version before this module split
```

Note: the volatility and gap-continuation modules are standalone, validated
pipelines (run directly as Python, not yet wired into `app.py`'s UI) — see "Running
the validated models" below.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the standalone scripts (sentiment correlation checks), install the extra deps:

```bash
pip install -r requirements-sentiment.txt
```

## Running the validated models

The volatility and gap-continuation results aren't wired into the Streamlit app —
they're standalone, validated pipelines you run directly:

```python
# Volatility prediction (R²=0.127, p<10⁻¹⁴⁰)
from evaluation_volatility_lstm import evaluate_pooled_lstm
result = evaluate_pooled_lstm(["GOOGL","AAPL","MSFT","AMZN","META"],
                               "2021-10-01", "2026-09-30", horizon=3, n_splits=5)

# Gap-continuation (71.9% acc, 91.6% on the confident quartile)
from evaluation_gap_continuation_lstm import evaluate_pooled_gap_model
result = evaluate_pooled_gap_model(["GOOGL","AAPL","MSFT","AMZN","META"],
                                    "2010-01-01", "2026-09-30", n_splits=5)
```

Leakage/timing proofs for both: `scripts/validate_volatility_target.py` and
`scripts/validate_gap_continuation_timing.py`.

## The story, step by step

### 1. Cleanup
The project started as a messy set of duplicate files, including one with a
**Twitter API bearer token committed in plaintext**. That's flagged for rotation
regardless of deletion — a leaked key stays live until revoked at the source.

### 2. The R² trap
The original model predicted raw **price levels**, scaled with one `MinMaxScaler`
fit across the entire dataset. An early run reported R²=0.97 — impressive-looking,
but fake: since tomorrow's price is almost always close to today's, "predict
tomorrow ≈ yesterday" gets a high R² by construction, regardless of real skill.

**Fix:** predict a **stationary return** instead of price level, use technical
indicators (RSI, MACD, price/moving-average ratios) as inputs, and track a
**naive baseline** ("tomorrow = today") plus **directional accuracy** as the real
scorecards — not R².

### 3. Walk-forward validation
A single train/test split can get lucky or unlucky. `evaluation.py` uses
`TimeSeriesSplit` to train on an early chronological chunk and test on the next
unseen chunk, repeated across multiple folds, with results averaged — the standard,
trustworthy way to evaluate a time series model.

### 4. The pipeline sanity check
Early walk-forward runs showed ~48-53% directional accuracy (chance level) and a
suspicious flat "+0.00%" forecast every time. Two explanations were possible: no
real signal exists in the data, or the pipeline itself is broken. To find out,
`scripts/pipeline_validation.py` / `notebooks/pipeline_validation_colab.ipynb` plant
a **synthetic feature deliberately correlated with the future return** and check
whether the model detects it.

**Result:** the original 3-stacked-LSTM+dropout architecture produced **identical**
accuracy whether or not the planted signal was present — proof it was collapsing to
predicting the mean, regardless of input (too much capacity/regularization for
~500-600 training samples). Switching to the current lighter architecture
(`model.py`: single `LSTM(32)` + small dense layer) fixed it: signal-injected runs
jumped to 83% directional accuracy vs ~53% for the no-signal control.

**Why this matters:** it turned an ambiguous result ("~50% accuracy, is that real?")
into a trustworthy one. Re-running the real data with the fixed architecture still
showed ~48-53% accuracy — but now that's a validated finding, not a symptom of a
broken model.

### 5. The sentiment investigation
Testing whether news sentiment adds anything price data alone doesn't hit several
real, worth-documenting obstacles:

- **Local hardware**: this machine has ~3.8GB usable RAM — PyTorch/FinBERT crashed
  on import locally. Fixed by running FinBERT on **Google Colab** instead (same
  model, different compute).
- **Google News RSS** only reliably returns ~2 weeks of recent headlines regardless
  of settings — a hard limitation of that free source. With only 5 matched trading
  days, VADER and FinBERT gave **opposite-signed** correlations on the same sample —
  proof it was noise, not a real relationship.
- **GDELT** (`notebooks/gdelt_colab_fetch.ipynb`) is a genuinely free, historical
  news database, but getting it working required: retry/backoff for aggressive
  rate-limiting, weekly-chunked queries (GDELT caps results per request), fixing an
  invalid query filter (`sourcelang:english` isn't valid GDELT syntax), and
  restricting to known tech/business news domains (a plain `"Google"` full-text
  search mostly matches incidental mentions like "hosted on Google Cloud," not
  actual Google news).

**Final result** (`data/gdelt_correlation_merged.csv`): 555 genuinely relevant
headlines across 74 days, 51 matched trading days, **Pearson r = -0.13, p = 0.36**
— not statistically significant. A real negative finding this time, not a
small-sample fluke.

**Why we didn't build a merged sentiment-LSTM**: the plan going in was to only
build one if the simple correlation check showed a real relationship first. It
didn't — so building a bigger model on top of a non-existent signal would risk
fitting noise and overselling the result.

### 6. Combined Outlook
The app's final section (`app.py`, bottom of the "Predict" flow) shows the LSTM's
price-forecast direction alongside a **live** 14-day news-sentiment score
(`sentiment.py`, lightweight VADER — no heavy dependencies), combined into a
Bullish/Bearish/Mixed/Neutral read. The validated caveat about both signals'
non-significance is shown directly above it — this is presented as "what two
signals currently indicate," not a validated trading signal.

### 7. The volatility pivot
After the direction-prediction result held up (negative, but validated) across
architecture changes, feature changes, and multi-ticker pooling, the next question
was whether a *different target* is more learnable. Realized volatility is a
better-studied candidate — it clusters/persists, unlike direction.

A RandomForest using VIX and days-to-next-earnings alongside the original
technical indicators found a real signal (r=0.34, p<10⁻²⁸ on GOOGL, replicated
r=0.23-0.49 across all 5 tickers). The LSTM version initially *underperformed*
(r=0.06, not significant) with a 20-day lookback and all 10 features — shortening
the window to 5 days, cutting to 4 relevant features, and pooling all 5 tickers for
training (LSTMs need more data than RandomForest to generalize) brought it to
**R²=0.127, correlation=0.367, p<10⁻¹⁴⁰**, stable across 3 random seeds
(`model_volatility_lstm.py`).

### 8. The gap-continuation model
A separate, real-world-grounded idea: instead of blind next-day forecasting,
predict whether *today's* close beats *yesterday's* close using *today's* opening
gap — information actually available at market open. A trivial rule ("bet on the
gap direction") already gets ~63-67% accuracy on its own; this is disclosed clearly
in `gap_continuation_features.py` and verified in
`scripts/validate_gap_continuation_timing.py` — it's a mechanical consequence of
the overnight gap being part of the full day's move, **not leakage**, but it means
a model must be judged against this baseline, not 50%.

Adding VIX and a diverse-sector peer-gap feature (distinguishing "this stock has
its own news" from "the whole market gapped") pushed the LSTM to **71.9% accuracy,
+5.4 percentage points over the naive rule**. A full hyperparameter sweep (units,
layers, dropout, lookback window, ticker identity, day-of-week) found no
configuration meaningfully beat this simple setup — the gain came from features,
not architecture. The most defensible result: restricting to the model's most
confident quartile of predictions gives **91.6% accuracy vs. 84-86% for the naive
rule on that same subset** — a real, calibration-based edge, confirmed (not just
"confidence = bigger gap size restated") in `evaluation_gap_continuation_lstm.py`.

## Key methodology notes

- **Stationary features, not price levels** — avoids the model extrapolating
  outside the price range it trained on.
- **Scalers fit on train data only** — avoids leaking test-period statistics into
  training.
- **Walk-forward validation, not a single split** — avoids drawing conclusions from
  one lucky/unlucky test period.
- **A naive baseline and directional accuracy, always reported alongside RMSE/R²**
  — R² alone is easy to game by accident (see the R² trap above).
- **A pipeline sanity check with a planted signal** — the only way to know whether
  a "no signal found" result is a real finding or a silent bug.

## Known limitations / future work

- The original direction-prediction model is only tested on GOOGL. The volatility
  and gap-continuation models are validated across 5 tickers
  (GOOGL/AAPL/MSFT/AMZN/META) — the "no direction signal" finding itself was not
  retested at that scale with the original architecture.
- Sentiment data is capped at ~90 days of real history (GDELT's practical limit for
  a free, no-signup source) — nowhere near the 5+ year training windows used
  elsewhere, which is why sentiment isn't merged into any of the three models.
  Re-tested directly against the volatility target with the available 51 matched
  days: still not significant (r=0.16, p=0.42), but that test itself is
  underpowered — a real answer needs a much longer sentiment history, not more
  modeling.
- Two other well-documented real market anomalies — post-earnings-announcement
  drift and time-series momentum — were tested (on both mega-cap tech and a
  small/mid-cap basket) and did not hold up (not significant, or significant
  in-sample but with negative out-of-sample R²). Documented as ruled out, not
  silently dropped.
- The volatility model's R² is positive for 3 of 5 tickers (GOOGL/MSFT/AMZN) but
  slightly negative for AAPL/META despite a significant correlation — the model
  tracks relative volatility changes but isn't perfectly calibrated in absolute
  magnitude for every ticker.
- The gap-continuation model's `DaysToEarnings`-style features aren't used (by
  design, to keep it independent of the volatility model's earnings-date
  dependency), but its `BroadPeerGap` feature assumes same-timezone, same-session
  reference tickers — not tested outside US large-caps.
- This machine's hardware (~3.8GB RAM) means heavier models (transformer-based
  sentiment, larger LSTMs) need to run on Colab rather than locally — see
  `notebooks/`.
