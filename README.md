# Stock Price Prediction — LSTM + Technical Indicators + News Sentiment

A stock price/direction prediction project built around one core principle: **every
result is validated before it's trusted.** The project doesn't claim to beat the
market — it demonstrates a properly validated pipeline, catches its own mistakes
along the way, and reports honest findings (including negative ones) instead of
overselling results.

## What this project actually shows

Across a properly validated LSTM (walk-forward evaluation, naive-baseline
comparison, a pipeline sanity check to rule out silent bugs) **and** a real
historical news-sentiment correlation test, neither price/technical data nor news
sentiment showed statistically significant predictive power for short-horizon GOOGL
returns. That's not a failure — it's a legitimate, evidence-backed finding,
consistent with weak-form market efficiency, and it's a stronger result than most
tutorial LSTM projects produce (most of which are fooled by an inflated R² — see
below).

## Project structure

```
stock_app/
├── app.py                  # Streamlit UI — thin, wires up the modules below
├── features.py             # Technical indicators + dataset windowing
├── model.py                # LSTM architecture + training helper
├── evaluation.py           # Walk-forward training/evaluation orchestration + run log
├── sentiment.py            # Live recent-window news sentiment (VADER)
├── experiment_log.csv      # Auto-appended log of every "Predict" run (not shown in UI)
├── requirements.txt        # Deps for app.py
├── requirements-sentiment.txt  # Extra deps for scripts/ (feedparser, nltk)
├── scripts/                # Standalone one-off analyses (not part of the main app)
│   ├── pipeline_validation.py         # Local synthetic-signal sanity check
│   ├── news_sentiment_correlation.py  # Recent-window VADER correlation check
│   └── finbert_sentiment_from_csv.py  # Correlation check using a FinBERT-scored CSV
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

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the standalone scripts (sentiment correlation checks), install the extra deps:

```bash
pip install -r requirements-sentiment.txt
```

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

- Only tested on GOOGL — a multi-ticker generalization test would strengthen the
  "no signal" finding into a more general claim.
- Sentiment data is capped at ~90 days of real history (GDELT's practical limit for
  a free, no-signup source) — nowhere near the LSTM's 5-year training window, which
  is why sentiment isn't merged into the main model as an input feature.
- This machine's hardware (~3.8GB RAM) means heavier models (transformer-based
  sentiment, larger LSTMs) need to run on Colab rather than locally — see
  `notebooks/`.
