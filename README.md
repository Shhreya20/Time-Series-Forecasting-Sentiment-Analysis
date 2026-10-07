# Stock Price Prediction — LSTM-based Volatility & Price-Continuation Forecasting

A machine learning project that uses LSTM networks to forecast short-term stock
market behavior from historical price and volume data. Every result here is
validated with walk-forward testing, naive-baseline comparison, and explicit
leakage checks before being reported.

## Results

**Volatility forecasting** (`model_volatility_lstm.py`) — predicts near-term
realized volatility for a stock using price, volume, market-volatility (VIX), and
earnings-proximity features.

- **R² = 0.127, correlation = 0.367, p < 10⁻¹⁴⁰**
- Validated across 5 tickers (GOOGL, AAPL, MSFT, AMZN, META) and stable across
  multiple random seeds.

**Same-day price-continuation forecasting** (`model_gap_continuation_lstm.py`) —
predicts whether a stock's closing price will finish higher than the previous
day's close, using the overnight price gap, market volatility, and a
diversified cross-sector reference signal.

- **71.9% accuracy**, a meaningful improvement over the baseline rate
- **91.6% accuracy on the model's most confident quartile of predictions** —
  the model's confidence scores are well-calibrated, so predictions can be
  filtered by confidence for higher-precision use cases.

## Project structure

```
stock_app/
├── app.py                          # Streamlit UI
├── features.py                     # Technical indicator feature engineering
├── model.py                        # LSTM architecture + training helper
├── evaluation.py                   # Walk-forward evaluation orchestration
├── sentiment.py                    # Live news-sentiment scoring (VADER)
│
├── volatility_features.py          # Feature/target construction for volatility forecasting
├── model_volatility.py             # RandomForest baseline
├── model_volatility_lstm.py        # Validated LSTM (R²=0.127)
├── evaluation_volatility.py        # Walk-forward evaluation (RandomForest)
├── evaluation_volatility_lstm.py   # Walk-forward evaluation (LSTM)
│
├── gap_continuation_features.py       # Feature/target construction for price-continuation
├── model_gap_continuation_lstm.py     # Validated LSTM (71.9% acc, 91.6% on confident quartile)
├── evaluation_gap_continuation_lstm.py # Walk-forward evaluation + confidence analysis
│
├── logs/                           # Logged results from every evaluation run
├── requirements.txt                # Dependencies for app.py
├── requirements-sentiment.txt      # Extra dependencies for sentiment scripts
├── scripts/
│   ├── pipeline_validation.py                 # Synthetic-signal pipeline sanity check
│   ├── news_sentiment_correlation.py          # News sentiment correlation analysis
│   ├── finbert_sentiment_from_csv.py          # FinBERT-based sentiment scoring
│   ├── validate_volatility_target.py          # Confirms no lookahead leakage in the volatility target
│   └── validate_gap_continuation_timing.py    # Confirms no lookahead leakage in the gap-continuation features
├── notebooks/
│   ├── pipeline_validation_colab.ipynb
│   ├── finbert_colab_notebook.ipynb
│   └── gdelt_colab_fetch.ipynb
└── data/
    └── gdelt_correlation_merged.csv    # Historical news/return dataset
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running the validated models

```python
# Volatility forecasting
from evaluation_volatility_lstm import evaluate_pooled_lstm
result = evaluate_pooled_lstm(["GOOGL","AAPL","MSFT","AMZN","META"],
                               "2021-10-01", "2026-09-30", horizon=3, n_splits=5)

# Same-day price-continuation forecasting
from evaluation_gap_continuation_lstm import evaluate_pooled_gap_model
result = evaluate_pooled_gap_model(["GOOGL","AAPL","MSFT","AMZN","META"],
                                    "2010-01-01", "2026-09-30", n_splits=5)
```

## Methodology

- **Walk-forward validation** (`TimeSeriesSplit`) — trains on an earlier
  chronological chunk and tests on the next unseen chunk, repeated across
  multiple folds, instead of relying on a single train/test split.
- **Naive-baseline comparison, always reported alongside accuracy/R²** —
  every result is measured against a simple baseline (e.g. "assume no change"),
  not just an absolute score.
- **Leakage/timing validation scripts** — `scripts/validate_volatility_target.py`
  and `scripts/validate_gap_continuation_timing.py` prove each model's features
  only use information available at prediction time.
- **Multi-ticker validation** — both the volatility and gap-continuation models
  are tested across 5 tickers and multiple random seeds, not a single run on a
  single stock.
- **Scalers fit on training data only**, to avoid leaking test-period statistics
  into training.

## Technologies used

Python, TensorFlow/Keras (LSTM), scikit-learn, pandas, NumPy, yfinance, Streamlit,
SciPy, NLTK (VADER), HuggingFace Transformers (FinBERT), Google Colab.
