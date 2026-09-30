"""LSTM architecture and training helper.

The architecture here was chosen deliberately, not by default: a pipeline sanity
check (planted-signal test, see scripts/pipeline_validation.py and
notebooks/pipeline_validation_colab.ipynb) showed an earlier 3-stacked-LSTM+dropout
version was collapsing to predicting close to the mean regardless of input — too
heavy for ~500-600 training samples and tiny return-scale targets. This lighter
architecture correctly distinguished a planted signal from noise (83% directional
accuracy vs 53% for a no-signal control, in that same test), so it's what should
actually be trusted.
"""
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


def build_model(time_step: int, num_features: int) -> Sequential:
    model = Sequential([
        LSTM(32, input_shape=(time_step, num_features)),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def fit_with_early_stopping(model: Sequential, X: np.ndarray, Y: np.ndarray, epochs: int):
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    return model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=64,
                      callbacks=[early_stop], verbose=0)
