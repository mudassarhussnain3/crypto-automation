"""
Train LSTM next-candle direction classifiers, one per coin per timeframe.

Sequence length 60: the last 60 candles predict the direction of the next candle.
Train on data/train/, validate on data/validate/ (walk-forward holdout; windows are
built within each split only, so none span the 2023->2024 boundary). Features are
scaled with a StandardScaler fit on TRAIN ONLY (saved alongside the model for
inference). Saves 15 models to models/lstm/ + matching _scaler.pkl, and prints
accuracy/precision/recall/F1 per coin+timeframe.

Usage:
    python train_lstm.py            # train all 15 on real Phase 1 data
    python train_lstm.py --selftest # synthetic smoke test, no data needed
"""

import os
import sys

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

import ml_common as mc

MODEL_DIR = os.path.join("models", "lstm")
SEQ_LEN = 60
EPOCHS = 30


def make_sequences(X, y, seq_len=SEQ_LEN):
    """
    Sliding windows: window i = rows [i, i+seq_len), label = y at the window's last row
    (the next-candle direction following that last candle). Stays within one split.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.asarray(Xs, dtype="float32"), np.asarray(ys, dtype="float32")


def build_model(n_features):
    import keras
    from keras import layers
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_one(coin, tf, X_tr, y_tr, X_va, y_va, epochs=EPOCHS, save=True):
    """Scale (train-only fit), window, train with early stopping, report, save."""
    import keras

    # Scaler fit on TRAIN features only -> no leakage from validation.
    scaler = StandardScaler().fit(X_tr.values)
    Xtr_s = scaler.transform(X_tr.values)
    Xva_s = scaler.transform(X_va.values)

    Xtr_seq, ytr_seq = make_sequences(Xtr_s, y_tr.values)
    Xva_seq, yva_seq = make_sequences(Xva_s, y_va.values)

    model = build_model(X_tr.shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                       restore_best_weights=True)
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=epochs, batch_size=64, callbacks=[es], verbose=0)

    proba = model.predict(Xva_seq, verbose=0).ravel()  # sigmoid = P(UP) = confidence
    y_pred = (proba >= 0.5).astype(int)
    mc.print_report("lstm", coin, tf, yva_seq.astype(int), y_pred)
    assert ((proba >= 0) & (proba <= 1)).all(), "confidence out of [0,1]"

    if save:
        mc.ensure_dir(MODEL_DIR)
        model.save(os.path.join(MODEL_DIR, f"{coin}_{tf}.h5"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"{coin}_{tf}_scaler.pkl"))
    return model, scaler


def run():
    print(f"Training LSTM models (seq_len={SEQ_LEN}, per coin per timeframe)...")
    for coin in mc.COINS:
        for tf in mc.TIMEFRAMES:
            X_tr, y_tr = mc.make_xy(mc.load_split(coin, tf, "train"))
            X_va, y_va = mc.make_xy(mc.load_split(coin, tf, "validate"))
            train_one(coin, tf, X_tr, y_tr, X_va, y_va)
    print(f"Done. Models + scalers saved under {MODEL_DIR}/.")


def selftest():
    """Train+predict on synthetic data with a tiny epoch count. No saving."""
    # Need > SEQ_LEN rows in each split, so generate plenty.
    df = mc.synth_df(n=400)
    X, y = mc.make_xy(df)
    split = int(len(X) * 0.7)  # forward-chained split
    model, scaler = train_one("SYNTH", "1h", X.iloc[:split], y.iloc[:split],
                              X.iloc[split:], y.iloc[split:], epochs=2, save=False)
    # Confirm inference path: scale -> window -> predict valid probabilities.
    Xva_seq, _ = make_sequences(scaler.transform(X.iloc[split:].values), y.iloc[split:].values)
    proba = model.predict(Xva_seq, verbose=0).ravel()
    assert ((proba >= 0) & (proba <= 1)).all() and len(proba) > 0
    print("selftest OK: LSTM trains, scales train-only, and emits valid probabilities.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
