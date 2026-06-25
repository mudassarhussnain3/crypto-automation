"""
Train XGBoost next-candle direction classifiers, one per coin per timeframe.

Train on data/train/, validate on data/validate/ (walk-forward holdout: train
2021-2023 strictly precedes validation 2024, no shuffling). Saves 15 models to
models/xgboost/ and prints accuracy/precision/recall/F1 per coin+timeframe.

Usage:
    python train_xgboost.py            # train all 15 on real Phase 1 data
    python train_xgboost.py --selftest # synthetic smoke test, no data needed
"""

import os
import sys

import joblib
from xgboost import XGBClassifier

import ml_common as mc

MODEL_DIR = os.path.join("models", "xgboost")


def build_model():
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )


def train_one(coin, tf, X_tr, y_tr, X_va, y_va, save=True):
    """Fit one model, report on validation, optionally save. Returns the model."""
    model = build_model()
    model.fit(X_tr, y_tr)  # no shuffle: temporal order preserved by the split
    y_pred = model.predict(X_va)
    conf = model.predict_proba(X_va).max(axis=1)  # confidence of the predicted class
    mc.print_report("xgboost", coin, tf, y_va, y_pred)
    assert ((conf >= 0) & (conf <= 1)).all(), "confidence out of [0,1]"
    if save:
        mc.ensure_dir(MODEL_DIR)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{coin}_{tf}.pkl"))
    return model


def run():
    print("Training XGBoost models (per coin per timeframe)...")
    for coin in mc.COINS:
        for tf in mc.TIMEFRAMES:
            X_tr, y_tr = mc.make_xy(mc.load_split(coin, tf, "train"))
            X_va, y_va = mc.make_xy(mc.load_split(coin, tf, "validate"))
            train_one(coin, tf, X_tr, y_tr, X_va, y_va)
    print(f"Done. Models saved under {MODEL_DIR}/ (no scaler needed — trees are scale-invariant).")


def selftest():
    """Train+predict on synthetic data; assert the full path works. No saving."""
    df = mc.synth_df(n=500)
    X, y = mc.make_xy(df)
    split = int(len(X) * 0.7)  # forward-chained: first 70% train, last 30% validate
    model = train_one("SYNTH", "1h", X.iloc[:split], y.iloc[:split],
                      X.iloc[split:], y.iloc[split:], save=False)
    proba = model.predict_proba(X.iloc[split:])
    assert proba.shape[1] == 2 and ((proba >= 0) & (proba <= 1)).all()
    print("selftest OK: XGBoost trains and emits valid probabilities.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
