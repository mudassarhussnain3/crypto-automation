"""
Shared helpers for the Phase 2 trainers (train_xgboost / train_random_forest / train_lstm).

Keeps data loading, the target definition, feature list, metric printing, and the
synthetic --selftest frame in one place so the three trainers stay thin and consistent.
Never reads data/test/ — that split is locked until final evaluation.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

COINS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
TIMEFRAMES = ["15m", "1h", "4h"]

# 28 features: OHLCV + 10 Phase 1 indicators + 13 Phase 2b engineered features
# (must match pipeline.add_indicators output columns).
FEATURES = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "macd_diff",
    "bb_high", "bb_mid", "bb_low", "atr", "vol_ma20", "vol_ratio",
    # Phase 2b
    "ema_9", "ema_21", "ema_trend",
    "rsi_change_3", "macd_diff_change_3",
    "higher_high", "lower_low", "close_position",
    "atr_percentile", "bb_width",
    "volume_zscore", "price_volume_corr",
    "btc_trend_1h",
]
TARGET = "target"

DATA_DIR = "data"


def load_split(coin, tf, split):
    """
    Read data/<split>/<COIN>_<tf>_<split>.csv (open_time as the index).

    `split` must be 'train' or 'validate' — test/ is locked and intentionally
    unreachable from here.
    """
    assert split in ("train", "validate"), f"refusing to read split '{split}' (test is locked)"
    path = os.path.join(DATA_DIR, split, f"{coin}_{tf}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python pipeline.py` first to generate Phase 1 data."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def make_xy(df):
    """
    Build features X and binary next-candle target y.

    target = 1 if next close > current close else 0. The last row has no 'next'
    candle, so its target is NaN and the row is dropped. Order is preserved
    (no shuffle) to keep the walk-forward property.
    """
    df = df.copy()
    df[TARGET] = (df["close"].shift(-1) > df["close"]).astype("float")
    df.loc[df.index[-1], TARGET] = np.nan  # last row's future is unknown
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return X, y


def print_report(algo, coin, tf, y_true, y_pred):
    """Print accuracy / precision / recall / F1 for one model on the validation set."""
    acc = accuracy_score(y_true, y_pred)
    # zero_division=0: a model that never predicts UP shouldn't crash the report.
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  [{algo}] {coin} {tf:>3} | "
          f"acc={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  "
          f"(n={len(y_true):,})")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def synth_df(n=400, seed=0):
    """
    Synthetic OHLCV + indicator frame for --selftest. No network, no real data.

    Random-walk close with plausible-looking indicator columns so the trainers
    exercise the full path (features -> target -> fit -> predict -> metrics).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 1, n)
    vol = rng.uniform(100, 1000, n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "rsi": rng.uniform(0, 100, n),
        "macd": rng.normal(0, 1, n),
        "macd_signal": rng.normal(0, 1, n),
        "macd_diff": rng.normal(0, 1, n),
        "bb_high": close + 2, "bb_mid": close, "bb_low": close - 2,
        "atr": rng.uniform(0.5, 3, n),
        "vol_ma20": vol,
        "vol_ratio": rng.uniform(0.5, 2, n),
        # Phase 2b engineered features — dummy but plausible values for --selftest.
        "ema_9": close + rng.normal(0, 0.5, n),
        "ema_21": close + rng.normal(0, 1, n),
        "ema_trend": rng.normal(0, 0.01, n),
        "rsi_change_3": rng.normal(0, 5, n),
        "macd_diff_change_3": rng.normal(0, 0.5, n),
        "higher_high": rng.integers(0, 2, n).astype(float),
        "lower_low": rng.integers(0, 2, n).astype(float),
        "close_position": rng.uniform(0, 1, n),
        "atr_percentile": rng.uniform(0, 1, n),
        "bb_width": rng.uniform(0.01, 0.1, n),
        "volume_zscore": rng.normal(0, 1, n),
        "price_volume_corr": rng.uniform(-1, 1, n),
        "btc_trend_1h": rng.integers(0, 2, n).astype(float),
    }, index=idx)
    return df


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
