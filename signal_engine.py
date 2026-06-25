"""
Phase 3 — Signal Engine.

Fuses the 3 trained models (LSTM, XGBoost, Random Forest) into selective BUY/SELL
signals. Pulls the CURRENT market state from Binance (the 2025-26 live data lives in
the locked data/test/, so we refetch instead of reading it), runs all 3 models across
15m/1h/4h per coin, and fires a signal only when ALL 6 filters pass. SL/TP are
ATR-based (1:2 R/R). Fired signals go out via telegram_notify.send_signal().

Decision timeframe = 1h (entry, ATR, volume, regime, reported timeframe); 15m & 4h are
alignment confirmation only.

Usage:
    python signal_engine.py            # one live check across all coins
    python signal_engine.py --selftest # synthetic filter-ladder test, no models/data/network
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np

import ml_common as mc
import pipeline as pl
import telegram_notify as tn

SEQ_LEN = 60                 # must match train_lstm.py
LOOKBACK = 750               # recent candles to fetch per coin+timeframe
TIMEFRAMES = ["15m", "1h", "4h"]
DECISION_TF = "1h"
ATR_PCT = 20                 # regime gate: current ATR must beat this percentile

ALGOS = ["lstm", "xgboost", "random_forest"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_recent(symbol, tf, n=LOOKBACK):
    """Fetch the last ~n closed candles + indicators for one symbol/timeframe."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (n + SEQ_LEN + 60) * pl.INTERVAL_MS[tf]  # pad warmup + LSTM window
    df = pl.fetch_klines(symbol, tf, start_ms)
    if df.empty:
        return df
    return pl.add_indicators(df)


# --------------------------------------------------------------------------- #
# Per-model P(UP) on the latest candle
# --------------------------------------------------------------------------- #
def predict_pup(coin, tf, df):
    """Return {'lstm':p, 'xgboost':p, 'random_forest':p} — each P(UP) for the last candle."""
    import joblib

    feats = df[mc.FEATURES]
    out = {}

    # Trees: probability of class 1 (UP) on the latest row (1-row DataFrame keeps names).
    for algo in ("xgboost", "random_forest"):
        model = joblib.load(os.path.join("models", algo, f"{coin}_{tf}.pkl"))
        out[algo] = float(model.predict_proba(feats.iloc[[-1]])[:, 1][0])

    # LSTM: scale the last 60 candles with the saved scaler, predict sigmoid = P(UP).
    import keras
    scaler = joblib.load(os.path.join("models", "lstm", f"{coin}_{tf}_scaler.pkl"))
    lstm = keras.models.load_model(os.path.join("models", "lstm", f"{coin}_{tf}.h5"))
    window = scaler.transform(feats.iloc[-SEQ_LEN:].values).reshape(1, SEQ_LEN, len(mc.FEATURES))
    out["lstm"] = float(lstm.predict(window, verbose=0).ravel()[0])
    return out


# --------------------------------------------------------------------------- #
# Core decision (pure — no I/O, fully unit-testable)
# --------------------------------------------------------------------------- #
def agree(probs):
    """'BUY' if all P(UP)>0.5, 'SELL' if all<0.5, else None (models split)."""
    if all(p > 0.5 for p in probs):
        return "BUY"
    if all(p < 0.5 for p in probs):
        return "SELL"
    return None


def evaluate_coin(coin, tf_probs, candle1h, btc_trend, atr_p20):
    """
    Apply the 6 filters in order. Returns (signal_dict, None) on a full pass, or
    (None, reason) naming the first filter that blocked.

    tf_probs : {"15m":[p,p,p], "1h":[...], "4h":[...]} of P(UP) per model
    candle1h : {"close","atr","volume","vol_ma20"} from the latest 1h candle
    btc_trend: "UP" / "DOWN"
    atr_p20  : 20th-percentile ATR threshold over the recent 1h window
    """
    # 1. All 3 models agree on the 1h (decision) timeframe.
    direction = agree(tf_probs["1h"])
    if direction is None:
        return None, "models disagree (1h)"

    # 2. All 3 timeframes align on the same direction.
    if agree(tf_probs["15m"]) != direction or agree(tf_probs["4h"]) != direction:
        return None, "timeframes not aligned (15m/4h vs 1h)"

    # 3. Confidence threshold on raw mean P(UP) over the 1h models.
    mean_pup = float(np.mean(tf_probs["1h"]))
    if direction == "BUY" and not mean_pup > 0.65:
        return None, f"confidence too low for BUY (mean P(UP)={mean_pup:.2f} <= 0.65)"
    if direction == "SELL" and not mean_pup < 0.35:
        return None, f"confidence too low for SELL (mean P(UP)={mean_pup:.2f} >= 0.35)"

    # 4. Volume confirmation.
    if not candle1h["volume"] > candle1h["vol_ma20"]:
        return None, "volume below 20-period average"

    # 5. Market regime — avoid sideways (ATR in bottom 20%).
    if not candle1h["atr"] > atr_p20:
        return None, "ATR in bottom 20% (sideways market)"

    # 6. BTC correlation — don't fight the market leader.
    if direction == "BUY" and btc_trend == "DOWN":
        return None, "BTC trend opposite (BTC DOWN, signal BUY)"
    if direction == "SELL" and btc_trend == "UP":
        return None, "BTC trend opposite (BTC UP, signal SELL)"

    return build_signal(coin, direction, candle1h, mean_pup), None


def build_signal(coin, direction, candle1h, mean_pup):
    """Build the signal dict with ATR-based SL/TP (1:2 R/R) and directional confidence."""
    entry = candle1h["close"]
    atr = candle1h["atr"]
    if direction == "BUY":
        stop_loss = entry - 2 * atr
        take_profit = entry + 4 * atr
        confidence = mean_pup * 100                  # P(UP)
    else:  # SELL
        stop_loss = entry + 2 * atr
        take_profit = entry - 4 * atr
        confidence = (1 - mean_pup) * 100            # P(DOWN)
    return {
        "coin": coin,
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": round(confidence, 1),
        "timeframe": DECISION_TF,
        "risk_reward": 2.0,
    }


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #
def get_btc_trend():
    """BTC 1h trend: UP if last close > SMA20 (bb_mid), else DOWN. BTC isn't one of the 5 coins."""
    df = fetch_recent("BTCUSDT", DECISION_TF)
    last = df.iloc[-1]
    return "UP" if last["close"] > last["bb_mid"] else "DOWN"


def run():
    print("Signal engine — live check (decision timeframe = 1h)\n")
    btc_trend = get_btc_trend()
    print(f"BTC 1h trend: {btc_trend}\n")

    for coin in mc.COINS:
        # Fetch all timeframes, get per-model P(UP) for the latest candle.
        dfs = {tf: fetch_recent(coin, tf) for tf in TIMEFRAMES}
        tf_probs = {tf: list(predict_pup(coin, tf, dfs[tf]).values()) for tf in TIMEFRAMES}

        df1h = dfs[DECISION_TF]
        last = df1h.iloc[-1]
        candle1h = {"close": float(last["close"]), "atr": float(last["atr"]),
                    "volume": float(last["volume"]), "vol_ma20": float(last["vol_ma20"])}
        atr_p20 = float(np.percentile(df1h["atr"].values, ATR_PCT))

        signal, reason = evaluate_coin(coin, tf_probs, candle1h, btc_trend, atr_p20)
        if signal is None:
            print(f"  {coin}: no signal — blocked by: {reason}")
            continue

        print(f"  {coin}: SIGNAL {signal['direction']} "
              f"entry={signal['entry']:.4g} SL={signal['stop_loss']:.4g} "
              f"TP={signal['take_profit']:.4g} conf={signal['confidence']}%")
        try:
            tn.send_signal(signal)
        except RuntimeError as e:
            # .env not configured — don't crash the whole run, just report.
            print(f"    (Telegram not configured, not sent: {e})")


# --------------------------------------------------------------------------- #
# Self-test (pure logic, no models/data/network)
# --------------------------------------------------------------------------- #
def selftest():
    base = {"close": 100.0, "atr": 2.0, "volume": 1000.0, "vol_ma20": 500.0}
    strong_buy = {"15m": [0.8, 0.85, 0.9], "1h": [0.8, 0.82, 0.86], "4h": [0.78, 0.8, 0.83]}

    # (a) all-pass BUY
    sig, reason = evaluate_coin("ETHUSDT", strong_buy, base, "UP", atr_p20=1.0)
    assert reason is None and sig is not None, f"expected a signal, got {reason}"
    assert sig["direction"] == "BUY"
    assert sig["stop_loss"] == 100.0 - 2 * 2.0 and sig["take_profit"] == 100.0 + 4 * 2.0
    assert sig["risk_reward"] == 2.0 and sig["confidence"] > 75
    assert sig["timeframe"] == "1h"

    # all-pass SELL -> directional confidence should read high
    strong_sell = {tf: [1 - p for p in v] for tf, v in strong_buy.items()}
    sig_s, _ = evaluate_coin("ETHUSDT", strong_sell, base, "DOWN", atr_p20=1.0)
    assert sig_s["direction"] == "SELL" and sig_s["confidence"] > 75
    assert sig_s["stop_loss"] == 100.0 + 2 * 2.0 and sig_s["take_profit"] == 100.0 - 4 * 2.0

    # (b) volume fail
    low_vol = {**base, "volume": 100.0}  # below vol_ma20
    _, reason = evaluate_coin("ETHUSDT", strong_buy, low_vol, "UP", atr_p20=1.0)
    assert reason and reason.startswith("volume"), reason

    # (c) models disagree -> blocked at filter 1
    mixed = {"15m": [0.8, 0.2, 0.9], "1h": [0.8, 0.4, 0.9], "4h": [0.8, 0.8, 0.8]}
    sig2, reason2 = evaluate_coin("ETHUSDT", mixed, base, "UP", atr_p20=1.0)
    assert sig2 is None and "disagree" in reason2, reason2

    # (d) BTC opposite -> blocked at filter 6
    _, reason3 = evaluate_coin("ETHUSDT", strong_buy, base, "DOWN", atr_p20=1.0)
    assert reason3 and reason3.startswith("BTC"), reason3

    print("selftest OK: filter ladder + SL/TP + directional confidence all correct.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
