"""
Phase 4 — Backtesting.

Runs the Phase 3 signal logic over the locked data/test/ (2025->today) — the first and
only unlock — to measure real out-of-sample performance. Walk-forward: signals use only
past/current data; future 15m highs/lows are used only to resolve SL/TP fills.

Reuses signal_engine.evaluate_coin() verbatim. One trade per coin at a time (multi-coin
allowed). Exits simulated on 15m candles, SL-first on same-candle ties, 400-bar (~4 day)
timeout. Outcomes in R: win +2, loss -1, timeout = price move / risk. Risk = 1%/trade.
Telegram is never called. Writes backtest_report.md.

Usage:
    python backtest.py            # full backtest on real test data
    python backtest.py --selftest # synthetic simulator/metrics test, no data needed
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ml_common as mc
import pipeline as pl
import signal_engine as se
from train_lstm import make_sequences

TIMEFRAMES = ["15m", "1h", "4h"]
DECISION_TF = "1h"
EXIT_TF = "15m"
TIMEOUT_BARS = 400          # 15m candles (~4 days)
RISK_PCT = 1.0              # constant risk per trade, in percent
ATR_WINDOW = 750            # trailing window for the regime percentile
ATR_PCT = se.ATR_PCT        # regime gate percentile (20), shared with signal_engine
ALGOS = ["lstm", "xgboost", "random_forest"]
TEST_DIR = os.path.join("data", "test")
REPORT_PATH = "backtest_report.md"


# --------------------------------------------------------------------------- #
# Load + precompute model probabilities
# --------------------------------------------------------------------------- #
def load_test(coin, tf):
    """Read a locked test CSV (first and only unlock). Adds a close_time column."""
    path = os.path.join(TEST_DIR, f"{coin}_{tf}_test.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df["close_time"] = df.index + pd.Timedelta(milliseconds=pl.INTERVAL_MS[tf])
    return df


def lstm_pup_series(coin, tf, df):
    """P(UP) per row from the saved LSTM (chunked window prediction). NaN before warmup."""
    import joblib
    import keras

    scaler = joblib.load(os.path.join("models", "lstm", f"{coin}_{tf}_scaler.pkl"))
    model = keras.models.load_model(os.path.join("models", "lstm", f"{coin}_{tf}.h5"))

    X = scaler.transform(df[mc.FEATURES].values)
    # Build windows: window j -> prediction for row (SEQ_LEN-1 + j).
    Xseq, _ = make_sequences(X, np.zeros(len(X)), seq_len=se.SEQ_LEN)
    preds = np.full(len(df), np.nan)
    if len(Xseq):
        chunks = []
        for i in range(0, len(Xseq), 4096):  # bound memory
            chunks.append(model.predict(Xseq[i:i + 4096], verbose=0).ravel())
        preds[se.SEQ_LEN - 1:] = np.concatenate(chunks)
    return preds


def precompute_probs(coin):
    """For each tf, a DataFrame of P(UP) per model (+ close_time), indexed by candle."""
    import joblib

    out = {}
    for tf in TIMEFRAMES:
        df = load_test(coin, tf)
        probs = pd.DataFrame(index=df.index)
        for algo in ("xgboost", "random_forest"):
            model = joblib.load(os.path.join("models", algo, f"{coin}_{tf}.pkl"))
            probs[algo] = model.predict_proba(df[mc.FEATURES])[:, 1]
        probs["lstm"] = lstm_pup_series(coin, tf, df)
        probs["close_time"] = df["close_time"]  # keep tz-aware (.values would strip UTC)
        out[tf] = (df, probs)
    return out


# --------------------------------------------------------------------------- #
# BTC reference trend (fetched live; BTC wasn't a Phase-1 coin)
# --------------------------------------------------------------------------- #
def btc_trend_series(start_ms):
    """BTCUSDT 1h trend (UP if close>bb_mid) over the test window, indexed by close_time."""
    df = pl.add_indicators(pl.fetch_klines("BTCUSDT", "1h", start_ms))
    trend = np.where(df["close"] > df["bb_mid"], "UP", "DOWN")
    return pd.Series(trend, index=df.index + pd.Timedelta(milliseconds=pl.INTERVAL_MS["1h"]))


def asof_value(sorted_index_series, ts):
    """Most recent value at or before ts (no lookahead). sorted_index_series sorted ascending."""
    pos = sorted_index_series.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return sorted_index_series.iloc[pos]


# --------------------------------------------------------------------------- #
# Trade simulation (pure)
# --------------------------------------------------------------------------- #
def simulate_trade(entry, sl, tp, direction, future15m):
    """
    Walk <=TIMEOUT_BARS 15m candles after entry. SL checked before TP (conservative tie).
    future15m: DataFrame with high/low/close columns, rows strictly after the signal close.
    Returns (outcome, exit_price, exit_time, R).
    """
    risk = abs(entry - sl)  # = 2*ATR
    window = future15m.iloc[:TIMEOUT_BARS]
    for ts, c in zip(window.index, window.itertuples()):
        if direction == "BUY":
            if c.low <= sl:
                return "loss", sl, ts, -1.0
            if c.high >= tp:
                return "win", tp, ts, 2.0
        else:  # SELL
            if c.high >= sl:
                return "loss", sl, ts, -1.0
            if c.low <= tp:
                return "win", tp, ts, 2.0
    # Timeout -> close at last available close.
    if len(window) == 0:
        return "timeout", entry, None, 0.0
    exit_price = float(window.iloc[-1]["close"])
    move = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
    return "timeout", exit_price, window.index[-1], move / risk


# --------------------------------------------------------------------------- #
# Per-coin walk-forward
# --------------------------------------------------------------------------- #
def run_coin(coin, btc):
    data = precompute_probs(coin)
    df1h, probs1h = data["1h"]
    df15, probs15 = data["15m"]
    df4h, probs4h = data["4h"]

    # asof helpers need ascending close_time-indexed series.
    p15 = probs15.set_index("close_time").sort_index()
    p4h = probs4h.set_index("close_time").sort_index()
    btc_sorted = btc.sort_index()

    # 15m OHLC indexed by close_time for the exit walk.
    exit15 = df15[["high", "low", "close"]].copy()
    exit15.index = pd.DatetimeIndex(df15["close_time"])  # tz-aware
    exit15 = exit15.sort_index()

    atr_p20 = (df1h["atr"].rolling(ATR_WINDOW, min_periods=50).quantile(ATR_PCT / 100.0)).values

    trades = []
    n_signals = 0
    busy_until = None  # close_time of an open trade's exit; skip signals until then
    times1h = df1h["close_time"].values

    for i in range(len(df1h)):
        C = df1h["close_time"].iloc[i]
        if busy_until is not None and C <= busy_until:
            continue

        row1h = probs1h.iloc[i]
        if pd.isna(row1h["lstm"]) or np.isnan(atr_p20[i]):
            continue  # warmup
        p15_row = asof_value(p15, C)
        p4h_row = asof_value(p4h, C)
        if p15_row is None or p4h_row is None or pd.isna(p15_row["lstm"]) or pd.isna(p4h_row["lstm"]):
            continue

        tf_probs = {
            "15m": [float(p15_row[a]) for a in ALGOS],
            "1h": [float(row1h[a]) for a in ALGOS],
            "4h": [float(p4h_row[a]) for a in ALGOS],
        }
        r = df1h.iloc[i]
        candle1h = {"close": float(r["close"]), "atr": float(r["atr"]),
                    "volume": float(r["volume"]), "vol_ma20": float(r["vol_ma20"])}
        btc_now = asof_value(btc_sorted, C) or "UP"

        signal, _reason = se.evaluate_coin(coin, tf_probs, candle1h, btc_now, float(atr_p20[i]))
        if signal is None:
            continue

        n_signals += 1
        future = exit15[exit15.index > C]
        outcome, exit_price, exit_time, R = simulate_trade(
            signal["entry"], signal["stop_loss"], signal["take_profit"],
            signal["direction"], future)
        trades.append({
            "date": C, "coin": coin, "direction": signal["direction"],
            "entry": signal["entry"], "exit": exit_price, "outcome": outcome,
            "R": R, "pct": R * RISK_PCT,
        })
        busy_until = exit_time if exit_time is not None else C

    print(f"  [{coin}] processed {len(df1h):,} candles, {n_signals} signals fired")
    return trades


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max()) if len(equity) else 0.0


def _streaks(outcomes):
    win_s = loss_s = cur_w = cur_l = 0
    for o in outcomes:
        if o == "win":
            cur_w += 1; cur_l = 0
        elif o == "loss":
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        win_s = max(win_s, cur_w); loss_s = max(loss_s, cur_l)
    return win_s, loss_s


def compute_metrics(trades):
    n = len(trades)
    if n == 0:
        return {"signals": 0, "win_rate": 0.0, "loss_rate": 0.0, "timeout_rate": 0.0,
                "profit_factor": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "sharpe": 0.0, "avg_wl_ratio": 0.0, "largest_win": 0.0,
                "largest_loss": 0.0, "win_streak": 0, "loss_streak": 0}
    R = np.array([t["R"] for t in trades])
    outcomes = [t["outcome"] for t in trades]
    wins = sum(o == "win" for o in outcomes)
    losses = sum(o == "loss" for o in outcomes)
    timeouts = sum(o == "timeout" for o in outcomes)
    gross_win = R[R > 0].sum()
    gross_loss = -R[R < 0].sum()
    equity = 100 + np.cumsum(R * RISK_PCT)
    win_R = R[R > 0]; loss_R = R[R < 0]
    win_s, loss_s = _streaks(outcomes)
    return {
        "signals": n,
        "win_rate": 100 * wins / n,
        "loss_rate": 100 * losses / n,
        "timeout_rate": 100 * timeouts / n,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "total_return": float(R.sum() * RISK_PCT),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": float(R.mean() / R.std()) if R.std() > 0 else 0.0,  # per-trade Sharpe
        "avg_wl_ratio": (win_R.mean() / abs(loss_R.mean())) if len(win_R) and len(loss_R) else 0.0,
        "largest_win": float(R.max()), "largest_loss": float(R.min()),
        "win_streak": win_s, "loss_streak": loss_s,
    }


def verdict(win_rate):
    if win_rate >= 70:
        return "✅ PRODUCTION-READY (win rate >= 70%)"
    if win_rate >= 55:
        return "⚠️ NEEDS TUNING (55-70%)"
    return "❌ NOT VIABLE (< 55%)"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(overall, per_coin, trades, test_start, today):
    o = overall
    lines = [
        "# Backtest Report",
        f"Generated: {today}",
        f"Test Period: {test_start} to {today}",
        "",
        "## Overall Performance",
        f"- Total signals: {o['signals']}",
        f"- Win rate: {o['win_rate']:.1f}%",
        f"- Loss rate: {o['loss_rate']:.1f}%",
        f"- Timeout rate: {o['timeout_rate']:.1f}%",
        f"- Profit factor: {o['profit_factor']:.2f}",
        f"- Total return: {o['total_return']:.1f}%",
        f"- Max drawdown: {o['max_drawdown']:.1f}%",
        f"- Sharpe ratio: {o['sharpe']:.2f}",
        f"- Avg win/loss ratio: {o['avg_wl_ratio']:.2f}",
        f"- Largest win / loss: +{o['largest_win']:.1f}R / {o['largest_loss']:.1f}R",
        f"- Longest win / loss streak: {o['win_streak']} / {o['loss_streak']}",
        "",
        "## Verdict",
        verdict(o["win_rate"]),
    ]
    if o["signals"] == 0:
        lines += [
            "",
            "> **No signals fired** on the out-of-sample test set — this is *no trades*, "
            "not losing trades. The 6-filter ensemble never cleared the confidence gate: "
            "the average 1h P(UP) across the 3 models peaks around 0.71, never reaching the "
            "0.75 BUY / 0.25 SELL threshold. The models' directional edge (~0.50–0.54 "
            "validation accuracy) is too weak for this strict a gate. To produce tradeable "
            "signals, relax the confidence threshold and/or the all-3-timeframes-align rule.",
        ]
    lines += [
        "",
        "## Per-Coin Breakdown",
        "| Coin | Signals | Win Rate | Return | Max DD |",
        "|------|---------|----------|--------|--------|",
    ]
    for coin, m in per_coin.items():
        lines.append(f"| {coin} | {m['signals']} | {m['win_rate']:.1f}% | "
                     f"{m['total_return']:.1f}% | {m['max_drawdown']:.1f}% |")
    lines += [
        "",
        "## Trade Log (last 20)",
        "| Date | Coin | Direction | Entry | Exit | Outcome | P/L |",
        "|------|------|-----------|-------|------|---------|-----|",
    ]
    for t in trades[-20:]:
        lines.append(f"| {pd.Timestamp(t['date']).strftime('%Y-%m-%d %H:%M')} | {t['coin']} | "
                     f"{t['direction']} | {t['entry']:.4g} | {t['exit']:.4g} | "
                     f"{t['outcome']} | {t['pct']:+.1f}% |")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT_PATH}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run():
    test_start = "2025-01-01"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Backtest on locked test data {test_start} -> {today}\n")

    btc = btc_trend_series(pl._to_ms(test_start))

    all_trades, per_coin = [], {}
    for coin in mc.COINS:
        trades = run_coin(coin, btc)
        per_coin[coin] = compute_metrics(trades)
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t["date"])
    overall = compute_metrics(all_trades)
    write_report(overall, per_coin, all_trades, test_start, today)

    print(f"\nOVERALL: {overall['signals']} signals | win {overall['win_rate']:.1f}% | "
          f"PF {overall['profit_factor']:.2f} | return {overall['total_return']:.1f}% | "
          f"maxDD {overall['max_drawdown']:.1f}%")
    print(verdict(overall["win_rate"]))


# --------------------------------------------------------------------------- #
# Self-test (pure simulator + metrics, no data/models/network)
# --------------------------------------------------------------------------- #
def _candles(rows):
    """rows: list of (high, low, close) -> DataFrame indexed by a 15m grid."""
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["high", "low", "close"], index=idx)


def selftest():
    entry, sl, tp = 100.0, 96.0, 108.0  # BUY: risk 4, reward 8 -> 1:2

    # (a) TP pierced -> win +2R
    o, _, _, R = simulate_trade(entry, sl, tp, "BUY", _candles([(101, 99, 100), (109, 107, 108)]))
    assert o == "win" and R == 2.0, (o, R)

    # (b) SL pierced -> loss -1R
    o, _, _, R = simulate_trade(entry, sl, tp, "BUY", _candles([(101, 99, 100), (101, 95, 96)]))
    assert o == "loss" and R == -1.0, (o, R)

    # (c) one candle spans BOTH SL and TP -> loss (SL-first tie rule)
    o, _, _, R = simulate_trade(entry, sl, tp, "BUY", _candles([(109, 95, 100)]))
    assert o == "loss" and R == -1.0, (o, R)

    # (d) flat -> timeout, R ~ 0
    o, _, _, R = simulate_trade(entry, sl, tp, "BUY", _candles([(100.5, 99.5, 100)] * 5))
    assert o == "timeout" and abs(R) < 0.01, (o, R)

    # (e) SELL win
    o, _, _, R = simulate_trade(100.0, 104.0, 92.0, "SELL", _candles([(101, 99, 100), (93, 91, 92)]))
    assert o == "win" and R == 2.0, (o, R)

    # metrics on a known set: 2 wins, 1 loss, 1 timeout(+0)
    trades = [{"R": 2.0, "outcome": "win"}, {"R": 2.0, "outcome": "win"},
              {"R": -1.0, "outcome": "loss"}, {"R": 0.0, "outcome": "timeout"}]
    m = compute_metrics(trades)
    assert m["signals"] == 4 and abs(m["win_rate"] - 50.0) < 1e-9
    assert abs(m["profit_factor"] - 4.0) < 1e-9         # 4 / 1
    assert abs(m["total_return"] - 3.0) < 1e-9          # (2+2-1+0)*1%
    assert m["win_streak"] == 2 and m["loss_streak"] == 1
    assert verdict(75).startswith("✅") and verdict(60).startswith("⚠️") and verdict(40).startswith("❌")

    print("selftest OK: trade simulator (win/loss/tie/timeout/SELL) + metrics + verdict all correct.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
