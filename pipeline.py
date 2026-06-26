"""
Crypto data pipeline: fetch Binance OHLCV -> indicators -> leakage-free splits -> CSVs.

Fetches OHLCV for 5 coins x 3 timeframes from Binance's public REST endpoint
(no API key), computes 6 technical indicators, validates each frame, splits into
train/validate/test by date with zero overlap, and writes one CSV per
coin+timeframe+split under ./data/.

Usage:
    python pipeline.py             # run the full pipeline
    python pipeline.py --selftest  # run the offline self-check (no network)
"""

import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import ta

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]

# Binance interval code -> pandas frequency alias (for the gap-detection grid).
TIMEFRAMES = {"15m": "15min", "1h": "1h", "4h": "4h"}

# Interval code -> milliseconds (for paginating the fetch loop).
INTERVAL_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000, "1d": 24 * 60 * 60_000}

START = "2021-01-01"  # inclusive start of the whole dataset

# Date windows per split. End is inclusive of the whole day; None = today.
SPLITS = {
    "train":    ("2021-01-01", "2023-12-31"),
    "validate": ("2024-01-01", "2024-12-31"),
    "test":     ("2025-01-01", None),
}
SPLIT_DIR = {"train": "train", "validate": "validate", "test": "test"}

BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = "data"

# Indicator columns (everything that must be non-NaN after the warmup drop).
INDICATOR_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "bb_high", "bb_mid", "bb_low",
    "atr", "vol_ma20", "vol_ratio",
    # Phase 2b engineered features (all trailing-window / past-lookback only).
    "ema_9", "ema_21", "ema_trend",
    "rsi_change_3", "macd_diff_change_3",
    "higher_high", "lower_low", "close_position",
    "atr_percentile", "bb_width",
    "volume_zscore", "price_volume_corr",
    "btc_trend_1h",
]


# --------------------------------------------------------------------------- #
# 1. Fetch
# --------------------------------------------------------------------------- #
def _to_ms(date_str):
    """'YYYY-MM-DD' (UTC midnight) -> epoch milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol, interval, start_ms):
    """
    Page through Binance /api/v3/klines from start_ms to now.

    Binance returns at most 1000 candles per call, so we advance startTime to
    (last_open_time + 1ms) each page and stop when a page is short (the tail).
    Returns a DataFrame indexed by UTC open_time with OHLCV columns.
    """
    rows = []
    cursor = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cursor, "limit": 1000}
        data = _get_with_retry(params)
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break  # short page => reached the present
        cursor = data[-1][0] + 1  # next candle after the last open_time
        time.sleep(0.25)          # stay well under Binance's rate weight

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()

    # The final candle is still forming -> drop it so indicators see only closed bars.
    if not df.empty:
        df = df.iloc[:-1]
    return df


def _get_with_retry(params, attempts=5):
    """GET with a simple linear backoff; raises after exhausting attempts."""
    for i in range(attempts):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            # 429/418 = rate limited; back off harder.
            wait = 2 * (i + 1) if r.status_code in (429, 418) else (i + 1)
            print(f"    HTTP {r.status_code}, retry in {wait}s...")
            time.sleep(wait)
        except requests.RequestException as e:
            print(f"    network error ({e}), retry in {i + 1}s...")
            time.sleep(i + 1)
    raise RuntimeError(f"Binance request failed after {attempts} attempts: {params}")


# --------------------------------------------------------------------------- #
# 2. Indicators
# --------------------------------------------------------------------------- #
def add_indicators(df, btc_trend=None):
    """
    Add indicators + Phase 2b engineered features on the FULL continuous series.

    Computing on the full series before splitting is correct: a feature at time t
    uses only past/current data (trailing lookback), never future data, so this is
    NOT leakage. It also avoids artificial NaNs at split boundaries.

    btc_trend: optional Series of BTC 1h trend (1/0) indexed by BTC *close_time*.
    When given, btc_trend_1h is aligned backward-asof onto each candle's close
    (the most recent BTC 1h candle closed at/before this candle -> no lookahead).
    When None, falls back to a self-referential close>EMA20 (used by synthetic
    selftests and by signal_engine until it passes real BTC trend).
    """
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()

    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_ma20"]

    # --- Phase 2b engineered features (trailing windows only) --------------- #
    # 1. Trend strength
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_trend"] = (df["ema_9"] - df["ema_21"]) / close

    # 2. Momentum divergence (change over the last 3 candles)
    df["rsi_change_3"] = df["rsi"] - df["rsi"].shift(3)
    df["macd_diff_change_3"] = df["macd_diff"] - df["macd_diff"].shift(3)

    # 3. Price action (shift(1) excludes the current candle -> no self-reference)
    df["higher_high"] = (high > high.shift(1).rolling(5).max()).astype(float)
    df["lower_low"] = (low < low.shift(1).rolling(5).min()).astype(float)
    rng = (high - low)
    df["close_position"] = ((close - low) / rng).where(rng > 0, 0.5)  # flat candle -> mid

    # 4. Volatility regime
    df["atr_percentile"] = df["atr"].rolling(100).rank(pct=True)  # trailing percentile rank
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    # 5. Volume profile (guard zero-variance windows so we don't punch mid-series holes)
    vstd = vol.rolling(50).std()
    df["volume_zscore"] = ((vol - vol.rolling(50).mean()) / vstd).where(vstd > 0, 0.0)
    df["price_volume_corr"] = close.rolling(20).corr(vol).fillna(0.0)

    # 6. BTC context (cross-coin) — backward-asof on close_time, else self-referential
    if btc_trend is not None:
        interval = df.index.to_series().diff().median()       # infer candle interval
        ct = df.index + interval                               # this candle's close time
        bt = btc_trend.sort_index()
        pos = bt.index.searchsorted(ct, side="right") - 1     # most recent BTC trend closed <= ct
        vals = np.where(pos >= 0, bt.to_numpy()[np.clip(pos, 0, len(bt) - 1)], 0)
        df["btc_trend_1h"] = vals.astype(float)
    else:
        df["btc_trend_1h"] = (close > close.ewm(span=20, adjust=False).mean()).astype(float)

    # Drop leading warmup rows where any feature is undefined (early 2021 only).
    df = df.dropna(subset=INDICATOR_COLS)
    return df


# --------------------------------------------------------------------------- #
# 3. Gap report
# --------------------------------------------------------------------------- #
def count_missing_candles(df, tf):
    """How many candles are missing vs a perfect grid (reported, not filled)."""
    if df.empty:
        return 0
    expected = pd.date_range(df.index.min(), df.index.max(),
                             freq=TIMEFRAMES[tf], tz="UTC")
    return int(len(expected) - len(df.index))


# --------------------------------------------------------------------------- #
# 4. Split (no leakage)
# --------------------------------------------------------------------------- #
def split_by_date(df):
    """Slice the indicator-complete frame into the date windows. End day inclusive."""
    out = {}
    for name, (start, end) in SPLITS.items():
        lo = pd.Timestamp(start, tz="UTC")
        mask = df.index >= lo
        if end is not None:
            # inclusive of the whole end day -> strictly before the next day
            hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            mask &= df.index < hi
        out[name] = df[mask]
    return out


# --------------------------------------------------------------------------- #
# 5. Validation
# --------------------------------------------------------------------------- #
def validate(df, tf):
    """Return {check_name: bool} for one split frame. True = passed."""
    checks = {}
    checks["non_empty"] = not df.empty
    if df.empty:
        return checks

    checks["no_duplicate_timestamps"] = not df.index.has_duplicates
    checks["no_missing_values"] = int(df[INDICATOR_COLS].isna().sum().sum()) == 0

    # OHLCV sanity
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    checks["high>=low"] = bool((h >= l).all())
    checks["high>=max(o,c)"] = bool((h >= np.maximum(o, c)).all())
    checks["low<=min(o,c)"] = bool((l <= np.minimum(o, c)).all())
    checks["prices_positive"] = bool((df[["open", "high", "low", "close"]] > 0).all().all())
    checks["volume_nonneg"] = bool((v >= 0).all())

    # Indicator ranges
    checks["rsi_0_100"] = bool(df["rsi"].between(0, 100).all())
    checks["bb_ordered"] = bool(((df["bb_high"] >= df["bb_mid"]) &
                                 (df["bb_mid"] >= df["bb_low"])).all())
    checks["atr_nonneg"] = bool((df["atr"] >= 0).all())
    checks["vol_ma_positive"] = bool((df["vol_ma20"] > 0).all())
    checks["vol_ratio_nonneg"] = bool((df["vol_ratio"] >= 0).all())
    return checks


def check_split_overlap(splits):
    """Confirm zero overlap and correct ordering across train/validate/test."""
    tr, va, te = splits["train"], splits["validate"], splits["test"]
    results = {}
    # pairwise disjoint indices
    results["train/validate disjoint"] = tr.index.intersection(va.index).empty
    results["validate/test disjoint"] = va.index.intersection(te.index).empty
    results["train/test disjoint"] = tr.index.intersection(te.index).empty
    # strict chronological ordering (skip a side if a split is empty)
    if not tr.empty and not va.empty:
        results["max(train) < min(validate)"] = tr.index.max() < va.index.min()
    if not va.empty and not te.empty:
        results["max(validate) < min(test)"] = va.index.max() < te.index.min()
    return results


# --------------------------------------------------------------------------- #
# 6. Run + summary
# --------------------------------------------------------------------------- #
def _btc_trend_series(start_ms):
    """BTC 1h trend (close > EMA20 -> 1/0) indexed by close_time, for the BTC context feature."""
    btc = fetch_klines("BTCUSDT", "1h", start_ms)
    trend = (btc["close"] > btc["close"].ewm(span=20, adjust=False).mean()).astype(float)
    trend.index = btc.index + pd.Timedelta(milliseconds=INTERVAL_MS["1h"])  # index by close_time
    return trend


def run():
    start_ms = _to_ms(START)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Crypto data pipeline | {START} -> {today} (UTC)\n")

    for name in SPLIT_DIR.values():
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)

    print("[BTCUSDT 1h] downloading reference trend for btc_trend_1h feature...")
    btc_trend = _btc_trend_series(start_ms)  # fetched once, reused for every coin/timeframe

    summary = []       # per-file rows for the final table
    overlap_report = []  # per coin+tf leakage gate

    for symbol in SYMBOLS:
        for tf, _freq in TIMEFRAMES.items():
            print(f"[{symbol} {tf}] downloading...")
            raw = fetch_klines(symbol, tf, start_ms)
            if raw.empty:
                print(f"    no data returned, skipping")
                continue

            missing = count_missing_candles(raw, tf)
            df = add_indicators(raw, btc_trend=btc_trend)
            print(f"    {len(df):,} candles  {df.index.min().date()} -> "
                  f"{df.index.max().date()}  | missing candles: {missing}")

            splits = split_by_date(df)

            # leakage gate
            overlap = check_split_overlap(splits)
            overlap_report.append((symbol, tf, overlap))

            for split_name, sdf in splits.items():
                checks = validate(sdf, tf)
                path = os.path.join(DATA_DIR, SPLIT_DIR[split_name],
                                    f"{symbol}_{tf}_{split_name}.csv")
                sdf.to_csv(path)
                nan_total = int(sdf[INDICATOR_COLS].isna().sum().sum()) if not sdf.empty else 0
                rng = (f"{sdf.index.min().date()} -> {sdf.index.max().date()}"
                       if not sdf.empty else "—")
                summary.append({
                    "file": os.path.relpath(path, DATA_DIR),
                    "split": split_name,
                    "rows": len(sdf),
                    "range": rng,
                    "missing_vals": nan_total,
                    "failed": [k for k, ok in checks.items() if not ok],
                })

    _print_summary(summary, overlap_report)


def _print_summary(summary, overlap_report):
    print("\n" + "=" * 78)
    print("SUMMARY REPORT")
    print("=" * 78)
    for split_name in SPLIT_DIR:
        print(f"\n[{split_name.upper()}]")
        for row in (r for r in summary if r["split"] == split_name):
            flag = "OK" if not row["failed"] else "FAIL " + ",".join(row["failed"])
            print(f"  {row['file']:<34} rows={row['rows']:>7,}  "
                  f"{row['range']:<24} missing_vals={row['missing_vals']:<3} [{flag}]")

    print("\n" + "-" * 78)
    print("LEAKAGE GATE — zero overlap between splits")
    print("-" * 78)
    all_clean = True
    for symbol, tf, overlap in overlap_report:
        bad = [k for k, ok in overlap.items() if not ok]
        if bad:
            all_clean = False
            print(f"  {symbol} {tf}: FAIL -> {', '.join(bad)}")
    if all_clean:
        print("  All coin/timeframe splits are disjoint and correctly ordered.")

    total_files = len(summary)
    total_fail = sum(1 for r in summary if r["failed"])
    print("\n" + "=" * 78)
    print(f"Wrote {total_files} files | validation failures: {total_fail} | "
          f"leakage: {'CLEAN' if all_clean else 'PROBLEM'}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 7. Self-check (offline, no network)
# --------------------------------------------------------------------------- #
def selftest():
    """Assert split logic + indicator wiring on a synthetic series. No network."""
    idx = pd.date_range("2021-01-01", "2025-06-01", freq="1h", tz="UTC")
    price = pd.Series(np.linspace(100, 500, len(idx)), index=idx)  # rising trend
    df = pd.DataFrame({
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price, "volume": np.full(len(idx), 1000.0),
    })

    df = add_indicators(df)
    assert df[INDICATOR_COLS].isna().sum().sum() == 0, "indicators have NaNs after warmup drop"
    assert df["rsi"].between(0, 100).all(), "RSI out of [0,100]"
    # Monotonically rising prices -> RSI should be very high.
    assert df["rsi"].iloc[-1] > 95, f"RSI on rising series too low: {df['rsi'].iloc[-1]}"
    assert (df["bb_high"] >= df["bb_low"]).all(), "Bollinger bands inverted"

    splits = split_by_date(df)
    tr, va, te = splits["train"], splits["validate"], splits["test"]
    assert len(tr) and len(va) and len(te), "a split is empty"
    # Ordering + zero overlap (the leakage guarantee).
    assert tr.index.max() < va.index.min() < te.index.min(), "splits out of order"
    assert tr.index.intersection(va.index).empty, "train/validate overlap"
    assert va.index.intersection(te.index).empty, "validate/test overlap"
    assert tr.index.intersection(te.index).empty, "train/test overlap"
    # Boundary dates land in the right bucket.
    assert tr.index.max().year == 2023 and va.index.min().year == 2024, "train|validate boundary wrong"
    assert va.index.max().year == 2024 and te.index.min().year == 2025, "validate|test boundary wrong"

    print("selftest OK: indicators valid, splits ordered & non-overlapping.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
