# Crypto Data Pipeline

Fetches Binance OHLCV for 5 coins (ETH, SOL, XRP, BNB, DOGE) across 3 timeframes
(15m, 1h, 4h), computes technical indicators (RSI, MACD, Bollinger Bands, ATR,
volume MA, volume ratio), validates each frame, and splits into
train/validate/test by date with **zero leakage**. Includes a standalone Telegram
notifier for formatted trade signals.

## Setup

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Telegram** — copy the template and fill in your bot token + chat id:
   ```bash
   cp .env.example .env      # Windows: copy .env.example .env
   ```
   Edit `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC-your-token
   TELEGRAM_CHAT_ID=123456789
   ```
   Get a token from [@BotFather](https://t.me/BotFather); get your chat id by
   messaging the bot and checking `https://api.telegram.org/bot<TOKEN>/getUpdates`.
   `.env` is gitignored — secrets never get committed.

3. **Run the offline self-check** (no network, no Telegram needed):
   ```bash
   python pipeline.py --selftest
   ```
   Asserts indicators are valid and the splits are ordered and non-overlapping.

4. **Send a test Telegram signal** (needs `.env` filled in):
   ```bash
   python telegram_notify.py --test-telegram
   ```
   Sends a sample ETHUSDT BUY signal (entry 3200, SL 3100, TP 3400, R/R 2.0).

5. **Run the full pipeline** — downloads 2021→today and writes CSVs under `data/`:
   ```bash
   python pipeline.py
   ```
   Takes ~15–20 min (the 15m timeframe back to 2021 is the slow part). Prints
   per-coin progress, then a summary + leakage-gate report. Output:
   ```
   data/train/      <COIN>_<tf>_train.csv
   data/validate/   <COIN>_<tf>_validate.csv
   data/test/       <COIN>_<tf>_test.csv
   ```

## Notes

- **Splits (no leakage):** train `2021-01-01 → 2023-12-31`, validate `2024`,
  test `2025-01-01 → today`. Indicators are computed on the full continuous
  series *before* slicing (past-only lookback is not leakage), and no global
  scaling is applied. The pipeline asserts zero index overlap across splits.
- **Data is not committed.** `data/` is gitignored — re-create it with `python pipeline.py`.
- On Windows, `python telegram_notify.py` may print a `UnicodeEncodeError` for the
  emoji — that's only the console; Telegram receives the message fine over UTF-8.
