"""
Standalone Telegram signal notifier. No strategy logic.

Given a signal dict, format it and send it to a Telegram chat via
python-telegram-bot. Wire send_signal() into a signal engine later.

Config (loaded from a .env file or env vars, no secrets in code):
    TELEGRAM_BOT_TOKEN   bot token from @BotFather
    TELEGRAM_CHAT_ID     target chat / channel id

Usage:
    python telegram_notify.py --test-telegram   # send a sample signal
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from telegram import Bot

load_dotenv()  # populate os.environ from a local .env if present

REQUIRED_KEYS = [
    "coin", "direction", "entry", "stop_loss",
    "take_profit", "confidence", "timeframe",
]


def _config():
    """Read token + chat id from env; fail loudly if missing."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables."
        )
    return token, chat_id


def format_signal(sig):
    """Build the Telegram message text from a signal dict."""
    missing = [k for k in REQUIRED_KEYS if k not in sig]
    if missing:
        raise ValueError(f"signal missing keys: {missing}")

    direction = str(sig["direction"]).upper()
    emoji = "🟢" if direction == "BUY" else "🔴"
    entry = float(sig["entry"])
    sl = float(sig["stop_loss"])
    tp = float(sig["take_profit"])

    # Use provided R/R, else derive from entry/SL/TP (reward / risk).
    rr = sig.get("risk_reward")
    if rr is None:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk else float("nan")
    rr = float(rr)

    conf = float(sig["confidence"])
    conf_pct = conf * 100 if conf <= 1 else conf  # accept 0.78 or 78

    return (
        f"{emoji} *{direction}*  `{sig['coin']}`  ({sig['timeframe']})\n"
        f"Entry:  `{entry:,.4g}`\n"
        f"Stop:   `{sl:,.4g}`\n"
        f"Target: `{tp:,.4g}`\n"
        f"R/R:    `{rr:.2f}`   |  Confidence: `{conf_pct:.0f}%`"
    )


def send_signal(sig):
    """Format and send a signal to Telegram. Sync wrapper over the async send."""
    token, chat_id = _config()
    text = format_signal(sig)
    bot = Bot(token=token)
    asyncio.run(bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown"))
    print(f"Sent {sig['direction']} {sig['coin']} signal to Telegram.")


def _sample():
    # SL 100 below entry, TP 200 above -> risk/reward = 1:2 = 2.0
    return {
        "coin": "ETHUSDT", "direction": "BUY", "entry": 3200.0,
        "stop_loss": 3100.0, "take_profit": 3400.0, "confidence": 0.78,
        "timeframe": "1h", "risk_reward": 2.0,
    }


if __name__ == "__main__":
    if "--test-telegram" in sys.argv:
        send_signal(_sample())
    else:
        print("Run with --test-telegram to send a sample signal.")
