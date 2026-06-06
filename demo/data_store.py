"""Persistent local data store for ETH/USDT K-line data.

Maintains a rolling 30-day CSV. On each update, only fetches new candles
since the last stored timestamp, appends them, and trims old data.
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import (
    fetch_klines_since,
    convert_to_kronos_format,
    get_last_30_days,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "ethusdt_15m.csv")

COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def load_local_data():
    """Read the local CSV. Returns empty DataFrame if it doesn't exist."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamps"])
    return df


def update_local_data():
    """Incrementally update the local data store.

    1. Load existing CSV
    2. Fetch only new candles since last stored timestamp
    3. Append and deduplicate
    4. Trim to last 30 days
    5. Save back to CSV
    6. Return the full DataFrame
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_local_data()
    now_ms = int(time.time() * 1000)

    if df.empty:
        # First run: full 30-day fetch
        df = get_last_30_days()
        df.to_csv(DATA_PATH, index=False)
        return df

    last_ts = df["timestamps"].max()
    last_ts_ms = int(last_ts.timestamp() * 1000)

    if now_ms - last_ts_ms < 60_000:
        # Less than a minute since last candle — nothing new to fetch
        return df

    # Fetch only new candles
    raw = fetch_klines_since(last_ts_ms)
    if raw:
        new_df = convert_to_kronos_format(raw)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamps"]).sort_values("timestamps")

    # Trim to last 30 days
    cutoff = pd.Timestamp.now(tz=df["timestamps"].dt.tz) - pd.Timedelta(days=30)
    df = df[df["timestamps"] >= cutoff]

    df.to_csv(DATA_PATH, index=False)
    return df
