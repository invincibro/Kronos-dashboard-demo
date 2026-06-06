"""Fetch ETH/USDT K-line data from Binance REST API."""

import time
import requests
import pandas as pd

BINANCE_KLINE_URL = "https://data-api.binance.vision/api/v3/klines"
MAX_LIMIT = 1000
INTERVAL = "5m"
SYMBOL = "ETHUSDT"


def fetch_klines(start_time_ms, end_time_ms, symbol=SYMBOL, interval=INTERVAL):
    """Fetch raw K-lines from Binance with pagination.

    Returns list of raw Binance candle arrays:
    [open_time, open, high, low, close, volume, close_time,
     quote_asset_volume, trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
    """
    all_klines = []
    current_start = start_time_ms

    while current_start < end_time_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": MAX_LIMIT,
        }
        resp = requests.get(BINANCE_KLINE_URL, params=params, timeout=30)
        resp.raise_for_status()
        klines = resp.json()

        if not klines:
            break

        all_klines.extend(klines)

        if len(klines) < MAX_LIMIT:
            break

        current_start = klines[-1][0] + 1
        time.sleep(0.2)

    return all_klines


def fetch_klines_since(since_ms):
    """Fetch K-lines from since_ms to now. Falls back to fetch_klines for large gaps."""
    end_time_ms = int(time.time() * 1000)
    start_ms = since_ms + 1

    if end_time_ms - start_ms <= 0:
        return []

    return fetch_klines(start_ms, end_time_ms)


def convert_to_kronos_format(raw_klines):
    """Convert raw Binance K-lines to Kronos-compatible DataFrame.

    Columns: timestamps, open, high, low, close, volume, amount
    Binance index 7 (quote_asset_volume) -> amount
    """
    if not raw_klines:
        return pd.DataFrame(
            columns=["timestamps", "open", "high", "low", "close", "volume", "amount"]
        )

    rows = []
    for k in raw_klines:
        rows.append(
            {
                "timestamps": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "amount": float(k[7]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def get_last_30_days():
    """Fetch last 30 days of K-lines for a symbol."""
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - 30 * 24 * 60 * 60 * 1000

    raw = fetch_klines(start_time_ms, end_time_ms)
    return convert_to_kronos_format(raw)
