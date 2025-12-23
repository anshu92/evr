from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Iterable

import pandas as pd
import yfinance as yf

from stock_screener.utils import sanitize_ticker


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def download_price_history(
    tickers: list[str],
    lookback_days: int,
    threads: bool,
    batch_size: int,
    logger,
) -> pd.DataFrame:
    """Download daily adjusted close + volume for tickers in batches."""

    clean: list[str] = []
    for t in tickers:
        s = sanitize_ticker(t)
        if s is not None:
            clean.append(s)
    clean = list(dict.fromkeys(clean))
    if not clean:
        raise ValueError("No valid tickers to download")

    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(days=int(lookback_days * 2))).date().isoformat()

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for batch in _chunks(clean, max(1, int(batch_size))):
        # Be gentle with Yahoo; batched download is more reliable than per-ticker.
        try:
            df = yf.download(
                tickers=" ".join(batch),
                start=start,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=threads,
            )
            if df is None or df.empty:
                failures.extend(batch)
            else:
                frames.append(df)
        except Exception:
            failures.extend(batch)
        # Small backoff to avoid rate limits when universes are large.
        sleep(1.0)

    if not frames:
        raise RuntimeError(f"No price data downloaded. Failures={len(failures)}")

    # If multiple frames, merge by columns (same date index).
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Normalize shape: always return MultiIndex columns (ticker, field).
    # yfinance returns:
    # - single ticker: columns: Open, High, Low, Close, Volume (Index)
    # - multi ticker: columns is MultiIndex with levels (field, ticker) or (ticker, field)
    if not isinstance(prices.columns, pd.MultiIndex):
        # Single ticker; wrap.
        t = clean[0]
        prices.columns = pd.MultiIndex.from_product([[t], prices.columns])
    else:
        # Ensure columns are (ticker, field)
        if prices.columns.names and "Ticker" in prices.columns.names:
            pass
        # Heuristic: if first level contains known fields, swap.
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if set(prices.columns.get_level_values(0)).issubset(fields):
            prices = prices.swaplevel(0, 1, axis=1)
        prices = prices.sort_index(axis=1)

    if failures:
        logger.warning("Price download failures: %s tickers (first 10=%s)", len(failures), failures[:10])
    logger.info("Downloaded price history: %s tickers, %s days", len(prices.columns.levels[0]), len(prices))
    return prices


