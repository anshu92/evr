from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_screener.utils import read_json, write_json


def fetch_usdcad(fx_ticker: str, lookback_days: int, cache_dir: Path, logger) -> pd.Series:
    """Fetch USD/CAD series (USD->CAD) with retry logic."""

    cache_file = cache_dir / "fx_usdcad.json"
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(days=int(lookback_days * 2))).date().isoformat()

    # Try fetching with retries
    for attempt in range(1, 4):  # 3 attempts
        try:
            df = yf.download(
                fx_ticker,
                start=start,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if df is None or df.empty:
                raise RuntimeError("Empty FX download")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                # Newer yfinance can return MultiIndex columns even for a single ticker.
                close = close.iloc[:, 0]
            series = close.dropna().astype(float)
            series.index = pd.to_datetime(series.index)
            payload = {
                "ticker": fx_ticker,
                "updated_utc": now.isoformat(),
                "close": {str(idx.date()): float(val) for idx, val in series.items()},
            }
            write_json(cache_file, payload)
            logger.info("FX %s: %s rows", fx_ticker, len(series))
            return series
        except ConnectionError as e:
            if attempt < 3:
                logger.warning(
                    "Connection error fetching FX (attempt %d/3): %s. Retrying in %ds...",
                    attempt,
                    str(e),
                    attempt * 2,
                )
                from time import sleep
                sleep(attempt * 2)
            else:
                logger.error("Failed to fetch FX after 3 attempts: %s", str(e))
                break
        except Exception as e:
            logger.warning("FX fetch failed: %s", e)
            break
    
    # Fall back to cache
    logger.warning("FX fetch failed; falling back to cache")
    if cache_file.exists():
        cached = read_json(cache_file)
        close = cached.get("close", {})
        s = pd.Series({pd.to_datetime(k): float(v) for k, v in close.items()}).sort_index()
        logger.info("FX %s: %s rows (from cache)", fx_ticker, len(s))
        return s
    raise RuntimeError(f"FX fetch failed and no cache available for {fx_ticker}")


