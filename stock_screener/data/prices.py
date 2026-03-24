from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import wraps
from time import sleep
from typing import Callable, TypeVar, Iterable

import pandas as pd
import yfinance as yf

from stock_screener.utils import sanitize_ticker


T = TypeVar('T')


def retry_on_exception(
    max_attempts: int = 3,
    delay_seconds: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator to retry a function on exception with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 1
            current_delay = delay_seconds
            last_exception = None
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise
                    
                    # Log retry attempt if logger is available in kwargs
                    logger = kwargs.get('logger')
                    if logger:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt,
                            max_attempts,
                            func.__name__,
                            str(e),
                            current_delay,
                        )
                    
                    sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1
            
            raise last_exception  # Should never reach here
        return wrapper
    return decorator


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
        # Add retry logic for transient failures
        for attempt in range(1, 4):  # 3 attempts
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
                break  # Success, exit retry loop
            except ConnectionError as e:
                if attempt < 3:
                    logger.warning(
                        "Connection error downloading batch (attempt %d/3): %s. Retrying in %ds...",
                        attempt,
                        str(e),
                        attempt * 2,
                    )
                    sleep(attempt * 2)  # Exponential backoff: 2s, 4s
                else:
                    logger.error("Failed to download batch after 3 attempts: %s", str(e))
                    failures.extend(batch)
            except Exception as e:
                logger.warning("Error downloading batch: %s", str(e))
                failures.extend(batch)
                break  # Don't retry for other exceptions
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


def download_intraday_prices(
    tickers: list[str],
    period: str = "5d",
    interval: str = "1h",
    threads: bool = True,
    batch_size: int = 50,
    logger=None,
) -> pd.DataFrame:
    """Download intraday price bars for a small set of tickers.

    Designed for lightweight portfolio monitoring runs — downloads only
    the tickers in the current portfolio + a small watchlist.

    Args:
        tickers: Tickers to download (typically 50-100).
        period: yfinance period string (e.g. "5d", "1mo").
        interval: Bar interval (e.g. "1h", "2h").
        threads: Enable threaded download.
        batch_size: Tickers per yfinance batch call.
        logger: Logger instance.

    Returns:
        DataFrame with MultiIndex columns ``(ticker, field)`` matching
        the format returned by ``download_price_history()``.
        Returns an empty DataFrame if no data is available (e.g. market closed).
    """
    clean: list[str] = []
    for t in tickers:
        s = sanitize_ticker(t)
        if s is not None:
            clean.append(s)
    clean = list(dict.fromkeys(clean))
    if not clean:
        if logger:
            logger.warning("No valid tickers for intraday download")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for batch in _chunks(clean, max(1, int(batch_size))):
        for attempt in range(1, 4):
            try:
                df = yf.download(
                    tickers=" ".join(batch),
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=threads,
                )
                if df is not None and not df.empty:
                    frames.append(df)
                break
            except Exception as e:
                if attempt < 3:
                    if logger:
                        logger.warning(
                            "Intraday download attempt %d/3 failed: %s. Retrying...",
                            attempt, str(e),
                        )
                    sleep(attempt * 1.5)
                else:
                    if logger:
                        logger.error("Intraday download failed after 3 attempts: %s", str(e))
                    failures.extend(batch)
        sleep(0.5)

    if not frames:
        if logger:
            logger.info("No intraday data available (market may be closed). failures=%d", len(failures))
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Normalize to (ticker, field) MultiIndex — same logic as download_price_history.
    if not isinstance(prices.columns, pd.MultiIndex):
        t = clean[0]
        prices.columns = pd.MultiIndex.from_product([[t], prices.columns])
    else:
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if set(prices.columns.get_level_values(0)).issubset(fields):
            prices = prices.swaplevel(0, 1, axis=1)
        prices = prices.sort_index(axis=1)

    if failures and logger:
        logger.warning("Intraday download failures: %d tickers", len(failures))
    if logger:
        logger.info("Downloaded intraday prices: %d tickers, %d bars (%s %s)",
                     len(prices.columns.levels[0]), len(prices), period, interval)
    return prices


def get_latest_prices(prices: pd.DataFrame) -> pd.Series:
    """Extract the most recent close price per ticker from a MultiIndex price DataFrame.

    Args:
        prices: DataFrame with ``(ticker, field)`` MultiIndex columns.

    Returns:
        Series indexed by ticker with the latest close price.
    """
    if prices.empty or not isinstance(prices.columns, pd.MultiIndex):
        return pd.Series(dtype=float)

    tickers = prices.columns.get_level_values(0).unique()
    latest: dict[str, float] = {}
    for t in tickers:
        try:
            close = prices[(t, "Close")].dropna()
            if not close.empty:
                latest[str(t)] = float(close.iloc[-1])
        except Exception:
            continue

    return pd.Series(latest, dtype=float, name="latest_close")


