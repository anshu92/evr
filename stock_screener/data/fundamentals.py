from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from stock_screener.utils import sanitize_ticker


_CACHE_TTL_DAYS = 7
_MAX_WORKERS = 4


def _is_fresh(path: Path, ttl_days: int = 7) -> bool:
    if not path.exists():
        return False
    age = datetime.now(tz=timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age <= timedelta(days=ttl_days)


def _safe_info_value(info: dict[str, Any], key: str) -> Any:
    val = info.get(key)
    if isinstance(val, (int, float, str)):
        return val
    return None


def _build_payload(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "sector": _safe_info_value(info, "sector"),
        "industry": _safe_info_value(info, "industry"),
        "quoteType": _safe_info_value(info, "quoteType"),
        "fundFamily": _safe_info_value(info, "fundFamily"),
        "category": _safe_info_value(info, "category"),
        "marketCap": _safe_info_value(info, "marketCap"),
        "beta": _safe_info_value(info, "beta"),
        # Valuation ratios
        "trailingPE": _safe_info_value(info, "trailingPE"),
        "forwardPE": _safe_info_value(info, "forwardPE"),
        "priceToBook": _safe_info_value(info, "priceToBook"),
        "priceToSalesTrailing12Months": _safe_info_value(info, "priceToSalesTrailing12Months"),
        "enterpriseToRevenue": _safe_info_value(info, "enterpriseToRevenue"),
        "enterpriseToEbitda": _safe_info_value(info, "enterpriseToEbitda"),
        # Profitability
        "profitMargins": _safe_info_value(info, "profitMargins"),
        "operatingMargins": _safe_info_value(info, "operatingMargins"),
        "returnOnEquity": _safe_info_value(info, "returnOnEquity"),
        "returnOnAssets": _safe_info_value(info, "returnOnAssets"),
        # Growth
        "revenueGrowth": _safe_info_value(info, "revenueGrowth"),
        "earningsGrowth": _safe_info_value(info, "earningsGrowth"),
        "earningsQuarterlyGrowth": _safe_info_value(info, "earningsQuarterlyGrowth"),
        # Financial health
        "debtToEquity": _safe_info_value(info, "debtToEquity"),
        "currentRatio": _safe_info_value(info, "currentRatio"),
        "quickRatio": _safe_info_value(info, "quickRatio"),
        # Dividend & payout
        "dividendYield": _safe_info_value(info, "dividendYield"),
        "payoutRatio": _safe_info_value(info, "payoutRatio"),
        # Analyst expectations
        "targetMeanPrice": _safe_info_value(info, "targetMeanPrice"),
        "recommendationMean": _safe_info_value(info, "recommendationMean"),
        "numberOfAnalystOpinions": _safe_info_value(info, "numberOfAnalystOpinions"),
    }


def _fetch_payload(ticker: str, max_retries: int = 2) -> dict[str, Any]:
    """Fetch fundamental data for a ticker with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            info = yf.Ticker(ticker).info
            return _build_payload(info)
        except ConnectionError as e:
            if attempt < max_retries:
                from time import sleep
                sleep(1.0 * attempt)  # 1s, 2s
            else:
                raise ConnectionError(f"Failed to fetch fundamentals for {ticker} after {max_retries} attempts") from e
        except KeyError as e:
            # KeyError usually means invalid ticker or API change
            raise KeyError(f"Invalid ticker or missing data for {ticker}") from e
        except Exception as e:
            # Other exceptions - don't retry
            raise RuntimeError(f"Failed to fetch fundamentals for {ticker}: {str(e)}") from e
    return {}  # Should never reach here


def fetch_fundamentals(tickers: list[str], *, cache_dir: Path, cache_ttl_days: int = 7, logger) -> pd.DataFrame:
    """Fetch minimal fundamentals with caching."""

    out_rows: list[dict[str, Any]] = []
    base = Path(cache_dir) / "fundamentals"
    base.mkdir(parents=True, exist_ok=True)

    to_fetch: list[tuple[str, Path]] = []

    for raw in tickers:
        t = sanitize_ticker(raw)
        if t is None:
            continue
        # Ticker is sanitized before being used in the cache filename.
        path = base / f"{t}.json"
        payload: dict[str, Any] | None = None

        if _is_fresh(path, ttl_days=cache_ttl_days):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in cache for %s: %s", t, str(e))
                payload = None
            except OSError as e:
                logger.warning("Failed to read cache for %s: %s", t, str(e))
                payload = None

        if payload is not None:
            out_rows.append({"ticker": t, **(payload or {})})
        else:
            to_fetch.append((t, path))

    if to_fetch:
        max_workers = min(_MAX_WORKERS, len(to_fetch))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_payload, t): (t, path) for t, path in to_fetch}
            for future in as_completed(futures):
                t, path = futures[future]
                payload: dict[str, Any] = {}
                try:
                    payload = future.result()
                except ConnectionError as exc:
                    logger.error("Connection error fetching fundamentals for %s: %s", t, exc)
                    payload = {}
                except KeyError as exc:
                    logger.warning("Invalid ticker or missing data for %s: %s", t, exc)
                    payload = {}
                except Exception as exc:
                    logger.warning("Could not fetch fundamentals for %s: %s", t, exc)
                    payload = {}
                if payload:
                    try:
                        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass
                out_rows.append({"ticker": t, **payload})

    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["ticker"]).set_index("ticker")
    return df
