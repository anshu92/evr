from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from stock_screener.utils import sanitize_ticker


_CACHE_TTL_DAYS = 7


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now(tz=timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age <= timedelta(days=_CACHE_TTL_DAYS)


def _safe_info_value(info: dict[str, Any], key: str) -> Any:
    val = info.get(key)
    if isinstance(val, (int, float, str)):
        return val
    return None


def fetch_fundamentals(tickers: list[str], *, cache_dir: Path, logger) -> pd.DataFrame:
    """Fetch minimal fundamentals with caching."""

    out_rows: list[dict[str, Any]] = []
    base = Path(cache_dir) / "fundamentals"
    base.mkdir(parents=True, exist_ok=True)

    for raw in tickers:
        t = sanitize_ticker(raw)
        if t is None:
            continue
        # Ticker is sanitized before being used in the cache filename.
        path = base / f"{t}.json"
        payload: dict[str, Any] | None = None

        if _is_fresh(path):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None

        if payload is None:
            try:
                info = yf.Ticker(t).info
                payload = {
                    "sector": _safe_info_value(info, "sector"),
                    "industry": _safe_info_value(info, "industry"),
                    "marketCap": _safe_info_value(info, "marketCap"),
                    "beta": _safe_info_value(info, "beta"),
                }
                path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not fetch fundamentals for %s: %s", t, exc)
                payload = {}

        out_rows.append({"ticker": t, **(payload or {})})

    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["ticker"]).set_index("ticker")
    return df
