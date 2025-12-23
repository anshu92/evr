from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_screener.config import Config
from stock_screener.utils import Universe, read_json, sanitize_ticker, write_json


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _parse_symbol_dir(text: str, symbol_col: str = "Symbol") -> list[str]:
    # nasdaqlisted.txt and otherlisted.txt are pipe-delimited with an EOF marker.
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    rows: list[list[str]] = []
    header: list[str] | None = None
    for ln in lines:
        if ln.startswith("EOF"):
            break
        parts = ln.split("|")
        if header is None:
            header = parts
            continue
        rows.append(parts)
    if not header:
        return []
    df = pd.DataFrame(rows, columns=header)
    if symbol_col not in df.columns:
        return []
    raw = df[symbol_col].astype(str).tolist()
    out: list[str] = []
    for r in raw:
        t = sanitize_ticker(r)
        if t is None:
            continue
        # Exclude test issues/special placeholders
        if t in {"N/A"}:
            continue
        out.append(t)
    return out


def fetch_us_universe(cfg: Config, cache_dir: Path, logger) -> Universe:
    """Fetch US listed tickers over HTTPS with cache fallback."""

    cache_file = cache_dir / "universe_us.json"
    now = datetime.now(tz=timezone.utc).isoformat()

    try:
        resp1 = requests.get(NASDAQ_LISTED_URL, timeout=30)
        resp1.raise_for_status()
        resp2 = requests.get(OTHER_LISTED_URL, timeout=30)
        resp2.raise_for_status()
        nasdaq = _parse_symbol_dir(resp1.text, symbol_col="Symbol")
        other = _parse_symbol_dir(resp2.text, symbol_col="ACT Symbol")

        tickers = list(dict.fromkeys(nasdaq + other))
        if cfg.max_us_tickers is not None:
            tickers = tickers[: cfg.max_us_tickers]

        meta: dict[str, Any] = {
            "updated_utc": now,
            "source": "nasdaqtrader_https",
            "nasdaq_count": len(nasdaq),
            "other_count": len(other),
            "total_count": len(tickers),
        }
        write_json(cache_file, {"tickers": tickers, "meta": meta})
        logger.info("US universe: %s tickers (cached)", len(tickers))
        return Universe(tickers=tickers, meta=meta)

    except Exception as e:
        logger.warning("US universe fetch failed; falling back to cache: %s", e)
        if cache_file.exists():
            cached = read_json(cache_file)
            tickers = [t for t in cached.get("tickers", []) if sanitize_ticker(t)]
            meta = dict(cached.get("meta", {}))
            meta["used_cache_due_to_error"] = str(e)
            logger.info("US universe: %s tickers (from cache)", len(tickers))
            return Universe(tickers=tickers, meta=meta)
        raise


