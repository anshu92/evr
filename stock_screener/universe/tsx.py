from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from stock_screener.config import Config
from stock_screener.utils import Universe, read_json, sanitize_ticker, write_json


def _extract_rows(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        v = obj.get("results")
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


def _yahoo_suffix_for_exchange(exchange: str | None) -> str:
    ex = (exchange or "").upper()
    if "TSXV" in ex or "VENTURE" in ex:
        return ".V"
    return ".TO"


def fetch_tsx_universe(cfg: Config, cache_dir: Path, logger) -> Universe:
    """Fetch TSX/TSXV tickers via TSX directory JSON with cache fallback."""

    cache_file = cache_dir / "universe_tsx.json"
    now = datetime.now(tz=timezone.utc).isoformat()

    base = cfg.tsx_directory_url.rstrip("/")
    # The TSX site currently exposes a letter-based endpoint:
    #   /json/company-directory/search/{exchange}/{letter}
    # where exchange is "tsx" or "tsxv", and letter is A-Z or 0-9.
    url = base
    seen: set[str] = set()
    tickers: list[str] = []
    raw_count = 0
    per_exchange: dict[str, int] = {"tsx": 0, "tsxv": 0}

    try:
        session = requests.Session()
        letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(i) for i in range(0, 10)]
        for exchange_code in ("tsx", "tsxv"):
            suffix = ".TO" if exchange_code == "tsx" else ".V"
            for letter in letters:
                letter_url = f"{url}/{exchange_code}/{letter}"
                resp = session.get(letter_url, timeout=30)
                resp.raise_for_status()
                obj = resp.json()
                rows = _extract_rows(obj)
                raw_count += len(rows)
                for r in rows:
                    sym = r.get("symbol") or r.get("Symbol") or r.get("ticker") or r.get("Ticker")
                    t = sanitize_ticker(str(sym)) if sym is not None else None
                    if t is None:
                        continue
                    yahoo = sanitize_ticker(f"{t}{suffix}")
                    if yahoo is None or yahoo in seen:
                        continue
                    seen.add(yahoo)
                    tickers.append(yahoo)
                    per_exchange[exchange_code] = per_exchange.get(exchange_code, 0) + 1

                if cfg.max_tsx_tickers is not None and len(tickers) >= cfg.max_tsx_tickers:
                    tickers = tickers[: cfg.max_tsx_tickers]
                    break
            if cfg.max_tsx_tickers is not None and len(tickers) >= cfg.max_tsx_tickers:
                break

        meta: dict[str, Any] = {
            "updated_utc": now,
            "source": url,
            "raw_rows": raw_count,
            "total_count": len(tickers),
            "per_exchange": per_exchange,
        }
        write_json(cache_file, {"tickers": tickers, "meta": meta})
        logger.info("TSX universe: %s tickers (cached)", len(tickers))
        return Universe(tickers=tickers, meta=meta)

    except Exception as e:
        logger.warning("TSX universe fetch failed; falling back to cache: %s", e)
        if cache_file.exists():
            cached = read_json(cache_file)
            tickers = [t for t in cached.get("tickers", []) if sanitize_ticker(t)]
            meta = dict(cached.get("meta", {}))
            meta["used_cache_due_to_error"] = str(e)
            logger.info("TSX universe: %s tickers (from cache)", len(tickers))
            return Universe(tickers=tickers, meta=meta)
        raise


