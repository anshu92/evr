"""Fetch insider transaction data per ticker from yfinance."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def fetch_insider_features(
    tickers: list[str],
    lookback_days: int = 90,
    cache_dir: Path | None = None,
    cache_ttl_hours: int = 24,
    logger=None,
) -> pd.DataFrame:
    """Fetch insider transactions and compute features for each ticker.

    Returns DataFrame indexed by ticker with columns:
      - insider_net_buys_90d: net shares bought minus sold
      - insider_buy_ratio_90d: ratio of buy transactions to total (0-1)
      - insider_activity_recency: days since most recent transaction
    """
    import yfinance as yf

    rows: list[dict] = []
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)

    for t in tickers:
        # Check cache
        if cache_dir:
            cached = _load_cache(t, cache_dir, cache_ttl_hours)
            if cached is not None:
                rows.append(cached)
                continue

        row = {
            "ticker": t,
            "insider_net_buys_90d": float("nan"),
            "insider_buy_ratio_90d": float("nan"),
            "insider_activity_recency": float("nan"),
        }

        try:
            ticker_obj = yf.Ticker(t)
            txns = ticker_obj.insider_transactions
            if txns is None or (isinstance(txns, pd.DataFrame) and txns.empty):
                rows.append(row)
                if cache_dir:
                    _save_cache(t, row, cache_dir)
                continue

            df = txns.copy()

            # Normalize column names (yfinance may vary)
            col_map = {}
            for c in df.columns:
                cl = str(c).lower().replace(" ", "_")
                if cl in ("start_date", "date"):
                    col_map[c] = "date"
                elif cl == "shares":
                    col_map[c] = "shares"
                elif cl == "transaction":
                    col_map[c] = "transaction"
                elif cl == "text":
                    col_map[c] = "text"
            df = df.rename(columns=col_map)

            # Parse dates and filter to lookback window
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
                df = df.dropna(subset=["date"])
                df = df[df["date"] >= cutoff]

            if df.empty:
                rows.append(row)
                if cache_dir:
                    _save_cache(t, row, cache_dir)
                continue

            # Classify transactions as buy or sell.
            # yfinance puts details in both 'transaction' and 'text' columns
            # (one or both may be populated depending on version).
            txn_text = pd.Series("", index=df.index)
            for _col in ("transaction", "text"):
                if _col in df.columns:
                    _vals = df[_col].astype(str).str.lower().fillna("")
                    txn_text = txn_text.where(txn_text != "", _vals)
                    txn_text = txn_text.fillna("") + " " + _vals
            is_buy = txn_text.str.contains("purchase|buy|acquisition", na=False)
            is_sell = txn_text.str.contains("sale|sell|disposition", na=False)

            # Compute features
            shares_col = "shares" if "shares" in df.columns else None
            if shares_col:
                shares = pd.to_numeric(df[shares_col], errors="coerce").fillna(0).abs()
                buy_shares = float(shares[is_buy].sum())
                sell_shares = float(shares[is_sell].sum())
                row["insider_net_buys_90d"] = buy_shares - sell_shares
            else:
                buy_shares = float(is_buy.sum())
                sell_shares = float(is_sell.sum())
                row["insider_net_buys_90d"] = buy_shares - sell_shares

            total_txns = is_buy.sum() + is_sell.sum()
            if total_txns > 0:
                row["insider_buy_ratio_90d"] = float(is_buy.sum()) / float(total_txns)

            if "date" in df.columns and not df["date"].isna().all():
                most_recent = df["date"].max()
                row["insider_activity_recency"] = float(
                    (datetime.now(tz=timezone.utc) - most_recent).days
                )

        except Exception as e:
            if logger:
                logger.debug("Insider data unavailable for %s: %s", t, e)

        rows.append(row)
        if cache_dir:
            _save_cache(t, row, cache_dir)

    if not rows:
        return pd.DataFrame(
            columns=["insider_net_buys_90d", "insider_buy_ratio_90d", "insider_activity_recency"],
        )

    result = pd.DataFrame(rows).set_index("ticker")
    if logger:
        n_with_data = result["insider_net_buys_90d"].notna().sum()
        logger.info("Insider data: %d/%d tickers have transactions", n_with_data, len(result))
    return result


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / "insiders" / f"{ticker}.json"


def _load_cache(ticker: str, cache_dir: Path, ttl_hours: int) -> dict | None:
    import json
    p = _cache_path(ticker, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("_cached_utc")
        if ts:
            cached_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (datetime.now(tz=timezone.utc) - cached_dt).total_seconds() / 3600.0
            if age > ttl_hours:
                return None
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return None


def _save_cache(ticker: str, row: dict, cache_dir: Path) -> None:
    import json
    p = _cache_path(ticker, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**row, "_cached_utc": datetime.now(tz=timezone.utc).isoformat()}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
