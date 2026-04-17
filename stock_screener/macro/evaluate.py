"""Update forward-return outcomes for surfaced macro names (best-effort)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import yfinance as yf

from stock_screener.macro.prices import fetch_usdcad_last


def _parse_iso(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _close_on(ticker: str, day: datetime, fx: float) -> float | None:
    try:
        start = (day - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
        h = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if h is None or h.empty:
            return None
        close = h["Close"]
        if hasattr(close, "iloc"):
            ser = close.iloc[:, 0] if getattr(close, "ndim", 1) > 1 else close
            v = float(ser.iloc[-1])
        else:
            v = float(close)
        if v != v or v <= 0:
            return None
        return v * fx
    except Exception:
        return None


def update_theme_outcomes(conn: sqlite3.Connection, *, horizons: tuple[int, ...] = (1, 5, 20)) -> int:
    """Fill theme_outcomes rows for recent surfaced_names lacking returns."""
    fx = fetch_usdcad_last()
    spy0 = _close_on("SPY", datetime.now(tz=timezone.utc), fx)
    cur = conn.execute(
        """
        SELECT id, theme_id, ticker, as_of_utc, anchor_etf
        FROM surfaced_names
        ORDER BY id DESC
        LIMIT 400
        """
    )
    rows = list(cur.fetchall())
    n = 0
    for r in rows:
        sid = int(r["id"])
        theme_id = str(r["theme_id"])
        ticker = str(r["ticker"]).upper()
        asof = _parse_iso(str(r["as_of_utc"]))
        anchor = str(r["anchor_etf"] or "SPY").upper()
        if asof is None:
            continue
        for h in horizons:
            exists = conn.execute(
                "SELECT 1 FROM theme_outcomes WHERE theme_id = ? AND ticker = ? AND horizon_days = ? AND entry_utc = ?",
                (theme_id, ticker, h, asof.isoformat()),
            ).fetchone()
            if exists:
                continue
            end = asof + timedelta(days=h + 5)
            if datetime.now(tz=timezone.utc) < asof + timedelta(days=max(1, h - 1)):
                continue
            p0 = _close_on(ticker, asof, fx)
            p1 = _close_on(ticker, asof + timedelta(days=h), fx)
            pa0 = _close_on(anchor, asof, fx)
            pa1 = _close_on(anchor, asof + timedelta(days=h), fx)
            ps0 = spy0
            ps1 = _close_on("SPY", asof + timedelta(days=h), fx) if spy0 else None
            if p0 is None or p1 is None or p0 <= 0:
                continue
            ret_abs = (p1 / p0) - 1.0
            ret_anchor = None
            if pa0 and pa1 and pa0 > 0:
                ret_anchor = (p1 / p0) - (pa1 / pa0)
            ret_spy = None
            if ps0 and ps1 and ps0 > 0:
                ret_spy = (p1 / p0) - (ps1 / ps0)
            conn.execute(
                """
                INSERT INTO theme_outcomes (
                    theme_id, ticker, entry_utc, horizon_days, return_abs, return_vs_anchor, return_vs_spy, return_vs_cash, event_type, theme_cluster
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    theme_id,
                    ticker,
                    asof.isoformat(),
                    h,
                    ret_abs,
                    ret_anchor,
                    ret_spy,
                    None,
                    None,
                    None,
                ),
            )
            n += 1
    conn.commit()
    return n
