"""Lightweight last-price helpers for macro paper fills (CAD)."""

from __future__ import annotations

import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def _series_last_close(close) -> float | None:
    if close is None:
        return None
    if hasattr(close, "iloc"):
        ser = close.iloc[:, 0] if getattr(close, "ndim", 1) > 1 else close
        v = float(ser.iloc[-1])
    else:
        v = float(close)
    if v != v or v <= 0:
        return None
    return v


def fetch_usdcad_last() -> float:
    """Spot USD->CAD multiplier (best-effort)."""
    try:
        d = yf.download("USDCAD=X", period="10d", interval="1d", progress=False, auto_adjust=True)
        if d is None or d.empty:
            return 1.36
        v = _series_last_close(d["Close"])
        if v is None:
            return 1.36
        return v
    except Exception as e:
        logger.debug("USDCAD fetch failed: %s", e)
        return 1.36


def last_close_cad(ticker: str, *, fx_usdcad: float | None = None) -> float | None:
    """Last daily close in CAD for US-listed ticker."""
    fx = float(fx_usdcad) if fx_usdcad is not None else fetch_usdcad_last()
    try:
        h = yf.Ticker(ticker).history(period="10d", auto_adjust=True)
        if h is None or h.empty:
            return None
        usd = _series_last_close(h["Close"])
        if usd is None:
            return None
        return usd * fx
    except Exception as e:
        logger.debug("Price fetch failed for %s: %s", ticker, e)
        return None
