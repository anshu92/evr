from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _is_tsx_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return t.endswith(".TO") or t.endswith(".V")


def _safe_pct_change(series: pd.Series, periods: int) -> float:
    if series is None or series.empty or len(series) <= periods:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[-1 - periods] - 1.0)


def _rolling_vol(returns: pd.Series, window: int) -> float:
    if returns is None or returns.empty or len(returns.dropna()) < window:
        return float("nan")
    return float(returns.dropna().iloc[-window:].std(ddof=0) * np.sqrt(252.0))


def _rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or close.empty or len(close.dropna()) < period + 1:
        return float("nan")
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])


def _ma_ratio(close: pd.Series, window: int) -> float:
    if close is None or close.empty or len(close.dropna()) < window:
        return float("nan")
    ma = close.rolling(window).mean().iloc[-1]
    if ma == 0 or pd.isna(ma):
        return float("nan")
    return float(close.iloc[-1] / ma - 1.0)


def compute_features(
    prices: pd.DataFrame,
    fx_usdcad: pd.Series,
    liquidity_lookback_days: int,
    feature_lookback_days: int,
    logger,
) -> pd.DataFrame:
    """Compute per-ticker features in CAD terms."""

    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("prices must have MultiIndex columns (ticker, field)")

    tickers: list[str] = list(prices.columns.levels[0])
    idx = pd.to_datetime(prices.index).sort_values()

    # Align FX to trading days, forward-fill.
    fx = fx_usdcad.copy()
    fx.index = pd.to_datetime(fx.index)
    fx = fx.reindex(idx).ffill()

    rows: list[dict[str, object]] = []
    for t in tickers:
        try:
            close = prices[(t, "Close")].astype(float).dropna()
        except Exception:
            continue
        if close.empty:
            continue

        vol = None
        try:
            vol = prices[(t, "Volume")].astype(float).dropna()
        except Exception:
            vol = pd.Series(dtype=float)

        is_tsx = _is_tsx_ticker(t)
        fx_factor = 1.0 if is_tsx else float(fx.loc[close.index].iloc[-1]) if not fx.empty else float("nan")

        # Convert last close to CAD (TSX assumed CAD already).
        last_close_local = float(close.iloc[-1])
        last_close_cad = last_close_local if is_tsx else last_close_local * fx_factor

        # Liquidity: average dollar volume in CAD
        lookback = max(1, int(liquidity_lookback_days))
        close_liq = close.iloc[-lookback:]
        vol_liq = vol.reindex(close_liq.index).fillna(0.0)
        fx_liq = 1.0 if is_tsx else fx.reindex(close_liq.index).ffill()
        dollar_vol = close_liq * vol_liq * fx_liq
        avg_dollar_vol_cad = float(dollar_vol.mean()) if not dollar_vol.empty else float("nan")

        # Returns/volatility
        rets = close.pct_change().dropna()
        ret_20d = _safe_pct_change(close, 20)
        ret_60d = _safe_pct_change(close, 60)
        ret_120d = _safe_pct_change(close, 120)
        vol_20d = _rolling_vol(rets, 20)
        vol_60d = _rolling_vol(rets, 60)

        # Technicals
        rsi_14 = _rsi(close, 14)
        ma20_ratio = _ma_ratio(close, 20)
        ma50_ratio = _ma_ratio(close, 50)
        ma200_ratio = _ma_ratio(close, 200)

        rows.append(
            {
                "ticker": t,
                "is_tsx": is_tsx,
                "last_date": str(pd.to_datetime(close.index[-1]).date()),
                "last_close_local": last_close_local,
                "last_close_cad": last_close_cad,
                "avg_dollar_volume_cad": avg_dollar_vol_cad,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "ret_120d": ret_120d,
                "vol_20d_ann": vol_20d,
                "vol_60d_ann": vol_60d,
                "rsi_14": rsi_14,
                "ma20_ratio": ma20_ratio,
                "ma50_ratio": ma50_ratio,
                "ma200_ratio": ma200_ratio,
                "n_days": int(len(close)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No features computed (empty)")

    # Keep the most recent observations per ticker (should already be unique).
    df = df.drop_duplicates(subset=["ticker"]).set_index("ticker").sort_index()
    logger.info("Computed features: %s tickers", len(df))
    return df


