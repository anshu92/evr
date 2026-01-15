from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


def _hash_to_float(val: str | None) -> float:
    if not val:
        return float("nan")
    h = hashlib.sha256(val.encode("utf-8")).hexdigest()
    return float(int(h[:10], 16) % 1_000_000) / 1_000_000.0


def _rolling_drawdown(close: pd.Series, window: int) -> float:
    if close is None or close.empty or len(close.dropna()) < window:
        return float("nan")
    recent = close.dropna().iloc[-window:]
    roll_max = recent.max()
    if roll_max <= 0 or pd.isna(roll_max):
        return float("nan")
    return float(recent.iloc[-1] / roll_max - 1.0)


def _dist_to_52w_high(close: pd.Series) -> float:
    if close is None or close.empty or len(close.dropna()) < 252:
        return float("nan")
    recent = close.dropna().iloc[-252:]
    hi = recent.max()
    if hi <= 0 or pd.isna(hi):
        return float("nan")
    return float(recent.iloc[-1] / hi - 1.0)


def _dist_to_52w_low(close: pd.Series) -> float:
    if close is None or close.empty or len(close.dropna()) < 252:
        return float("nan")
    recent = close.dropna().iloc[-252:]
    lo = recent.min()
    if lo <= 0 or pd.isna(lo):
        return float("nan")
    return float(recent.iloc[-1] / lo - 1.0)


def compute_features(
    prices: pd.DataFrame,
    fx_usdcad: pd.Series,
    liquidity_lookback_days: int,
    feature_lookback_days: int,
    logger,
    fundamentals: pd.DataFrame | None = None,
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
        fx_series = 1.0 if is_tsx else fx.reindex(close.index).ffill()
        fx_factor = float(fx_series.iloc[-1]) if not is_tsx and not fx_series.empty else 1.0

        # Convert to CAD (TSX assumed CAD already).
        close_cad = close * fx_series
        last_close_local = float(close.iloc[-1])
        last_close_cad = float(close_cad.iloc[-1])

        # Liquidity: average dollar volume in CAD
        lookback = max(1, int(liquidity_lookback_days))
        close_liq = close_cad.iloc[-lookback:]
        vol_liq = vol.reindex(close_liq.index).fillna(0.0)
        dollar_vol = close_liq * vol_liq
        avg_dollar_vol_cad = float(dollar_vol.mean()) if not dollar_vol.empty else float("nan")

        # Returns/volatility
        rets = close_cad.pct_change().dropna()
        ret_5d = _safe_pct_change(close_cad, 5)
        ret_10d = _safe_pct_change(close_cad, 10)
        ret_20d = _safe_pct_change(close_cad, 20)
        ret_60d = _safe_pct_change(close_cad, 60)
        ret_120d = _safe_pct_change(close_cad, 120)
        ret_accel_20_120 = ret_20d - ret_120d
        vol_20d = _rolling_vol(rets, 20)
        vol_60d = _rolling_vol(rets, 60)
        vol_ratio_20_60 = float(vol_20d / vol_60d) if vol_60d and not pd.isna(vol_60d) else float("nan")

        # Technicals
        rsi_14 = _rsi(close_cad, 14)
        ma20_ratio = _ma_ratio(close_cad, 20)
        ma50_ratio = _ma_ratio(close_cad, 50)
        ma200_ratio = _ma_ratio(close_cad, 200)
        drawdown_60d = _rolling_drawdown(close_cad, 60)
        dist_52w_high = _dist_to_52w_high(close_cad)
        dist_52w_low = _dist_to_52w_low(close_cad)

        avg_vol_30d = vol.rolling(30).mean()
        vol_base = float(avg_vol_30d.iloc[-1]) if not avg_vol_30d.empty else float("nan")
        vol_anom_30d = float(vol.iloc[-1] / vol_base) if vol_base and not pd.isna(vol_base) else float("nan")

        fx_ret_5d = float(fx.pct_change(5).iloc[-1]) if not is_tsx and len(fx) >= 6 else 0.0
        fx_ret_20d = float(fx.pct_change(20).iloc[-1]) if not is_tsx and len(fx) >= 21 else 0.0

        # Extract fundamental data
        sector = None
        industry = None
        market_cap = float("nan")
        beta = float("nan")
        trailing_pe = float("nan")
        forward_pe = float("nan")
        price_to_book = float("nan")
        price_to_sales = float("nan")
        enterprise_to_revenue = float("nan")
        enterprise_to_ebitda = float("nan")
        profit_margins = float("nan")
        operating_margins = float("nan")
        return_on_equity = float("nan")
        return_on_assets = float("nan")
        revenue_growth = float("nan")
        earnings_growth = float("nan")
        earnings_quarterly_growth = float("nan")
        debt_to_equity = float("nan")
        current_ratio = float("nan")
        quick_ratio = float("nan")
        dividend_yield = float("nan")
        payout_ratio = float("nan")
        target_mean_price = float("nan")
        recommendation_mean = float("nan")
        num_analyst_opinions = float("nan")
        
        if fundamentals is not None and t in fundamentals.index:
            row = fundamentals.loc[t]
            sector = row.get("sector") if isinstance(row, pd.Series) else None
            industry = row.get("industry") if isinstance(row, pd.Series) else None
            market_cap = float(row.get("marketCap")) if row.get("marketCap") is not None else float("nan")
            beta = float(row.get("beta")) if row.get("beta") is not None else float("nan")
            trailing_pe = float(row.get("trailingPE")) if row.get("trailingPE") is not None else float("nan")
            forward_pe = float(row.get("forwardPE")) if row.get("forwardPE") is not None else float("nan")
            price_to_book = float(row.get("priceToBook")) if row.get("priceToBook") is not None else float("nan")
            price_to_sales = float(row.get("priceToSalesTrailing12Months")) if row.get("priceToSalesTrailing12Months") is not None else float("nan")
            enterprise_to_revenue = float(row.get("enterpriseToRevenue")) if row.get("enterpriseToRevenue") is not None else float("nan")
            enterprise_to_ebitda = float(row.get("enterpriseToEbitda")) if row.get("enterpriseToEbitda") is not None else float("nan")
            profit_margins = float(row.get("profitMargins")) if row.get("profitMargins") is not None else float("nan")
            operating_margins = float(row.get("operatingMargins")) if row.get("operatingMargins") is not None else float("nan")
            return_on_equity = float(row.get("returnOnEquity")) if row.get("returnOnEquity") is not None else float("nan")
            return_on_assets = float(row.get("returnOnAssets")) if row.get("returnOnAssets") is not None else float("nan")
            revenue_growth = float(row.get("revenueGrowth")) if row.get("revenueGrowth") is not None else float("nan")
            earnings_growth = float(row.get("earningsGrowth")) if row.get("earningsGrowth") is not None else float("nan")
            earnings_quarterly_growth = float(row.get("earningsQuarterlyGrowth")) if row.get("earningsQuarterlyGrowth") is not None else float("nan")
            debt_to_equity = float(row.get("debtToEquity")) if row.get("debtToEquity") is not None else float("nan")
            current_ratio = float(row.get("currentRatio")) if row.get("currentRatio") is not None else float("nan")
            quick_ratio = float(row.get("quickRatio")) if row.get("quickRatio") is not None else float("nan")
            dividend_yield = float(row.get("dividendYield")) if row.get("dividendYield") is not None else float("nan")
            payout_ratio = float(row.get("payoutRatio")) if row.get("payoutRatio") is not None else float("nan")
            target_mean_price = float(row.get("targetMeanPrice")) if row.get("targetMeanPrice") is not None else float("nan")
            recommendation_mean = float(row.get("recommendationMean")) if row.get("recommendationMean") is not None else float("nan")
            num_analyst_opinions = float(row.get("numberOfAnalystOpinions")) if row.get("numberOfAnalystOpinions") is not None else float("nan")

        rows.append(
            {
                "ticker": t,
                "is_tsx": is_tsx,
                "last_date": str(pd.to_datetime(close.index[-1]).date()),
                "last_close_local": last_close_local,
                "last_close_cad": last_close_cad,
                "avg_dollar_volume_cad": avg_dollar_vol_cad,
                "ret_5d": ret_5d,
                "ret_10d": ret_10d,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "ret_120d": ret_120d,
                "ret_accel_20_120": ret_accel_20_120,
                "vol_20d_ann": vol_20d,
                "vol_60d_ann": vol_60d,
                "vol_ratio_20_60": vol_ratio_20_60,
                "rsi_14": rsi_14,
                "ma20_ratio": ma20_ratio,
                "ma50_ratio": ma50_ratio,
                "ma200_ratio": ma200_ratio,
                "drawdown_60d": drawdown_60d,
                "dist_52w_high": dist_52w_high,
                "dist_52w_low": dist_52w_low,
                "vol_anom_30d": vol_anom_30d,
                "log_market_cap": float(np.log10(market_cap)) if market_cap and market_cap > 0 else float("nan"),
                "beta": beta,
                "sector_hash": _hash_to_float(str(sector)) if sector else float("nan"),
                "industry_hash": _hash_to_float(str(industry)) if industry else float("nan"),
                "fx_ret_5d": fx_ret_5d,
                "fx_ret_20d": fx_ret_20d,
                # Raw fundamental fields
                "trailing_pe": trailing_pe,
                "forward_pe": forward_pe,
                "price_to_book": price_to_book,
                "price_to_sales": price_to_sales,
                "enterprise_to_revenue": enterprise_to_revenue,
                "enterprise_to_ebitda": enterprise_to_ebitda,
                "profit_margins": profit_margins,
                "operating_margins": operating_margins,
                "return_on_equity": return_on_equity,
                "return_on_assets": return_on_assets,
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
                "earnings_quarterly_growth": earnings_quarterly_growth,
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "quick_ratio": quick_ratio,
                "dividend_yield": dividend_yield,
                "payout_ratio": payout_ratio,
                "target_mean_price": target_mean_price,
                "recommendation_mean": recommendation_mean,
                "num_analyst_opinions": num_analyst_opinions,
                "n_days": int(len(close)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No features computed (empty)")

    rank_map = {
        "ret_20d": "rank_ret_20d",
        "ret_60d": "rank_ret_60d",
        "vol_60d_ann": "rank_vol_60d",
        "avg_dollar_volume_cad": "rank_avg_dollar_volume",
    }
    for col, out_col in rank_map.items():
        if col in df.columns:
            df[out_col] = df[col].rank(pct=True)

    # Composite fundamental features
    def _safe_zscore(series: pd.Series) -> pd.Series:
        """Z-score with NaN handling."""
        mu = series.mean()
        sd = series.std()
        if sd == 0 or pd.isna(sd):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sd

    # Value score: inverse of valuation ratios (lower is better)
    value_components = []
    if "trailing_pe" in df.columns:
        inv_pe = 1.0 / df["trailing_pe"].replace([0, np.inf, -np.inf], np.nan)
        if inv_pe.notna().sum() > 0:
            value_components.append(_safe_zscore(inv_pe))
    if "price_to_book" in df.columns:
        inv_pb = 1.0 / df["price_to_book"].replace([0, np.inf, -np.inf], np.nan)
        if inv_pb.notna().sum() > 0:
            value_components.append(_safe_zscore(inv_pb))
    if "price_to_sales" in df.columns:
        inv_ps = 1.0 / df["price_to_sales"].replace([0, np.inf, -np.inf], np.nan)
        if inv_ps.notna().sum() > 0:
            value_components.append(_safe_zscore(inv_ps))
    
    if value_components:
        df["value_score"] = pd.concat(value_components, axis=1).mean(axis=1)
    else:
        df["value_score"] = 0.0

    # Quality score: profitability and efficiency
    quality_components = []
    if "return_on_equity" in df.columns:
        quality_components.append(_safe_zscore(df["return_on_equity"]))
    if "operating_margins" in df.columns:
        quality_components.append(_safe_zscore(df["operating_margins"]))
    if "profit_margins" in df.columns:
        quality_components.append(_safe_zscore(df["profit_margins"]))
    
    if quality_components:
        df["quality_score"] = pd.concat(quality_components, axis=1).mean(axis=1)
    else:
        df["quality_score"] = 0.0

    # Growth score: revenue and earnings growth
    growth_components = []
    if "revenue_growth" in df.columns:
        growth_components.append(_safe_zscore(df["revenue_growth"]))
    if "earnings_growth" in df.columns:
        growth_components.append(_safe_zscore(df["earnings_growth"]))
    
    if growth_components:
        df["growth_score"] = pd.concat(growth_components, axis=1).mean(axis=1)
    else:
        df["growth_score"] = 0.0

    # Surprise factors: analyst expectations vs current price
    if "target_mean_price" in df.columns and "last_close_cad" in df.columns:
        df["pe_discount"] = (df["target_mean_price"] - df["last_close_cad"]) / df["last_close_cad"].replace(0, np.nan)
        df["pe_discount"] = df["pe_discount"].replace([np.inf, -np.inf], np.nan)
    else:
        df["pe_discount"] = 0.0

    # Fundamental momentum: quarterly earnings growth change
    if "earnings_quarterly_growth" in df.columns:
        df["roc_growth"] = df["earnings_quarterly_growth"]  # Simplified - would need historical data for true ROC
    else:
        df["roc_growth"] = 0.0

    # Interaction features
    if "value_score" in df.columns and "ret_120d" in df.columns:
        df["value_momentum"] = df["value_score"] * df["ret_120d"]
    else:
        df["value_momentum"] = 0.0

    if "vol_60d_ann" in df.columns and "log_market_cap" in df.columns:
        df["vol_size"] = df["vol_60d_ann"] * df["log_market_cap"]
    else:
        df["vol_size"] = 0.0

    if "quality_score" in df.columns and "growth_score" in df.columns:
        df["quality_growth"] = df["quality_score"] * df["growth_score"]
    else:
        df["quality_growth"] = 0.0

    # Keep the most recent observations per ticker (should already be unique).
    df = df.drop_duplicates(subset=["ticker"]).set_index("ticker").sort_index()
    logger.info("Computed features: %s tickers", len(df))
    return df


