from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from stock_screener.features.fundamental_scores import add_fundamental_composites

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


def _annualized_vol(close: pd.Series, window: int = 20) -> float:
    """Backward-compatible helper used by tests."""
    if close is None or close.empty:
        return float("nan")
    returns = pd.to_numeric(close, errors="coerce").pct_change(fill_method=None)
    return _rolling_vol(returns, window)


def _rsi(close: pd.Series, period: int = 14) -> float:
    if close is None or close.empty or len(close.dropna()) < period + 1:
        return float("nan")
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    # Handle monotonic moves explicitly to avoid NaN RSI:
    # - only gains -> RSI 100
    # - only losses -> RSI 0
    rs = rs.where(~((roll_down == 0) & (roll_up > 0)), np.inf)
    rs = rs.where(~((roll_up == 0) & (roll_down > 0)), 0.0)
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


def _drawdown(close: pd.Series, window: int = 60) -> float:
    """Backward-compatible helper used by tests."""
    return _rolling_drawdown(close, window)


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
    macro: pd.DataFrame | None = None,
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

        # Enforce feature lookback window.
        if feature_lookback_days and int(feature_lookback_days) > 0 and len(close) > int(feature_lookback_days):
            close = close.iloc[-int(feature_lookback_days) :]
            if vol is not None:
                vol = vol.reindex(close.index)

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

        # Momentum quality features
        momentum_reversal = ret_5d - ret_20d  # Short-term reversal signal
        momentum_acceleration = ret_20d - ret_60d  # Momentum change
        ret_20d_lagged = _safe_pct_change(close_cad.iloc[:-5], 20) if len(close_cad) > 25 else float("nan")  # Lagged momentum
        ret_60d_lagged = _safe_pct_change(close_cad.iloc[:-5], 60) if len(close_cad) > 65 else float("nan")  # Lagged long-term momentum

        # HIGH-IMPACT FEATURES: Volatility-adjusted returns (Sharpe-like signals)
        ret_5d_sharpe = ret_5d / vol_20d if vol_20d and not pd.isna(vol_20d) and vol_20d > 0 else float("nan")
        ret_20d_sharpe = ret_20d / vol_20d if vol_20d and not pd.isna(vol_20d) and vol_20d > 0 else float("nan")
        ret_60d_sharpe = ret_60d / vol_60d if vol_60d and not pd.isna(vol_60d) and vol_60d > 0 else float("nan")
        
        # Volume momentum and price-volume divergence
        vol_20d_ago = vol.iloc[-20] if len(vol) > 20 else float("nan")
        vol_5d_ago = vol.iloc[-5] if len(vol) > 5 else float("nan")
        volume_momentum_20d = (vol.iloc[-1] / vol_20d_ago - 1.0) if vol_20d_ago and not pd.isna(vol_20d_ago) and vol_20d_ago > 0 else float("nan")
        volume_surge_5d = (vol.iloc[-1] / vol_5d_ago - 1.0) if vol_5d_ago and not pd.isna(vol_5d_ago) and vol_5d_ago > 0 else float("nan")
        price_volume_div = ret_20d - volume_momentum_20d if not pd.isna(volume_momentum_20d) else float("nan")
        
        # Mean reversion signals
        ma20_zscore = float("nan")
        if len(close_cad) >= 20:
            ma20 = close_cad.rolling(20).mean()
            ma20_std = close_cad.rolling(20).std()
            if ma20_std.iloc[-1] and ma20_std.iloc[-1] > 0:
                ma20_zscore = float((close_cad.iloc[-1] - ma20.iloc[-1]) / ma20_std.iloc[-1])
        mean_reversion_signal = -ret_5d * drawdown_60d if not pd.isna(drawdown_60d) else float("nan")
        
        # Trend consistency (quality of momentum)
        ret_consistency_20d = float("nan")
        up_days_ratio_20d = float("nan")
        if len(rets) >= 20:
            recent_rets = rets.iloc[-20:]
            ret_std = recent_rets.std()
            if ret_std and ret_std > 0:
                ret_consistency_20d = 1.0 / (1.0 + float(ret_std) * np.sqrt(252))
            up_days_ratio_20d = float((recent_rets > 0).sum() / len(recent_rets))

        fx_ret_5d = float(fx.pct_change(5).iloc[-1]) if not is_tsx and len(fx) >= 6 else 0.0
        fx_ret_20d = float(fx.pct_change(20).iloc[-1]) if not is_tsx and len(fx) >= 21 else 0.0

        # Extract fundamental data
        sector = None
        industry = None
        quote_type = None
        fund_family = None
        fund_category = None
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
            quote_type = row.get("quoteType") if isinstance(row, pd.Series) else None
            fund_family = row.get("fundFamily") if isinstance(row, pd.Series) else None
            fund_category = row.get("category") if isinstance(row, pd.Series) else None
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

        # Normalize market cap to CAD for cross-market comparability.
        market_cap_cad = market_cap
        if not is_tsx and market_cap and not pd.isna(market_cap):
            if fx_factor and not pd.isna(fx_factor):
                market_cap_cad = market_cap * float(fx_factor)

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
                # Momentum quality features
                "momentum_reversal": momentum_reversal,
                "momentum_acceleration": momentum_acceleration,
                "ret_20d_lagged": ret_20d_lagged,
                "ret_60d_lagged": ret_60d_lagged,
                # HIGH-IMPACT: Volatility-adjusted returns (Sharpe-like)
                "ret_5d_sharpe": ret_5d_sharpe,
                "ret_20d_sharpe": ret_20d_sharpe,
                "ret_60d_sharpe": ret_60d_sharpe,
                # HIGH-IMPACT: Volume-price signals
                "volume_momentum_20d": volume_momentum_20d,
                "volume_surge_5d": volume_surge_5d,
                "price_volume_div": price_volume_div,
                # HIGH-IMPACT: Mean reversion signals
                "ma20_zscore": ma20_zscore,
                "mean_reversion_signal": mean_reversion_signal,
                # HIGH-IMPACT: Trend quality
                "ret_consistency_20d": ret_consistency_20d,
                "up_days_ratio_20d": up_days_ratio_20d,
                # Relative momentum (computed cross-sectionally after all tickers)
                "relative_momentum_20d": float("nan"),  # Placeholder - filled after DataFrame is built
                "relative_momentum_60d": float("nan"),  # Placeholder - filled after DataFrame is built
                "log_market_cap": float(np.log10(market_cap_cad)) if market_cap_cad and market_cap_cad > 0 else float("nan"),
                "beta": beta,
                "sector": sector,  # Raw sector for target encoding
                "industry": industry,  # Raw industry for target encoding
                "quote_type": quote_type,
                "fund_family": fund_family,
                "fund_category": fund_category,
                "sector_hash": _hash_to_float(str(sector)) if sector else float("nan"),
                "industry_hash": _hash_to_float(str(industry)) if industry else float("nan"),
                "sector_target_enc": float("nan"),  # Placeholder - filled by apply_target_encodings()
                "industry_target_enc": float("nan"),  # Placeholder - filled by apply_target_encodings()
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

    # Cross-sectional ranking (percentile ranks)
    rank_map = {
        "ret_5d": "rank_ret_5d",
        "ret_20d": "rank_ret_20d",
        "ret_60d": "rank_ret_60d",
        "vol_60d_ann": "rank_vol_60d",
        "avg_dollar_volume_cad": "rank_avg_dollar_volume",
        "ret_5d_sharpe": "rank_ret_5d_sharpe",
        "momentum_reversal": "rank_momentum_reversal",
        "ma20_zscore": "rank_ma20_zscore",
    }
    for col, out_col in rank_map.items():
        if col in df.columns:
            df[out_col] = df[col].rank(pct=True)

    if "rank_ret_20d" in df.columns and "rank_ret_60d" in df.columns:
        df["momentum_strength"] = (df["rank_ret_20d"] + df["rank_ret_60d"]) / 2.0
    else:
        df["momentum_strength"] = float("nan")

    if "sector" in df.columns:
        for col in ["ret_20d", "ret_60d", "ret_5d_sharpe", "vol_60d_ann", "momentum_reversal"]:
            out_col = f"sector_rank_{col}"
            if col in df.columns:
                df[out_col] = df.groupby("sector")[col].rank(pct=True)
            else:
                df[out_col] = float("nan")
    else:
        for col in ["ret_20d", "ret_60d", "ret_5d_sharpe", "vol_60d_ann", "momentum_reversal"]:
            df[f"sector_rank_{col}"] = float("nan")

    # Compute RELATIVE MOMENTUM (stock vs cap-weighted market average)
    if "ret_20d" in df.columns and "log_market_cap" in df.columns:
        mcap = np.power(10, df["log_market_cap"].fillna(9))
        weights = mcap / mcap.sum()
        market_ret_20d = (df["ret_20d"].fillna(0) * weights).sum()
        market_ret_60d = (df["ret_60d"].fillna(0) * weights).sum() if "ret_60d" in df.columns else 0
        df["relative_momentum_20d"] = df["ret_20d"] - market_ret_20d
        df["relative_momentum_60d"] = df["ret_60d"] - market_ret_60d if "ret_60d" in df.columns else float("nan")

    # Compute MARKET REGIME features (same value for all stocks on this date)
    # These help the model understand broader market conditions
    if "vol_20d_ann" in df.columns and "log_market_cap" in df.columns:
        # Market volatility regime: cap-weighted average volatility vs historical norm (~15%)
        mcap = np.power(10, df["log_market_cap"].fillna(9))
        weights = mcap / mcap.sum()
        market_vol = (df["vol_20d_ann"].fillna(0.20) * weights).sum()
        historical_norm = 0.15  # ~15% annualized is "normal" volatility
        df["market_vol_regime"] = market_vol / historical_norm  # >1 = high vol, <1 = low vol
    else:
        df["market_vol_regime"] = 1.0

    if "ret_20d" in df.columns and "log_market_cap" in df.columns:
        # Market trend: cap-weighted 20-day return
        mcap = np.power(10, df["log_market_cap"].fillna(9))
        weights = mcap / mcap.sum()
        df["market_trend_20d"] = (df["ret_20d"].fillna(0) * weights).sum()
    else:
        df["market_trend_20d"] = 0.0

    if "ma20_ratio" in df.columns:
        # Market breadth: % of stocks above their 20-day MA
        above_ma = (df["ma20_ratio"] > 0.0).sum()
        total = len(df)
        df["market_breadth"] = above_ma / total if total > 0 else 0.5
    else:
        df["market_breadth"] = 0.5

    if "ret_5d" in df.columns and "ret_20d" in df.columns and "log_market_cap" in df.columns:
        # Market momentum acceleration: difference between short and medium term
        mcap = np.power(10, df["log_market_cap"].fillna(9))
        weights = mcap / mcap.sum()
        market_ret_5d = (df["ret_5d"].fillna(0) * weights).sum()
        market_ret_20d_val = (df["ret_20d"].fillna(0) * weights).sum()
        # Annualize the difference for scale consistency
        df["market_momentum_accel"] = (market_ret_5d * 4) - market_ret_20d_val  # 5d * 4 ≈ 20d
    else:
        df["market_momentum_accel"] = 0.0

    # Feature interactions (must match training-time construction).
    if "ret_20d_sharpe" in df.columns and "momentum_strength" in df.columns:
        df["sharpe_x_rank"] = df["ret_20d_sharpe"] * df["momentum_strength"]
    else:
        df["sharpe_x_rank"] = float("nan")
    if "ret_5d" in df.columns and "vol_20d_ann" in df.columns:
        df["momentum_vol_interaction"] = df["ret_5d"] * df["vol_20d_ann"]
    else:
        df["momentum_vol_interaction"] = float("nan")
    if "rsi_14" in df.columns and "ret_5d" in df.columns:
        df["rsi_momentum_interaction"] = (df["rsi_14"] - 50.0) / 50.0 * df["ret_5d"]
    else:
        df["rsi_momentum_interaction"] = float("nan")
    if "log_market_cap" in df.columns and "relative_momentum_20d" in df.columns:
        df["size_momentum_interaction"] = df["log_market_cap"] * df["relative_momentum_20d"]
    else:
        df["size_momentum_interaction"] = float("nan")
    if "ma20_zscore" in df.columns and "ret_5d" in df.columns:
        df["zscore_reversal"] = -df["ma20_zscore"] * np.sign(df["ret_5d"])
    else:
        df["zscore_reversal"] = float("nan")

    # Attach macro regime features for schema parity (fallback to NaN if unavailable).
    macro_cols = ["vix", "treasury_10y", "treasury_13w", "yield_curve_slope"]
    if macro is not None and not macro.empty and "last_date" in df.columns:
        macro_df = macro.copy()
        macro_df.index = pd.to_datetime(macro_df.index).normalize()
        asof = pd.to_datetime(df["last_date"]).dt.normalize()
        for col in macro_cols:
            if col in macro_df.columns:
                df[col] = asof.map(macro_df[col])
            else:
                df[col] = float("nan")
    else:
        for col in macro_cols:
            df[col] = float("nan")

    # Add fundamental composite scores
    # Note: date_col=None is correct here since compute_features() produces single-date snapshots.
    # Z-scoring across all tickers for the current date IS cross-sectional normalization.
    # This is consistent with training which uses date_col="date" for multi-date panels.
    df = add_fundamental_composites(df, date_col=None)

    # Keep the most recent observations per ticker (should already be unique).
    df = df.drop_duplicates(subset=["ticker"]).set_index("ticker").sort_index()
    logger.info("Computed features: %s tickers", len(df))
    return df


def apply_target_encodings(
    features: pd.DataFrame,
    target_encodings: dict,
    logger,
) -> pd.DataFrame:
    """Apply saved target encodings from training to inference features.
    
    Args:
        features: DataFrame with 'sector' and 'industry' columns
        target_encodings: Dict from training metadata with 'sector', 'industry', and 'global_mean'
        logger: Logger instance
    """
    if not target_encodings:
        logger.warning("No target encodings provided")
        return features
    
    global_mean = target_encodings.get("global_mean", 0.0)
    sector_enc = target_encodings.get("sector", {})
    industry_enc = target_encodings.get("industry", {})
    
    df = features.copy()
    
    # Apply sector encoding
    if sector_enc and "sector" in df.columns:
        df["sector_target_enc"] = df["sector"].map(sector_enc).fillna(global_mean)
        n_mapped = df["sector_target_enc"].notna().sum()
        logger.info("Applied sector target encoding: %d/%d tickers mapped", n_mapped, len(df))
    
    # Apply industry encoding
    if industry_enc and "industry" in df.columns:
        df["industry_target_enc"] = df["industry"].map(industry_enc).fillna(global_mean)
        n_mapped = df["industry_target_enc"].notna().sum()
        logger.info("Applied industry target encoding: %d/%d tickers mapped", n_mapped, len(df))
    
    return df
