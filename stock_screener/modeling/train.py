from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce noise
except ImportError:
    optuna = None

from stock_screener.config import Config
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.fundamentals import fetch_fundamentals
from stock_screener.data.prices import download_price_history
from stock_screener.data.macro import fetch_macro_indicators
from stock_screener.features.fundamental_scores import add_fundamental_composites
from stock_screener.modeling.eval import (
    build_time_splits,
    build_walk_forward_periods,
    evaluate_predictions,
    evaluate_topn_returns,
    compute_calibration,
    compute_portfolio_metrics,
    simulate_realistic_portfolio,
    aggregate_walk_forward_results,
)
from stock_screener.modeling.model import (
    FEATURE_COLUMNS,
    TECHNICAL_FEATURES_ONLY,
    build_model,
    build_lgbm_model,
    load_bundle,
    load_model,
    predict_ensemble,
    predict_score,
    save_ensemble,
    save_model,
)
from stock_screener.modeling.transform import normalize_features_cross_section, winsorize_mad, build_calibration_map
from stock_screener.universe.tsx import fetch_tsx_universe
from stock_screener.universe.us import fetch_us_universe
from stock_screener.utils import Universe, ensure_dir, write_json, suppress_external_warnings

# Suppress known external library warnings
suppress_external_warnings()


@dataclass(frozen=True)
class TrainResult:
    n_samples: int
    n_tickers: int
    horizon_days: int


def _is_tsx_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return t.endswith(".TO") or t.endswith(".V")


def _hash_to_float(val: str | None) -> float:
    if not val:
        return float("nan")
    h = hashlib.sha256(val.encode("utf-8")).hexdigest()
    return float(int(h[:10], 16) % 1_000_000) / 1_000_000.0


def _apply_target_encoding(
    panel: pd.DataFrame,
    cat_col: str,
    target_col: str,
    date_col: str = "date",
    smoothing: float = 10.0,
) -> pd.Series:
    """Apply target encoding with smoothing to prevent overfitting.
    
    Uses expanding window with vectorized merge - fast O(n) implementation.
    No future data leakage - uses only historical data for each date.
    """
    if cat_col not in panel.columns or panel[cat_col].isna().all():
        return pd.Series(np.nan, index=panel.index)
    
    # Use expanding mean for global_mean to avoid future leakage
    # For simplicity, use 0.0 as the initial global mean (neutral)
    global_mean = 0.0
    
    # Compute per-date aggregates per category
    daily_agg = panel.groupby([date_col, cat_col])[target_col].agg(["sum", "count"]).reset_index()
    daily_agg.columns = [date_col, cat_col, "day_sum", "day_count"]
    
    # Sort dates and pivot to get (date x category) matrix
    dates_sorted = sorted(daily_agg[date_col].unique())
    all_cats = daily_agg[cat_col].unique()
    
    # Create pivot tables for cumulative computation
    sum_pivot = daily_agg.pivot_table(
        index=date_col, columns=cat_col, values="day_sum", fill_value=0, aggfunc="sum"
    ).reindex(dates_sorted, fill_value=0)
    count_pivot = daily_agg.pivot_table(
        index=date_col, columns=cat_col, values="day_count", fill_value=0, aggfunc="sum"
    ).reindex(dates_sorted, fill_value=0)
    
    # Compute LAGGED cumulative sums (shift by 1 to exclude current date)
    cumsum = sum_pivot.cumsum().shift(1, fill_value=0)
    cumcount = count_pivot.cumsum().shift(1, fill_value=0)
    
    # Compute smoothed encoding: (n * mean + m * global) / (n + m)
    # mean = cumsum / cumcount, so: (cumsum + m * global) / (cumcount + m)
    encoded_pivot = (cumsum + smoothing * global_mean) / (cumcount + smoothing)
    
    # Melt back to long format for merge
    encoded_long = encoded_pivot.reset_index().melt(
        id_vars=[date_col], var_name=cat_col, value_name="_enc"
    )
    
    # Merge with original panel
    panel_with_idx = panel[[date_col, cat_col]].copy()
    panel_with_idx["_orig_idx"] = panel.index
    
    merged = panel_with_idx.merge(encoded_long, on=[date_col, cat_col], how="left")
    merged = merged.set_index("_orig_idx").sort_index()
    
    result = merged["_enc"].fillna(global_mean)
    result.index = panel.index
    return result


def _compute_ticker_features(
    t: str,
    close: pd.Series,
    vol: pd.Series,
    idx: pd.Index,
    fx: pd.Series,
    fx_ret_5d: pd.Series,
    fx_ret_20d: pd.Series,
    fundamentals: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Compute features for a single ticker. Returns DataFrame or None if invalid."""
    is_tsx = _is_tsx_ticker(t)
    fx_series = 1.0 if is_tsx else fx
    fx_factor = float(fx_series.iloc[-1]) if not is_tsx and hasattr(fx_series, "iloc") and len(fx_series) else 1.0
    close_cad = close * fx_series

    rets = close_cad.pct_change(fill_method=None)
    vol_20 = rets.rolling(20).std(ddof=0) * np.sqrt(252.0)
    vol_60 = rets.rolling(60).std(ddof=0) * np.sqrt(252.0)
    vol_ratio_20_60 = vol_20 / vol_60.replace(0.0, np.nan)

    # RSI(14)
    delta = close_cad.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    ma20 = close_cad.rolling(20).mean()
    ma50 = close_cad.rolling(50).mean()
    ma200 = close_cad.rolling(200).mean()

    roll_max_60 = close_cad.rolling(60).max()
    drawdown_60d = close_cad / roll_max_60.replace(0.0, np.nan) - 1.0
    roll_max_252 = close_cad.rolling(252).max()
    roll_min_252 = close_cad.rolling(252).min()
    dist_52w_high = close_cad / roll_max_252.replace(0.0, np.nan) - 1.0
    dist_52w_low = close_cad / roll_min_252.replace(0.0, np.nan) - 1.0

    ret_5d = close_cad.pct_change(5, fill_method=None)
    ret_20d = close_cad.pct_change(20, fill_method=None)
    ret_60d = close_cad.pct_change(60, fill_method=None)
    momentum_reversal = ret_5d - ret_20d
    momentum_acceleration = ret_20d - ret_60d
    ret_20d_lagged = ret_20d.shift(5)
    ret_60d_lagged = ret_60d.shift(5)

    # Sharpe-like signals
    ret_5d_sharpe = ret_5d / vol_20.replace(0, np.nan)
    ret_20d_sharpe = ret_20d / vol_20.replace(0, np.nan)
    ret_60d_sharpe = ret_60d / vol_60.replace(0, np.nan)

    # Volume features
    vol_20d_ago = vol.shift(20)
    vol_5d_ago = vol.shift(5)
    volume_momentum_20d = (vol / vol_20d_ago.replace(0, np.nan)) - 1.0
    volume_surge_5d = (vol / vol_5d_ago.replace(0, np.nan)) - 1.0
    price_volume_div = ret_20d - volume_momentum_20d

    # Mean reversion
    ma20_std = close_cad.rolling(20).std()
    ma20_zscore = (close_cad - ma20) / ma20_std.replace(0, np.nan)
    mean_reversion_signal = -ret_5d * drawdown_60d

    # Trend consistency
    rolling_std_20 = rets.rolling(20).std()
    ret_consistency_20d = 1.0 / (1.0 + rolling_std_20 * np.sqrt(252))
    up_days_20d = (rets > 0).rolling(20).sum()
    up_days_ratio_20d = up_days_20d / 20.0

    vol_anom_30d = vol / vol.rolling(30).mean().replace(0.0, np.nan)
    avg_dollar_vol = (close_cad * vol).rolling(30).mean()
    n_days = close_cad.expanding().count()

    # Fundamentals
    sector_hash = np.nan
    industry_hash = np.nan
    log_market_cap = np.nan
    beta = np.nan
    trailing_pe = forward_pe = price_to_book = price_to_sales = np.nan
    enterprise_to_revenue = enterprise_to_ebitda = profit_margins = operating_margins = np.nan
    return_on_equity = return_on_assets = revenue_growth = earnings_growth = np.nan
    earnings_quarterly_growth = debt_to_equity = current_ratio = quick_ratio = np.nan
    dividend_yield = payout_ratio = target_mean_price = recommendation_mean = num_analyst_opinions = np.nan
    sector = None
    industry = None

    if fundamentals is not None and t in fundamentals.index:
        row = fundamentals.loc[t]
        sector = str(row.get("sector")) if row.get("sector") else None
        industry = str(row.get("industry")) if row.get("industry") else None
        sector_hash = _hash_to_float(sector) if sector else np.nan
        industry_hash = _hash_to_float(industry) if industry else np.nan
        market_cap = row.get("marketCap")
        market_cap_cad = None
        if market_cap is not None:
            market_cap_cad = float(market_cap)
            if not is_tsx and fx_factor and not pd.isna(fx_factor):
                market_cap_cad = market_cap_cad * float(fx_factor)
        log_market_cap = float(np.log10(market_cap_cad)) if market_cap_cad and float(market_cap_cad) > 0 else np.nan
        beta = float(row.get("beta")) if row.get("beta") is not None else np.nan
        # Extract other fundamentals
        trailing_pe = float(row.get("trailingPE")) if row.get("trailingPE") is not None else np.nan
        forward_pe = float(row.get("forwardPE")) if row.get("forwardPE") is not None else np.nan
        price_to_book = float(row.get("priceToBook")) if row.get("priceToBook") is not None else np.nan
        price_to_sales = float(row.get("priceToSalesTrailing12Months")) if row.get("priceToSalesTrailing12Months") is not None else np.nan
        enterprise_to_revenue = float(row.get("enterpriseToRevenue")) if row.get("enterpriseToRevenue") is not None else np.nan
        enterprise_to_ebitda = float(row.get("enterpriseToEbitda")) if row.get("enterpriseToEbitda") is not None else np.nan
        profit_margins = float(row.get("profitMargins")) if row.get("profitMargins") is not None else np.nan
        operating_margins = float(row.get("operatingMargins")) if row.get("operatingMargins") is not None else np.nan
        return_on_equity = float(row.get("returnOnEquity")) if row.get("returnOnEquity") is not None else np.nan
        return_on_assets = float(row.get("returnOnAssets")) if row.get("returnOnAssets") is not None else np.nan
        revenue_growth = float(row.get("revenueGrowth")) if row.get("revenueGrowth") is not None else np.nan
        earnings_growth = float(row.get("earningsGrowth")) if row.get("earningsGrowth") is not None else np.nan
        earnings_quarterly_growth = float(row.get("earningsQuarterlyGrowth")) if row.get("earningsQuarterlyGrowth") is not None else np.nan
        debt_to_equity = float(row.get("debtToEquity")) if row.get("debtToEquity") is not None else np.nan
        current_ratio = float(row.get("currentRatio")) if row.get("currentRatio") is not None else np.nan
        quick_ratio = float(row.get("quickRatio")) if row.get("quickRatio") is not None else np.nan
        dividend_yield = float(row.get("dividendYield")) if row.get("dividendYield") is not None else np.nan
        payout_ratio = float(row.get("payoutRatio")) if row.get("payoutRatio") is not None else np.nan
        target_mean_price = float(row.get("targetMeanPrice")) if row.get("targetMeanPrice") is not None else np.nan
        recommendation_mean = float(row.get("recommendationMean")) if row.get("recommendationMean") is not None else np.nan
        num_analyst_opinions = float(row.get("numberOfAnalystOpinions")) if row.get("numberOfAnalystOpinions") is not None else np.nan

    fx_ret_5d_series = 0.0 if is_tsx else fx_ret_5d
    fx_ret_20d_series = 0.0 if is_tsx else fx_ret_20d

    return pd.DataFrame(
        {
            "date": idx,
            "ticker": t,
            "is_tsx": int(is_tsx),
            "last_close_cad": close_cad.values,
            "avg_dollar_volume_cad": avg_dollar_vol.values,
            "ret_5d": close_cad.pct_change(5, fill_method=None).values,
            "ret_10d": close_cad.pct_change(10, fill_method=None).values,
            "ret_20d": close_cad.pct_change(20, fill_method=None).values,
            "ret_60d": close_cad.pct_change(60, fill_method=None).values,
            "ret_120d": close_cad.pct_change(120, fill_method=None).values,
            "ret_accel_20_120": (close_cad.pct_change(20, fill_method=None) - close_cad.pct_change(120, fill_method=None)).values,
            "vol_20d_ann": vol_20.values,
            "vol_60d_ann": vol_60.values,
            "vol_ratio_20_60": vol_ratio_20_60.values,
            "rsi_14": rsi_14.values,
            "ma20_ratio": (close_cad / ma20 - 1.0).values,
            "ma50_ratio": (close_cad / ma50 - 1.0).values,
            "ma200_ratio": (close_cad / ma200 - 1.0).values,
            "drawdown_60d": drawdown_60d.values,
            "dist_52w_high": dist_52w_high.values,
            "dist_52w_low": dist_52w_low.values,
            "vol_anom_30d": vol_anom_30d.values,
            "momentum_reversal": momentum_reversal.values,
            "momentum_acceleration": momentum_acceleration.values,
            "ret_20d_lagged": ret_20d_lagged.values,
            "ret_60d_lagged": ret_60d_lagged.values,
            "ret_5d_sharpe": ret_5d_sharpe.values,
            "ret_20d_sharpe": ret_20d_sharpe.values,
            "ret_60d_sharpe": ret_60d_sharpe.values,
            "volume_momentum_20d": volume_momentum_20d.values,
            "volume_surge_5d": volume_surge_5d.values,
            "price_volume_div": price_volume_div.values,
            "ma20_zscore": ma20_zscore.values,
            "mean_reversion_signal": mean_reversion_signal.values,
            "ret_consistency_20d": ret_consistency_20d.values,
            "up_days_ratio_20d": up_days_ratio_20d.values,
            "log_market_cap": log_market_cap,
            "beta": beta,
            "sector": sector,
            "industry": industry,
            "sector_hash": sector_hash,
            "industry_hash": industry_hash,
            "fx_ret_5d": fx_ret_5d_series.values if hasattr(fx_ret_5d_series, "values") else fx_ret_5d_series,
            "fx_ret_20d": fx_ret_20d_series.values if hasattr(fx_ret_20d_series, "values") else fx_ret_20d_series,
            "n_days": n_days.values,
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
        }
    )


def _build_panel_features(
    prices: pd.DataFrame,
    fx_usdcad: pd.Series,
    fundamentals: pd.DataFrame | None,
) -> pd.DataFrame:
    # prices: columns (ticker, field). We compute rolling features per ticker per date.
    idx = pd.to_datetime(prices.index).sort_values()
    fx = fx_usdcad.copy()
    fx.index = pd.to_datetime(fx.index)
    fx = fx.reindex(idx).ffill()
    fx_ret_5d = fx.pct_change(5, fill_method=None)
    fx_ret_20d = fx.pct_change(20, fill_method=None)

    # Prepare ticker data for parallel processing
    tickers = [t for t in prices.columns.levels[0] if (t, "Close") in prices.columns]
    
    # PARALLEL FEATURE COMPUTATION: Use joblib if available for ~2-4x speedup
    n_jobs = min(os.cpu_count() or 1, 4)  # Limit to 4 workers to manage memory
    
    if HAS_JOBLIB and len(tickers) > 100:
        # Use threading backend (better for pandas which releases GIL)
        def _process_ticker(t):
            close = prices[(t, "Close")].astype(float)
            vol_series = prices[(t, "Volume")].astype(float) if (t, "Volume") in prices.columns else pd.Series(index=idx, dtype=float)
            return _compute_ticker_features(t, close, vol_series, idx, fx, fx_ret_5d, fx_ret_20d, fundamentals)
        
        frames = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(_process_ticker)(t) for t in tickers
        )
        frames = [f for f in frames if f is not None]
    else:
        # Sequential fallback
        frames: list[pd.DataFrame] = []
        for t in tickers:
            close = prices[(t, "Close")].astype(float)
            vol_series = prices[(t, "Volume")].astype(float) if (t, "Volume") in prices.columns else pd.Series(index=idx, dtype=float)
            df = _compute_ticker_features(t, close, vol_series, idx, fx, fx_ret_5d, fx_ret_20d, fundamentals)
            if df is not None:
                frames.append(df)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    del frames  # Free memory immediately
    gc.collect()
    
    if not panel.empty:
        # MEMORY OPTIMIZATION: Convert float64 to float32 (50% RAM reduction)
        float_cols = panel.select_dtypes(include=['float64']).columns
        for col in float_cols:
            panel[col] = panel[col].astype('float32')
        gc.collect()
        # Cross-sectional ranks - captures relative position within each date
        rank_map = {
            "ret_20d": "rank_ret_20d",
            "ret_60d": "rank_ret_60d",
            "vol_60d_ann": "rank_vol_60d",
            "avg_dollar_volume_cad": "rank_avg_dollar_volume",
            # HIGH-IMPACT: Additional cross-sectional ranks
            "ret_5d": "rank_ret_5d",
            "ret_5d_sharpe": "rank_ret_5d_sharpe",
            "momentum_reversal": "rank_momentum_reversal",
            "ma20_zscore": "rank_ma20_zscore",
        }
        for col, out_col in rank_map.items():
            if col in panel.columns:
                panel[out_col] = panel.groupby("date")[col].rank(pct=True)
        
        # HIGH-IMPACT: Momentum strength (composite rank)
        if "rank_ret_20d" in panel.columns and "rank_ret_60d" in panel.columns:
            panel["momentum_strength"] = (panel["rank_ret_20d"] + panel["rank_ret_60d"]) / 2
        
        # SECTOR-RELATIVE FEATURES: Rank within sector to remove sector bias
        # This captures whether a stock is strong/weak relative to its sector peers
        if "sector" in panel.columns:
            sector_rank_features = ["ret_20d", "ret_60d", "ret_5d_sharpe", "vol_60d_ann", "momentum_reversal"]
            for col in sector_rank_features:
                if col in panel.columns:
                    out_col = f"sector_rank_{col}"
                    # Rank within (date, sector) group
                    panel[out_col] = panel.groupby(["date", "sector"])[col].rank(pct=True)
        
        panel = add_fundamental_composites(panel, date_col="date")
    
    return panel


def train_and_save(cfg: Config, logger) -> TrainResult:
    """Train an ensemble of ML models and write manifest to cfg.model_path."""
    
    # Choose feature set based on config
    # Use technical features only to avoid fundamental data lookahead bias
    feature_cols = TECHNICAL_FEATURES_ONLY if not cfg.use_fundamentals_in_training else FEATURE_COLUMNS
    logger.info(
        "Using %s features for training (fundamentals=%s)",
        len(feature_cols),
        cfg.use_fundamentals_in_training,
    )

    data_cache_dir = ensure_dir(cfg.data_cache_dir)

    us = fetch_us_universe(cfg=cfg, cache_dir=Path(cfg.cache_dir), logger=logger)
    tsx = fetch_tsx_universe(cfg=cfg, cache_dir=Path(cfg.cache_dir), logger=logger)
    tickers = list(dict.fromkeys(us.tickers + tsx.tickers))
    if cfg.max_total_tickers is not None:
        tickers = tickers[: cfg.max_total_tickers]

    fx = fetch_usdcad(
        fx_ticker=cfg.fx_ticker,
        lookback_days=max(cfg.feature_lookback_days, cfg.liquidity_lookback_days) + 365,
        cache_dir=Path(cfg.data_cache_dir),
        logger=logger,
    )
    prices = download_price_history(
        tickers=tickers,
        lookback_days=cfg.feature_lookback_days + 365,
        threads=cfg.yfinance_threads,
        batch_size=cfg.batch_size,
        logger=logger,
    )

    fundamentals = fetch_fundamentals(
        tickers=tickers,
        cache_dir=data_cache_dir,
        cache_ttl_days=cfg.fundamentals_cache_ttl_days,
        logger=logger,
    )
    
    # Fetch macro indicators for regime detection
    macro = fetch_macro_indicators(
        lookback_days=cfg.feature_lookback_days + 365,
        logger=logger,
    )
    
    panel = _build_panel_features(prices=prices, fx_usdcad=fx, fundamentals=fundamentals)
    # Free source data immediately after panel is built
    del prices, fx, fundamentals
    gc.collect()
    
    if panel.empty:
        raise RuntimeError("No training panel built")
    
    # Merge macro indicators as date-level features
    if not macro.empty:
        macro_reset = macro.reset_index()
        macro_reset["date"] = pd.to_datetime(macro_reset["date"]).dt.normalize()
        panel = panel.merge(macro_reset, on="date", how="left")
        del macro, macro_reset
        gc.collect()
        logger.info(f"Merged macro indicators: {list(panel.columns[-4:])}")
    
    horizon = int(cfg.label_horizon_days)
    max_horizon = int(getattr(cfg, "max_holding_days_hard", 10))  # Extended horizon for peak detection
    top_n = max(1, int(cfg.portfolio_size))
    cost_bps = float(getattr(cfg, "trade_cost_bps", 0.0))
    # Shift by (horizon + 1) so predictions made with today's features predict tomorrow onward
    # This ensures we can act on predictions before market open
    panel["future_ret"] = panel.groupby("ticker")["last_close_cad"].shift(-(horizon + 1)) / panel["last_close_cad"] - 1.0
    
    # PEAK DETECTION: Find the day of maximum price within the extended horizon
    # Uses vectorized rolling operations for speed (no Python loops)
    def _compute_peak_targets_vectorized(group):
        """Vectorized peak detection - find day and return of max price in next N days."""
        prices = group["last_close_cad"].values
        n = len(prices)
        
        # Pre-allocate output arrays
        peak_days = np.full(n, np.nan)
        peak_returns = np.full(n, np.nan)
        
        # Build matrix of future prices: each row i contains prices[i+1:i+1+max_horizon]
        # Use stride tricks for efficiency
        if n <= max_horizon:
            return pd.DataFrame({"days_to_peak": peak_days, "peak_return": peak_returns}, index=group.index)
        
        # Create shifted views for each day in horizon
        # shifted[d] = prices shifted by -(d+1), i.e., future price on day d+1
        future_matrix = np.full((n, max_horizon), np.nan)
        for d in range(max_horizon):
            shift = d + 1
            if shift < n:
                future_matrix[:-shift, d] = prices[shift:]
        
        # Mark rows with all-NaN future FIRST
        all_nan_mask = np.all(np.isnan(future_matrix), axis=1)
        
        # For rows with all NaN, temporarily fill with 0 to avoid nanargmax error
        temp_matrix = future_matrix.copy()
        temp_matrix[all_nan_mask, 0] = 0.0
        
        # Find argmax and max across the horizon dimension
        peak_idx = np.nanargmax(temp_matrix, axis=1)
        with np.errstate(all='ignore'):  # Suppress "All-NaN slice" warning
            peak_price = np.nanmax(future_matrix, axis=1)
        
        # Convert to 1-indexed days (1 = tomorrow)
        peak_days = (peak_idx + 1).astype(float)
        peak_days[all_nan_mask] = np.nan
        
        # Compute peak returns
        valid_price_mask = prices > 0
        peak_returns = np.where(valid_price_mask, peak_price / prices - 1.0, np.nan)
        peak_returns[all_nan_mask] = np.nan
        
        return pd.DataFrame({"days_to_peak": peak_days, "peak_return": peak_returns}, index=group.index)
    
    # Apply vectorized function to each ticker group
    peak_df = panel.groupby("ticker", group_keys=False).apply(_compute_peak_targets_vectorized)
    panel["days_to_peak"] = peak_df["days_to_peak"]
    panel["peak_return"] = peak_df["peak_return"]
    logger.info("Computed peak detection targets: days_to_peak (1-%d), peak_return", max_horizon)
    del peak_df
    gc.collect()
    
    # Compute market benchmark return for each date using MARKET-CAP WEIGHTING
    # This is more realistic than equal-weighted (approximates SPY/TSX benchmark)
    def _compute_cap_weighted_return(group):
        """Compute market-cap weighted return for a date."""
        if "log_market_cap" not in group.columns:
            return group["future_ret"].mean()  # Fallback to equal-weighted
        
        # Convert log10(market_cap) back to market cap
        mcap = np.power(10, group["log_market_cap"].fillna(group["log_market_cap"].median()))
        valid_mask = mcap.notna() & group["future_ret"].notna()
        
        if valid_mask.sum() < 2:
            return group["future_ret"].mean()
        
        mcap_valid = mcap[valid_mask]
        ret_valid = group.loc[valid_mask, "future_ret"]
        weights = mcap_valid / mcap_valid.sum()
        return (ret_valid * weights).sum()
    
    market_returns = panel.groupby("date").apply(_compute_cap_weighted_return)
    panel["market_ret"] = panel["date"].map(market_returns)
    logger.info("Using cap-weighted market benchmark (approximates SPY/TSX)")
    
    # Compute RELATIVE MOMENTUM (stock vs market) - VECTORIZED for speed
    # Pre-compute market cap weights once (avoid repeated computation)
    _mcap = np.power(10, panel["log_market_cap"].fillna(9).values)
    
    def _fast_weighted_mean(val_col, weights, dates):
        """Memory-efficient weighted mean per date - no DataFrame copy."""
        vals = panel[val_col].fillna(0).values
        weighted_vals = vals * weights
        # Use pandas groupby on Series (not DataFrame) to minimize memory
        date_series = pd.Series(dates)
        weighted_sum = pd.Series(weighted_vals).groupby(date_series).sum()
        weight_sum = pd.Series(weights).groupby(date_series).sum().replace(0, np.nan)
        wmean = weighted_sum / weight_sum
        return date_series.map(wmean).values
    
    _dates = panel["date"].values
    market_ret_20d = _fast_weighted_mean("ret_20d", _mcap, _dates)
    market_ret_60d = _fast_weighted_mean("ret_60d", _mcap, _dates)
    panel["relative_momentum_20d"] = panel["ret_20d"] - market_ret_20d
    panel["relative_momentum_60d"] = panel["ret_60d"] - market_ret_60d
    logger.info("Added relative momentum features (stock vs cap-weighted market)")
    
    # Compute MARKET REGIME features (same value for all stocks on each date)
    # Market volatility regime: cap-weighted average volatility vs historical norm
    if "vol_20d_ann" in panel.columns:
        market_vol = _fast_weighted_mean("vol_20d_ann", _mcap, _dates)
        historical_norm = 0.15  # ~15% annualized is "normal" volatility
        panel["market_vol_regime"] = market_vol / historical_norm
    else:
        panel["market_vol_regime"] = 1.0
    
    # Market trend: cap-weighted 20-day return per date
    panel["market_trend_20d"] = market_ret_20d
    
    # Market breadth: % of stocks above their 20-day MA on each date (fast)
    if "ma20_ratio" in panel.columns:
        above_ma = (panel["ma20_ratio"] > 1.0).astype(int).values
        date_series = pd.Series(_dates)
        breadth_count = date_series.groupby(date_series).count()
        above_sum = pd.Series(above_ma).groupby(date_series).sum()
        breadth = (above_sum / breadth_count).fillna(0.5)
        panel["market_breadth"] = date_series.map(breadth).values
    else:
        panel["market_breadth"] = 0.5
    
    # Market momentum acceleration: short-term vs medium-term momentum
    if "ret_5d" in panel.columns:
        market_ret_5d = _fast_weighted_mean("ret_5d", _mcap, _dates)
        panel["market_momentum_accel"] = (market_ret_5d * 4) - market_ret_20d
    else:
        panel["market_momentum_accel"] = 0.0
    
    # Clean up temp arrays and free memory
    del _mcap, _dates
    gc.collect()
    logger.info("Added market regime features (vol_regime, trend, breadth, momentum_accel)")
    
    # HIGH-IMPACT: Feature interaction terms
    # These capture non-linear relationships that boost trees might miss
    if "ret_20d_sharpe" in panel.columns and "momentum_strength" in panel.columns:
        panel["sharpe_x_rank"] = panel["ret_20d_sharpe"] * panel["momentum_strength"]
    if "ret_5d" in panel.columns and "vol_20d_ann" in panel.columns:
        panel["momentum_vol_interaction"] = panel["ret_5d"] * panel["vol_20d_ann"]
    if "rsi_14" in panel.columns and "ret_5d" in panel.columns:
        # RSI extremes with momentum direction
        panel["rsi_momentum_interaction"] = (panel["rsi_14"] - 50) / 50 * panel["ret_5d"]
    if "log_market_cap" in panel.columns and "relative_momentum_20d" in panel.columns:
        panel["size_momentum_interaction"] = panel["log_market_cap"] * panel["relative_momentum_20d"]
    if "ma20_zscore" in panel.columns and "ret_5d" in panel.columns:
        # Mean reversion potential
        panel["zscore_reversal"] = -panel["ma20_zscore"] * np.sign(panel["ret_5d"])
    logger.info("Added feature interaction terms")
    
    # Compute market-relative returns (alpha = stock return - market return)
    panel["future_alpha"] = panel["future_ret"] - panel["market_ret"]
    
    # Configure which target to use for training
    use_market_relative = getattr(cfg, 'use_market_relative_returns', True)
    if use_market_relative:
        target_col = "future_alpha"
        logger.info("Training on market-relative returns (alpha). Market avg will be subtracted.")
    else:
        target_col = "future_ret"
        logger.info("Training on absolute returns.")

    # Apply TARGET ENCODING for sector/industry (replaces useless hash encoding)
    # This encodes sectors by their historical mean return, giving the model meaningful signals
    if "sector" in panel.columns and panel["sector"].notna().any():
        logger.info("Applying target encoding for sector...")
        panel["sector_target_enc"] = _apply_target_encoding(
            panel, cat_col="sector", target_col=target_col, date_col="date", smoothing=20.0
        )
        n_sectors = panel["sector"].nunique()
        logger.info("Target-encoded %d unique sectors", n_sectors)
    
    if "industry" in panel.columns and panel["industry"].notna().any():
        logger.info("Applying target encoding for industry...")
        panel["industry_target_enc"] = _apply_target_encoding(
            panel, cat_col="industry", target_col=target_col, date_col="date", smoothing=20.0
        )
        n_industries = panel["industry"].nunique()
        logger.info("Target-encoded %d unique industries", n_industries)

    # Drop rows without labels/features
    panel = panel.dropna(subset=[target_col])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["last_close_cad", "avg_dollar_volume_cad"])
    panel = panel[panel["n_days"] >= 90]
    panel = panel[panel["last_close_cad"] >= float(cfg.min_price_cad)]
    panel = panel[panel["avg_dollar_volume_cad"] >= float(cfg.min_avg_dollar_volume_cad)]

    # CRITICAL: Cross-sectional winsorization and normalization
    # Financial returns are relative games - normalize within each date
    logger.info("Applying cross-sectional winsorization (MAD-based) and normalization...")
    panel = normalize_features_cross_section(panel, date_col="date")

    # Also winsorize labels to remove extreme outliers (MAD-based)
    panel["future_ret"] = panel.groupby("date")["future_ret"].transform(winsorize_mad)
    if "future_alpha" in panel.columns:
        panel["future_alpha"] = panel.groupby("date")["future_alpha"].transform(winsorize_mad)
    logger.info("Winsorized target labels to reduce outlier impact")
    
    # Validate that required features exist
    missing_features = [c for c in feature_cols if c not in panel.columns]
    if missing_features:
        logger.error("Missing %d required features: %s", len(missing_features), missing_features[:20])
        raise ValueError(f"Missing required features in training data: {missing_features[:10]}")
    
    # Check for features with all NaN values
    nan_features = [c for c in feature_cols if panel[c].isna().all()]
    if nan_features:
        logger.warning("Features with all NaN values: %s", nan_features)


    dates = pd.to_datetime(panel["date"]).dt.normalize().unique().tolist()
    splits, holdout = build_time_splits(
        dates, 
        n_splits=cfg.train_cv_splits,
        val_window=cfg.train_val_window_days,
        embargo_days=cfg.train_embargo_days,
    )
    val_dates = splits[-1].val_dates if splits else []

    def _subset_by_dates(dates_list: list[pd.Timestamp]) -> pd.DataFrame:
        if not dates_list:
            return panel.iloc[0:0].copy()
        mask = panel["date"].isin(dates_list)
        return panel.loc[mask]

    # Use configured target column for all training and evaluation
    label_col = target_col

    def _rank_ic_for_preds(df: pd.DataFrame, pred: pd.Series) -> dict[str, object]:
        if df.empty:
            return {"summary": {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}}
        temp = df.copy()
        temp["pred"] = pred
        return evaluate_predictions(temp, date_col="date", label_col=label_col, pred_col="pred", group_col="is_tsx")

    def _topn_for_preds(df: pd.DataFrame, pred: pd.Series) -> dict[str, object]:
        if df.empty:
            return {
                "summary": {
                    "mean_ret": float("nan"),
                    "std_ret": float("nan"),
                    "ret_ir": float("nan"),
                    "mean_net_ret": float("nan"),
                    "std_net_ret": float("nan"),
                    "net_ret_ir": float("nan"),
                    "n_days": 0,
                }
            }
        temp = df.copy()
        temp["pred"] = pred
        # CRITICAL: Always use future_ret for portfolio returns, even when training on alpha
        # The model predicts alpha (relative returns), but portfolio P&L uses absolute returns
        # Also pass market_ret and beta for alpha calculation
        return evaluate_topn_returns(
            temp, date_col="date", label_col="future_ret", pred_col="pred", 
            top_n=top_n, cost_bps=cost_bps,
            market_ret_col="market_ret" if "market_ret" in temp.columns else None,
            beta_col="beta" if "beta" in temp.columns else None,
        )


    # Hyperparameter tuning with Optuna (if available) or fallback to manual search
    regressor_scores = {}  # Initialize to avoid UnboundLocalError
    
    if optuna and getattr(cfg, 'use_optuna', False):
        logger.info("Using Optuna for hyperparameter optimization...")
        n_trials = getattr(cfg, 'optuna_n_trials', 12)
        timeout = getattr(cfg, 'optuna_timeout_seconds', 180)
        
        def _optuna_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
            }
            scores: list[float] = []
            for split in splits:
                train_df = _subset_by_dates(split.train_dates)
                val_df = _subset_by_dates(split.val_dates)
                if train_df.empty or val_df.empty:
                    continue
                model = build_model(random_state=42)
                model.set_params(**params, early_stopping_rounds=50)
                model.fit(
                    train_df[feature_cols],
                    train_df[label_col].astype(float),
                    eval_set=[(val_df[feature_cols], val_df[label_col].astype(float))],
                    verbose=False,
                )
                preds = model.predict(val_df[feature_cols])
                metrics = _topn_for_preds(val_df, pd.Series(preds, index=val_df.index))
                scores.append(float(metrics["summary"]["mean_net_ret"]))
            return float(np.nanmean(scores)) if scores else float("nan")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(_optuna_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best_regressor_params = study.best_params
        
        # Store all trial results in consistent format with manual search
        # Format: {str(params): score} for consistency
        regressor_scores = {}
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params_str = str(trial.params)
                regressor_scores[params_str] = trial.value
        
        logger.info("Optuna best params: %s (score=%.6f)", best_regressor_params, study.best_value)
        logger.info("Optuna completed %d trials", len(study.trials))
    else:
        # Fallback to manual hyperparameter search
        logger.info("Using manual hyperparameter search...")
        regressor_candidates = [
            {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 5},
            {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 8},
        ]

        def _eval_regressor_params(params: dict[str, float]) -> float:
            scores: list[float] = []
            for split in splits:
                train_df = _subset_by_dates(split.train_dates)
                val_df = _subset_by_dates(split.val_dates)
                if train_df.empty or val_df.empty:
                    continue
                model = build_model(random_state=42)
                model.set_params(**params, early_stopping_rounds=50)
                model.fit(
                    train_df[feature_cols],
                    train_df[label_col].astype(float),
                    eval_set=[(val_df[feature_cols], val_df[label_col].astype(float))],
                    verbose=False,
                )
                preds = model.predict(val_df[feature_cols])
                metrics = _topn_for_preds(val_df, pd.Series(preds, index=val_df.index))
                scores.append(float(metrics["summary"]["mean_net_ret"]))
            return float(np.nanmean(scores)) if scores else float("nan")

        regressor_scores = {str(p): _eval_regressor_params(p) for p in regressor_candidates}
        best_regressor_params = max(regressor_candidates, key=lambda p: regressor_scores.get(str(p), float("-inf")))

    train_df = _subset_by_dates(holdout.train_dates)
    val_df = _subset_by_dates(val_dates)
    holdout_df = _subset_by_dates(holdout.holdout_dates)
    
    # Note: panel is kept for walk-forward validation and metadata
    gc.collect()

    # Compute sample weights: more recent samples get higher weight
    # This helps the model adapt to recent market conditions
    train_dates_sorted = sorted(train_df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(train_dates_sorted)}
    train_df = train_df.copy()
    train_df["_date_idx"] = train_df["date"].map(date_to_idx)
    # Exponential decay weight: recent data weighted ~2x more than oldest
    decay_rate = 1.0 / len(train_dates_sorted)  # ~2x weight for newest vs oldest
    train_df["_sample_weight"] = np.exp(decay_rate * train_df["_date_idx"])
    # Normalize weights to mean=1
    train_df["_sample_weight"] = train_df["_sample_weight"] / train_df["_sample_weight"].mean()
    sample_weights = train_df["_sample_weight"].values
    logger.info("Added recency sample weights: oldest=%.2f, newest=%.2f", sample_weights.min(), sample_weights.max())

    # Use configured ensemble composition or default
    n_xgb = getattr(cfg, 'ensemble_xgb_count', 3)
    n_lgbm = getattr(cfg, 'ensemble_lgbm_count', 3)
    use_lgbm = getattr(cfg, 'use_lightgbm', True)
    use_ranking = getattr(cfg, 'use_ranking_objective', True)
    
    if not use_lgbm:
        n_xgb = 7  # Fallback to 7 XGBoost models
        n_lgbm = 0
    
    model_dir = Path(cfg.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    reg_rel_paths: list[str] = []
    model_types: list[str] = []

    # Prepare group data for LTR (Learning-to-Rank)
    # XGBRanker needs group sizes (number of samples per query/date)
    train_groups = None
    val_groups = None
    train_rank_labels = None
    val_rank_labels = None
    if use_ranking:
        train_df = train_df.sort_values("date")  # Sort by date for grouping
        train_groups = train_df.groupby("date").size().values
        
        # Convert continuous returns to integer relevance grades (0-4)
        # XGBRanker requires non-negative integer labels
        def returns_to_grades(returns: pd.Series, n_grades: int = 5) -> pd.Series:
            """Convert returns to integer relevance grades per cross-section."""
            # Use quantiles to assign grades 0-4 (0=worst, 4=best)
            grades = pd.qcut(returns.rank(method="first"), q=n_grades, labels=False, duplicates="drop")
            return grades.fillna(0).astype(int)
        
        # Apply per-date to maintain cross-sectional ranking
        train_rank_labels = train_df.groupby("date")[label_col].transform(
            lambda x: returns_to_grades(x, n_grades=5)
        ).astype(int)
        
        if not val_df.empty:
            val_df = val_df.sort_values("date")
            val_groups = val_df.groupby("date").size().values
            val_rank_labels = val_df.groupby("date")[label_col].transform(
                lambda x: returns_to_grades(x, n_grades=5)
            ).astype(int)
        
        logger.info("Using LTR objective with %d training groups (dates), labels as grades 0-4", len(train_groups))
    
    objective_type = "LTR (rank)" if use_ranking else "regression"
    
    # PRE-TRAINING FEATURE IC SELECTION: Drop features with negative IC on validation data
    # This removes features that are inversely correlated with future returns
    pre_training_ic = {}
    if not val_df.empty and len(feature_cols) > 10:
        from scipy.stats import spearmanr
        for col in feature_cols:
            if col in val_df.columns and val_df[col].notna().sum() > 100:
                try:
                    mask = val_df[col].notna() & val_df[label_col].notna()
                    if mask.sum() > 100:
                        ic, _ = spearmanr(val_df.loc[mask, col], val_df.loc[mask, label_col])
                        if not np.isnan(ic):
                            pre_training_ic[col] = float(ic)
                except Exception:
                    pass
        
        # Keep only features with non-negative IC (neutral or helpful)
        positive_ic_features = [f for f in feature_cols if pre_training_ic.get(f, 0.0) >= -0.01]
        negative_ic_features = [f for f in feature_cols if pre_training_ic.get(f, 0.0) < -0.01]
        
        # Always keep at least 50% of features
        min_features = max(15, len(feature_cols) // 2)
        if len(positive_ic_features) >= min_features:
            logger.info(
                "IC-based feature selection: keeping %d/%d features (dropped %d with IC < -0.01)",
                len(positive_ic_features), len(feature_cols), len(negative_ic_features)
            )
            if negative_ic_features:
                # Log features with most negative IC
                worst_features = sorted([(f, pre_training_ic.get(f, 0)) for f in negative_ic_features], key=lambda x: x[1])[:5]
                logger.info("Dropped features (worst IC): %s", [(f, f"{ic:.3f}") for f, ic in worst_features])
            feature_cols = positive_ic_features
    
    logger.info(
        "Training mixed ensemble: %d XGBoost + %d LightGBM models on %s samples, %s tickers (%s)",
        n_xgb,
        n_lgbm,
        len(train_df),
        train_df["ticker"].nunique(),
        objective_type,
    )
    
    # Track selected features for metadata
    selected_features = feature_cols.copy()
    dropped_features = []
    
    # Train XGBoost models (with ranking or regression objective)
    for i in range(n_xgb):
        seed = 42 + i * 10
        
        # Use selected_features (may be filtered after first model)
        current_features = selected_features
        
        if use_ranking:
            # Use XGBRanker for learning-to-rank
            m = xgb.XGBRanker(
                n_estimators=400,  # Balanced: quality vs speed
                learning_rate=best_regressor_params.get("learning_rate", 0.04),
                max_depth=best_regressor_params.get("max_depth", 5),
                subsample=0.75,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                min_child_weight=best_regressor_params.get("min_child_weight", 8),
                objective="rank:pairwise",
                eval_metric="ndcg@30",
                n_jobs=0,
                random_state=seed,
            )
            # Use integer rank labels for XGBRanker
            if not val_df.empty and val_groups is not None and val_rank_labels is not None:
                m.fit(
                    train_df[current_features],
                    train_rank_labels,
                    group=train_groups,
                    eval_set=[(val_df[current_features], val_rank_labels)],
                    eval_group=[val_groups],
                    verbose=False,
                )
            else:
                m.fit(train_df[current_features], train_rank_labels, group=train_groups)
        else:
            # Use XGBRegressor for regression
            m = build_model(random_state=seed)
            m.set_params(**best_regressor_params, early_stopping_rounds=50)
            if not val_df.empty:
                m.fit(
                    train_df[current_features],
                    train_df[label_col].astype(float),
                    sample_weight=sample_weights,
                    eval_set=[(val_df[current_features], val_df[label_col].astype(float))],
                    verbose=False,
                )
            else:
                m.fit(
                    train_df[current_features], 
                    train_df[label_col].astype(float),
                    sample_weight=sample_weights
                )
        
        # After first model: perform feature selection based on importance
        if i == 0 and len(feature_cols) > 10:
            try:
                # Get feature importances (gain-based)
                importance = m.get_booster().get_score(importance_type="gain")
                if importance:
                    # Map feature names (fX -> actual name)
                    feature_name_map = {f"f{j}": fname for j, fname in enumerate(current_features)}
                    importance_named = {feature_name_map.get(k, k): v for k, v in importance.items()}
                    
                    # Sort by importance
                    sorted_imp = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)
                    total_importance = sum(v for _, v in sorted_imp)
                    
                    if total_importance > 0:
                        # Keep features that account for top 90% of cumulative importance
                        cumsum = 0.0
                        keep_features = []
                        threshold = 0.90  # Keep features up to 90% cumulative importance
                        
                        for fname, imp in sorted_imp:
                            cumsum += imp / total_importance
                            keep_features.append(fname)
                            if cumsum >= threshold:
                                break
                        
                        # Always keep at least 10 features
                        min_features = min(10, len(feature_cols))
                        if len(keep_features) < min_features:
                            keep_features = [f for f, _ in sorted_imp[:min_features]]
                        
                        # Identify dropped features
                        dropped_features = [f for f in feature_cols if f not in keep_features]
                        
                        if dropped_features:
                            logger.info(
                                "Feature selection: keeping %d/%d features (%.1f%% importance)",
                                len(keep_features), len(feature_cols), threshold * 100
                            )
                            logger.info("Dropped features: %s", dropped_features[:10])  # Log first 10
                            selected_features = keep_features
            except Exception as e:
                logger.warning("Feature selection failed, using all features: %s", e)
        
        rel = f"xgb_model_{i}.json"
        save_model(m, model_dir / rel)
        reg_rel_paths.append(rel)
        model_types.append("xgboost")
        logger.info("Trained XGBoost %s model %d/%d", "ranker" if use_ranking else "regressor", i+1, n_xgb)
    
    # Train LightGBM models if enabled (using selected features)
    # Use QUANTILE TRANSFORMATION: convert continuous target to cross-sectional percentile rank
    # This is more robust to outliers and focuses on relative ranking
    if use_lgbm and n_lgbm > 0:
        # Transform target to cross-sectional percentile rank (0-1 scale)
        train_target_quantile = train_df.groupby("date")[label_col].rank(pct=True).astype(float)
        val_target_quantile = val_df.groupby("date")[label_col].rank(pct=True).astype(float) if not val_df.empty else None
        logger.info("Using quantile-transformed target for LightGBM (more robust to outliers)")
        
        for i in range(n_lgbm):
            seed = 42 + i * 10
            m = build_lgbm_model(random_state=seed)
            # Translate XGBoost params to LightGBM equivalents
            lgbm_params = {
                "max_depth": best_regressor_params.get("max_depth", 6),
                "learning_rate": best_regressor_params.get("learning_rate", 0.05),
                "min_child_samples": best_regressor_params.get("min_child_weight", 5) * 4,  # Approximate conversion
                "subsample": best_regressor_params.get("subsample", 0.8),
                "colsample_bytree": best_regressor_params.get("colsample_bytree", 0.8),
                "reg_lambda": best_regressor_params.get("reg_lambda", 1.0),
            }
            m.set_params(**lgbm_params)
            if not val_df.empty and val_target_quantile is not None:
                m.fit(
                    train_df[selected_features],
                    train_target_quantile,  # Use quantile-transformed target
                    sample_weight=sample_weights,
                    eval_set=[(val_df[selected_features], val_target_quantile)],
                    callbacks=[],  # Disable callbacks to reduce verbosity
                )
            else:
                m.fit(
                    train_df[selected_features], 
                    train_target_quantile,  # Use quantile-transformed target
                    sample_weight=sample_weights
                )
            rel = f"lgbm_model_{i}.txt"
            save_model(m, model_dir / rel)
            reg_rel_paths.append(rel)
            model_types.append("lightgbm")
            logger.info("Trained LightGBM model %d/%d", i+1, n_lgbm)

    reg_holdout_preds: list[float] = []
    model_ics: list[float] = []  # Track IC per model for adaptive weighting
    
    if not holdout_df.empty:
        # First pass: compute IC for each model to determine weights
        for rel, mtype in zip(reg_rel_paths, model_types):
            model = load_model(model_dir / rel, mtype)
            if mtype == "xgboost":
                model_pred = model.predict(xgb.DMatrix(holdout_df[selected_features])).astype(float)
            else:  # lightgbm
                model_pred = model.predict(holdout_df[selected_features]).astype(float)
            
            # Compute IC for this model
            model_metrics = _rank_ic_for_preds(holdout_df, pd.Series(model_pred, index=holdout_df.index))
            model_ic = model_metrics.get("summary", {}).get("mean_ic", 0.0)
            model_ics.append(max(model_ic, 0.0))  # Floor at 0 (don't reward negative IC)
            del model_pred, model
        
        gc.collect()
        
        # Compute weights from ICs
        if sum(model_ics) > 0:
            weights = np.array(model_ics) / sum(model_ics)
            logger.info(f"IC-weighted ensemble: model ICs={[f'{ic:.4f}' for ic in model_ics]}, weights={[f'{w:.3f}' for w in weights]}")
        else:
            weights = np.ones(len(model_ics)) / len(model_ics)
            logger.warning("All model ICs <= 0, using equal weighting")
        
        # Second pass: compute weighted predictions (memory efficient)
        # IMPORTANT: Standardize predictions before combining to handle scale mismatch
        # XGBRanker outputs ranking scores (~0-10) while LightGBM outputs alpha (~-0.5 to +0.5)
        preds = np.zeros(len(holdout_df), dtype=float)
        for i, (rel, mtype) in enumerate(zip(reg_rel_paths, model_types)):
            model = load_model(model_dir / rel, mtype)
            if mtype == "xgboost":
                model_pred = model.predict(xgb.DMatrix(holdout_df[selected_features])).astype(float)
            else:  # lightgbm
                model_pred = model.predict(holdout_df[selected_features]).astype(float)
            
            # Standardize to z-scores before combining (handles scale mismatch)
            pred_std = float(np.nanstd(model_pred))
            if pred_std > 0:
                model_pred = (model_pred - np.nanmean(model_pred)) / pred_std
            
            preds += model_pred * weights[i]
            del model_pred, model
        
        reg_holdout_preds = preds.tolist()
        gc.collect()

    reg_holdout_metrics = _rank_ic_for_preds(holdout_df, pd.Series(reg_holdout_preds, index=holdout_df.index))
    reg_holdout_topn = _topn_for_preds(holdout_df, pd.Series(reg_holdout_preds, index=holdout_df.index))

    # Compute train IC to compare with holdout IC (detect overfitting)
    # Use same IC-weighted ensemble with standardization as holdout
    train_preds = np.zeros(len(train_df), dtype=float)
    for i, (rel, mtype) in enumerate(zip(reg_rel_paths, model_types)):
        model = load_model(model_dir / rel, mtype)
        if mtype == "xgboost":
            model_pred = model.predict(xgb.DMatrix(train_df[selected_features])).astype(float)
        else:  # lightgbm
            model_pred = model.predict(train_df[selected_features]).astype(float)
        # Standardize before combining (same as holdout)
        pred_std = float(np.nanstd(model_pred))
        if pred_std > 0:
            model_pred = (model_pred - np.nanmean(model_pred)) / pred_std
        train_preds += model_pred * weights[i]
        del model_pred, model
    train_metrics = _rank_ic_for_preds(train_df, pd.Series(train_preds, index=train_df.index))
    
    # Log train vs holdout IC to detect overfitting
    train_ic = train_metrics.get("summary", {}).get("mean_ic", 0.0)
    holdout_ic = reg_holdout_metrics.get("summary", {}).get("mean_ic", 0.0)
    logger.info(
        "Train IC: %.4f, Holdout IC: %.4f (ratio: %.2f) - %s",
        train_ic, holdout_ic,
        holdout_ic / train_ic if train_ic != 0 else 0,
        "OVERFITTING" if train_ic > 0.1 and holdout_ic < train_ic * 0.5 else "OK"
    )

    # Load all trained models for walk-forward validation
    reg_models = []
    for rel, mtype in zip(reg_rel_paths, model_types):
        reg_models.append(load_model(model_dir / rel, mtype))

    # =========================================================================
    # PEAK TIMING MODEL: Predict days_to_peak (1 to max_horizon)
    # =========================================================================
    peak_model_path = None
    peak_metrics = {}
    
    # Filter rows with valid days_to_peak
    peak_train = train_df[train_df["days_to_peak"].notna()].copy()
    peak_holdout = holdout_df[holdout_df["days_to_peak"].notna()].copy() if not holdout_df.empty else pd.DataFrame()
    
    if len(peak_train) > 1000 and lgb is not None:
        logger.info("Training peak timing model on %d samples...", len(peak_train))
        
        # Use LightGBM for peak prediction (regression on days 1-10)
        peak_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=3.0,
            min_child_samples=50,
            random_state=42,
            verbose=-1,
        )
        
        peak_model.fit(
            peak_train[selected_features], 
            peak_train["days_to_peak"].astype(float)
        )
        
        # Save peak model
        peak_rel = "peak_model.txt"
        save_model(peak_model, model_dir / peak_rel)
        peak_model_path = peak_rel
        logger.info("Trained peak timing model")
        
        # Evaluate on holdout
        if len(peak_holdout) > 100:
            peak_preds = peak_model.predict(peak_holdout[selected_features])
            actual_days = peak_holdout["days_to_peak"].values
            
            # Compute MAE (mean absolute error in days)
            mae = np.mean(np.abs(peak_preds - actual_days))
            
            # Compute correlation (do predictions correlate with actual peak days?)
            from scipy.stats import spearmanr
            corr, _ = spearmanr(peak_preds, actual_days)
            
            # Compute accuracy within N days
            within_1_day = np.mean(np.abs(peak_preds - actual_days) <= 1)
            within_2_days = np.mean(np.abs(peak_preds - actual_days) <= 2)
            
            peak_metrics = {
                "mae_days": float(mae),
                "correlation": float(corr) if not np.isnan(corr) else 0.0,
                "within_1_day": float(within_1_day),
                "within_2_days": float(within_2_days),
                "n_samples": len(peak_holdout),
            }
            logger.info(
                "Peak timing holdout: MAE=%.2f days, corr=%.3f, within_1d=%.1f%%, within_2d=%.1f%%",
                mae, corr if not np.isnan(corr) else 0, within_1_day * 100, within_2_days * 100
            )
    else:
        logger.info("Insufficient samples for peak timing model (%d < 1000)", len(peak_train))



    # Compute feature importance from first XGBoost model
    feature_importance = {}
    xgb_models = [rel for rel, mtype in zip(reg_rel_paths, model_types) if mtype == "xgboost"]
    if xgb_models:
        first_model = load_model(model_dir / xgb_models[0], "xgboost")
        # Use get_score() for xgb.Booster
        try:
            importance = first_model.get_score(importance_type="gain")
            feature_importance = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]}
            
            # Map feature codes (f0, f1) to actual names for better logging
            feature_name_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
            top_features_named = [f"{k} ({feature_name_map.get(k, k)})" for k in list(feature_importance.keys())[:10]]
            logger.info("Top 10 features by gain: %s", top_features_named)
        except Exception as e:
            logger.warning("Could not extract feature importance: %s", e)

    # Compute per-feature IC on holdout
    feature_ic = {}
    if not holdout_df.empty:
        for col in FEATURE_COLUMNS:
            if col in holdout_df.columns and holdout_df[col].notna().sum() > 10:
                try:
                    from scipy.stats import spearmanr
                    # Use only non-NaN values to avoid bias from filling with 0
                    mask = holdout_df[col].notna() & holdout_df[label_col].notna()
                    if mask.sum() > 10:
                        ic, _ = spearmanr(holdout_df.loc[mask, col], holdout_df.loc[mask, label_col])
                        feature_ic[col] = float(ic) if not np.isnan(ic) else 0.0
                except Exception:
                    pass
        top_features = sorted(feature_ic.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        logger.info("Top 10 features by |IC|: %s", [(k, f"{v:.3f}") for k, v in top_features])

    # Enhanced validation metrics: by sector and market cap
    validation_metrics = {}
    if not holdout_df.empty and len(reg_holdout_preds) > 0:
        holdout_with_preds = holdout_df.copy()
        holdout_with_preds["pred_return"] = reg_holdout_preds
        
        # IC by sector (if sector data available)
        if "sector_hash" in holdout_with_preds.columns:
            sector_ics = {}
            for sector in holdout_with_preds["sector_hash"].dropna().unique():
                sector_df = holdout_with_preds[holdout_with_preds["sector_hash"] == sector]
                if len(sector_df) > 20:
                    sector_metrics = evaluate_predictions(
                        sector_df, date_col="date", label_col=label_col, 
                        pred_col="pred_return", group_col=None
                    )
                    sector_ics[f"sector_{sector:.3f}"] = sector_metrics.get("summary", {})
            if sector_ics:
                validation_metrics["by_sector"] = sector_ics
                logger.info("IC by sector: %d sectors evaluated", len(sector_ics))
        
        # IC by market cap quintile
        if "log_market_cap" in holdout_with_preds.columns:
            mcap_ics = {}
            holdout_with_preds["mcap_quintile"] = pd.qcut(
                holdout_with_preds["log_market_cap"], q=5, labels=False, duplicates="drop"
            )
            for q in range(5):
                q_df = holdout_with_preds[holdout_with_preds["mcap_quintile"] == q]
                if len(q_df) > 20:
                    q_metrics = evaluate_predictions(
                        q_df, date_col="date", label_col=label_col,
                        pred_col="pred_return", group_col=None
                    )
                    mcap_ics[f"quintile_{q+1}"] = q_metrics.get("summary", {})
            if mcap_ics:
                validation_metrics["by_market_cap_quintile"] = mcap_ics
                logger.info("IC by market cap quintile: %d quintiles evaluated", len(mcap_ics))

    # Enhanced metrics including calibration and portfolio stats
    calibration_metrics = {}
    portfolio_metrics = {}
    
    if not holdout_df.empty and len(reg_holdout_preds) > 0:
        # Compute calibration against the TRAINED target (alpha or absolute)
        # This measures how well the model predicts what it was trained to predict
        calibration_result = compute_calibration(
            pd.Series(reg_holdout_preds, index=holdout_df.index),
            holdout_df[label_col],
            n_bins=10
        )
        calibration_metrics = {
            "calibration_error": calibration_result["calibration_error"],
            "n_deciles": len(calibration_result["by_decile"]) if isinstance(calibration_result["by_decile"], pd.DataFrame) else 0,
        }
        logger.info("Calibration error: %.6f", calibration_result["calibration_error"])
        
        # Compute portfolio metrics from top-N daily returns
        # NOTE: These returns are ALWAYS absolute returns (future_ret), not alpha
        # This is correct because portfolio P&L is measured in absolute terms
        if "daily" in reg_holdout_topn and isinstance(reg_holdout_topn["daily"], pd.DataFrame):
            daily_df = reg_holdout_topn["daily"]
            if "mean_ret" in daily_df.columns and len(daily_df) > 0:
                # Extract alpha columns if available
                daily_alpha = None
                daily_market = None
                if "beta_adj_alpha" in daily_df.columns:
                    daily_alpha = daily_df["beta_adj_alpha"]
                elif "simple_alpha" in daily_df.columns:
                    daily_alpha = daily_df["simple_alpha"]
                if "market_ret" in daily_df.columns:
                    daily_market = daily_df["market_ret"]
                
                portfolio_metrics = compute_portfolio_metrics(
                    daily_df["mean_ret"],
                    daily_alpha=daily_alpha,
                    daily_market=daily_market,
                )
                
                # Log both raw and alpha-adjusted metrics
                alpha_ann = portfolio_metrics.get("alpha_ann", float("nan"))
                alpha_sharpe = portfolio_metrics.get("alpha_sharpe", float("nan"))
                logger.info(
                    "Portfolio metrics: Sharpe=%.2f, Sortino=%.2f, MaxDD=%.2f%%",
                    portfolio_metrics.get("sharpe_ratio", 0),
                    portfolio_metrics.get("sortino_ratio", 0),
                    portfolio_metrics.get("max_drawdown", 0) * 100,
                )
                if not np.isnan(alpha_ann):
                    logger.info(
                        "Alpha metrics: Ann.Alpha=%.2f%%, Alpha.Sharpe=%.2f",
                        alpha_ann * 100,
                        alpha_sharpe,
                    )

    # Run REALISTIC portfolio simulation (5-day holding, proper costs)
    # This is more realistic than daily rebalancing
    realistic_metrics = {}
    if not holdout_df.empty and len(reg_holdout_preds) > 0:
        holdout_with_preds = holdout_df.copy()
        holdout_with_preds["pred_return"] = reg_holdout_preds
        holdout_with_preds["ticker"] = holdout_with_preds.index if "ticker" not in holdout_with_preds.columns else holdout_with_preds["ticker"]
        
        realistic_result = simulate_realistic_portfolio(
            holdout_with_preds.reset_index(drop=True),
            date_col="date",
            ticker_col="ticker",
            label_col="future_ret",
            pred_col="pred_return",
            top_n=top_n,
            hold_days=horizon,  # Hold for prediction horizon
            cost_bps=20.0,  # More realistic round-trip cost
            market_ret_col="market_ret" if "market_ret" in holdout_with_preds.columns else None,
        )
        realistic_metrics = realistic_result.get("summary", {})
        
        if realistic_metrics.get("n_rebalances", 0) > 0:
            logger.info(
                "Realistic portfolio (%d-day hold): Sharpe=%.2f, Ann.Ret=%.1f%%, MaxDD=%.1f%%, Turnover=%.0f%%",
                horizon,
                realistic_metrics.get("sharpe_ratio", 0),
                realistic_metrics.get("ann_return", 0) * 100,
                realistic_metrics.get("max_drawdown", 0) * 100,
                realistic_metrics.get("avg_turnover", 0) * 100,
            )
            if not np.isnan(realistic_metrics.get("alpha_ann", float("nan"))):
                logger.info(
                    "Realistic alpha: Ann.Alpha=%.2f%%",
                    realistic_metrics.get("alpha_ann", 0) * 100,
                )

    # Walk-forward validation across multiple periods
    # This gives more robust evaluation across different market regimes
    walk_forward_results = {}
    wf_periods = build_walk_forward_periods(
        dates, n_periods=4, test_window=60, embargo_days=cfg.train_embargo_days, min_train=252
    )
    
    if len(wf_periods) >= 2 and reg_models:
        logger.info("Running walk-forward validation across %d periods...", len(wf_periods))
        period_metrics = []
        
        for period in wf_periods:
            # Get data for this period
            period_test_df = panel[panel["date"].isin(period.test_dates)]
            
            if period_test_df.empty or len(period_test_df) < 100:
                continue
            
            # Make predictions using IC-weighted ensemble with standardization
            period_X = period_test_df[selected_features].astype(float)
            period_preds = np.zeros(len(period_test_df))
            for i, m in enumerate(reg_models):
                if hasattr(m, "predict"):
                    try:
                        model_pred = m.predict(xgb.DMatrix(period_X) if isinstance(m, xgb.Booster) else period_X)
                    except Exception:
                        model_pred = m.predict(period_X)
                else:
                    model_pred = m.predict(xgb.DMatrix(period_X))
                model_pred = np.asarray(model_pred, dtype=float)
                # Standardize before combining (handles XGBRanker vs LightGBM scale mismatch)
                pred_std = float(np.nanstd(model_pred))
                if pred_std > 0:
                    model_pred = (model_pred - np.nanmean(model_pred)) / pred_std
                period_preds += model_pred * weights[i]
            
            # Evaluate with realistic simulation
            period_test_df = period_test_df.copy()
            period_test_df["pred_return"] = period_preds
            period_test_df["ticker"] = period_test_df.index if "ticker" not in period_test_df.columns else period_test_df["ticker"]
            
            period_result = simulate_realistic_portfolio(
                period_test_df.reset_index(drop=True),
                date_col="date",
                ticker_col="ticker",
                label_col="future_ret",
                pred_col="pred_return",
                top_n=top_n,
                hold_days=horizon,
                cost_bps=20.0,
                market_ret_col="market_ret" if "market_ret" in period_test_df.columns else None,
            )
            
            period_summary = period_result.get("summary", {})
            period_summary["period_id"] = period.period_id
            period_summary["test_start"] = str(period.test_start.date())
            period_summary["test_end"] = str(period.test_end.date())
            period_summary["n_test_samples"] = len(period_test_df)
            period_metrics.append(period_summary)
            
            logger.info(
                "  Period %d (%s to %s): Sharpe=%.2f, Alpha=%.1f%%",
                period.period_id,
                period.test_start.strftime("%Y-%m"),
                period.test_end.strftime("%Y-%m"),
                period_summary.get("sharpe_ratio", 0),
                period_summary.get("alpha_ann", 0) * 100,
            )
        
        if period_metrics:
            walk_forward_results = aggregate_walk_forward_results(period_metrics)
            agg = walk_forward_results.get("aggregate", {})
            consistency = walk_forward_results.get("consistency", 0)
            
            sharpe_agg = agg.get("sharpe_ratio", {})
            alpha_agg = agg.get("alpha_ann", {})
            
            logger.info(
                "Walk-forward summary: Sharpe=%.2f±%.2f, Alpha=%.1f%%±%.1f%%, Consistency=%.0f%%",
                sharpe_agg.get("mean", 0),
                sharpe_agg.get("std", 0),
                alpha_agg.get("mean", 0) * 100,
                alpha_agg.get("std", 0) * 100,
                consistency * 100,
            )

    # Compute final sector/industry encodings for inference
    # These are the mean target values per category from ALL training data
    sector_encodings = {}
    industry_encodings = {}
    global_mean = float(panel[label_col].mean())
    
    if "sector" in panel.columns:
        sector_stats = panel.groupby("sector")[label_col].agg(["mean", "count"])
        smoothing = 20.0
        for sector in sector_stats.index:
            n = sector_stats.loc[sector, "count"]
            mean = sector_stats.loc[sector, "mean"]
            # Apply same smoothing as during training
            encoded = (n * mean + smoothing * global_mean) / (n + smoothing)
            sector_encodings[str(sector)] = float(encoded)
        logger.info("Saved %d sector encodings for inference", len(sector_encodings))
    
    if "industry" in panel.columns:
        industry_stats = panel.groupby("industry")[label_col].agg(["mean", "count"])
        for industry in industry_stats.index:
            n = industry_stats.loc[industry, "count"]
            mean = industry_stats.loc[industry, "mean"]
            encoded = (n * mean + smoothing * global_mean) / (n + smoothing)
            industry_encodings[str(industry)] = float(encoded)
        logger.info("Saved %d industry encodings for inference", len(industry_encodings))

    metadata = {
        "horizon_days": horizon,
        "prediction_offset": 1,
        "note": "Model predicts (horizon+1) days forward to enable same-day actionability",
        "training_config": {
            "use_market_relative_returns": use_market_relative,
            "use_ranking_objective": use_ranking,
            "target_column": label_col,
            "n_xgb_models": n_xgb,
            "n_lgbm_models": n_lgbm,
        },
        "feature_columns": selected_features,  # Features used in final models (after selection)
        "feature_selection": {
            "original_count": len(feature_cols),
            "selected_count": len(selected_features),
            "dropped_features": dropped_features,
        },
        "filters": {
            "min_price_cad": float(cfg.min_price_cad),
            "min_avg_dollar_volume_cad": float(cfg.min_avg_dollar_volume_cad),
            "min_history_days": 90,
        },
        "regressor": {
            "params": best_regressor_params,
            "cv_metric": "mean_net_ret_topn",
            "cv_scores_topn": regressor_scores,
            "topn": {"top_n": int(top_n), "cost_bps": float(cost_bps)},
            "holdout": reg_holdout_metrics["summary"],
            "holdout_topn": reg_holdout_topn["summary"],
        },
        "date_range": {
            "start": str(pd.to_datetime(panel["date"]).min().date()),
            "end": str(pd.to_datetime(panel["date"]).max().date()),
        },
        "n_samples": int(len(panel)),
        "n_tickers": int(panel["ticker"].nunique()),
        "validation_metrics": validation_metrics,
        "calibration": calibration_metrics,
        "portfolio_metrics": portfolio_metrics,
        "realistic_portfolio_metrics": realistic_metrics,  # 5-day hold simulation
        "walk_forward_validation": walk_forward_results,  # Multi-period robustness
        "feature_importance": feature_importance,
        "feature_ic": feature_ic,  # Per-feature IC for adaptive selection
        "model_ics": model_ics,  # Per-model IC for ensemble weighting
        # Calibration map for realistic prediction magnitudes
        # Use label_col (future_alpha or future_ret) for calibration to match training target
        "prediction_calibration": build_calibration_map(panel[label_col].dropna(), n_quantiles=20),
        # Target encodings for inference
        "target_encodings": {
            "sector": sector_encodings,
            "industry": industry_encodings,
            "global_mean": global_mean,
        },
        # Peak timing model info
        "peak_model": peak_model_path,
        "peak_metrics": peak_metrics,
        "max_horizon_days": max_horizon,
    }
    metadata_rel = "metrics.json"
    write_json(model_dir / metadata_rel, metadata)

    # Save the ensemble manifest with model types and peak model
    save_ensemble(
        cfg.model_path, 
        model_rel_paths=reg_rel_paths, 
        model_types=model_types, 
        weights=None,
        peak_model_path=peak_model_path,
    )
    logger.info("Saved model bundle manifest to %s", cfg.model_path)
    return TrainResult(n_samples=int(len(panel)), n_tickers=int(panel["ticker"].nunique()), horizon_days=horizon)


def evaluate_model(cfg: Config, logger) -> dict[str, object]:
    """Evaluate the current model bundle on a fresh panel."""

    data_cache_dir = ensure_dir(cfg.data_cache_dir)
    us = fetch_us_universe(cfg=cfg, cache_dir=Path(cfg.cache_dir), logger=logger)
    tsx = fetch_tsx_universe(cfg=cfg, cache_dir=Path(cfg.cache_dir), logger=logger)
    tickers = list(dict.fromkeys(us.tickers + tsx.tickers))
    if cfg.max_total_tickers is not None:
        tickers = tickers[: cfg.max_total_tickers]

    fx = fetch_usdcad(
        fx_ticker=cfg.fx_ticker,
        lookback_days=max(cfg.feature_lookback_days, cfg.liquidity_lookback_days) + 365,
        cache_dir=Path(cfg.data_cache_dir),
        logger=logger,
    )
    prices = download_price_history(
        tickers=tickers,
        lookback_days=cfg.feature_lookback_days + 365,
        threads=cfg.yfinance_threads,
        batch_size=cfg.batch_size,
        logger=logger,
    )

    fundamentals = fetch_fundamentals(
        tickers=tickers,
        cache_dir=data_cache_dir,
        cache_ttl_days=cfg.fundamentals_cache_ttl_days,
        logger=logger,
    )
    
    # Fetch macro indicators for regime detection
    macro = fetch_macro_indicators(
        lookback_days=cfg.feature_lookback_days + 365,
        logger=logger,
    )
    
    panel = _build_panel_features(prices=prices, fx_usdcad=fx, fundamentals=fundamentals)
    if panel.empty:
        raise RuntimeError("No evaluation panel built")

    horizon = int(cfg.label_horizon_days)
    # Shift by (horizon + 1) so predictions made with today's features predict tomorrow onward
    # This ensures we can act on predictions before market open
    panel["future_ret"] = panel.groupby("ticker")["last_close_cad"].shift(-(horizon + 1)) / panel["last_close_cad"] - 1.0
    panel = panel.dropna(subset=["future_ret"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["last_close_cad", "avg_dollar_volume_cad"])
    panel = panel[panel["n_days"] >= 90]
    panel = panel[panel["last_close_cad"] >= float(cfg.min_price_cad)]
    panel = panel[panel["avg_dollar_volume_cad"] >= float(cfg.min_avg_dollar_volume_cad)]

    # CRITICAL: Cross-sectional winsorization and normalization
    # Financial returns are relative games - normalize within each date
    logger.info("Applying cross-sectional winsorization (MAD-based) and normalization...")
    panel = normalize_features_cross_section(panel, date_col="date")

    # Also winsorize labels to remove extreme outliers (MAD-based)
    panel["future_ret"] = panel.groupby("date")["future_ret"].transform(winsorize_mad)


    top_n = max(1, int(cfg.portfolio_size))
    cost_bps = float(getattr(cfg, "trade_cost_bps", 0.0))

    bundle = load_bundle(Path(cfg.model_path))
    metrics: dict[str, object] = {"ranker": None, "regressor": None}

    if bundle.get("ranker") is not None:
        preds = predict_score(bundle["ranker"], panel)
        rank_payload = evaluate_predictions(
            panel.assign(pred=preds), date_col="date", label_col="future_ret", pred_col="pred", group_col="is_tsx"
        )
        topn_payload = evaluate_topn_returns(
            panel.assign(pred=preds),
            date_col="date",
            label_col="future_ret",
            pred_col="pred",
            top_n=top_n,
            cost_bps=cost_bps,
        )
        rank_payload["topn"] = topn_payload
        metrics["ranker"] = rank_payload

    reg_models = bundle.get("regressor_models") or []
    if reg_models:
        preds = predict_ensemble(reg_models, bundle.get("regressor_weights"), panel)
        reg_payload = evaluate_predictions(
            panel.assign(pred=preds), date_col="date", label_col="future_ret", pred_col="pred", group_col="is_tsx"
        )
        topn_payload = evaluate_topn_returns(
            panel.assign(pred=preds),
            date_col="date",
            label_col="future_ret",
            pred_col="pred",
            top_n=top_n,
            cost_bps=cost_bps,
        )
        reg_payload["topn"] = topn_payload
        metrics["regressor"] = reg_payload

    return metrics
