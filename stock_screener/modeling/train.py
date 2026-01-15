from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from stock_screener.config import Config
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.fundamentals import fetch_fundamentals
from stock_screener.data.prices import download_price_history
from stock_screener.modeling.eval import build_time_splits, evaluate_predictions
from stock_screener.modeling.model import (
    FEATURE_COLUMNS,
    build_model,
    load_bundle,
    predict_ensemble,
    predict_score,
    save_ensemble,
    save_model,
)
from stock_screener.universe.tsx import fetch_tsx_universe
from stock_screener.universe.us import fetch_us_universe
from stock_screener.utils import Universe, ensure_dir, write_json


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

    frames: list[pd.DataFrame] = []
    tickers = list(prices.columns.levels[0])
    for t in tickers:
        if (t, "Close") not in prices.columns:
            continue
        close = prices[(t, "Close")].astype(float)
        vol = prices[(t, "Volume")].astype(float) if (t, "Volume") in prices.columns else pd.Series(index=idx, dtype=float)
        is_tsx = _is_tsx_ticker(t)
        fx_series = 1.0 if is_tsx else fx
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

        vol_anom_30d = vol / vol.rolling(30).mean().replace(0.0, np.nan)
        avg_dollar_vol = (close_cad * vol).rolling(30).mean()
        n_days = close_cad.expanding().count()

        sector_hash = np.nan
        industry_hash = np.nan
        log_market_cap = np.nan
        beta = np.nan
        # New fundamental fields
        trailing_pe = np.nan
        forward_pe = np.nan
        price_to_book = np.nan
        price_to_sales = np.nan
        enterprise_to_revenue = np.nan
        enterprise_to_ebitda = np.nan
        profit_margins = np.nan
        operating_margins = np.nan
        return_on_equity = np.nan
        return_on_assets = np.nan
        revenue_growth = np.nan
        earnings_growth = np.nan
        earnings_quarterly_growth = np.nan
        debt_to_equity = np.nan
        current_ratio = np.nan
        quick_ratio = np.nan
        dividend_yield = np.nan
        payout_ratio = np.nan
        target_mean_price = np.nan
        recommendation_mean = np.nan
        num_analyst_opinions = np.nan
        
        if fundamentals is not None and t in fundamentals.index:
            row = fundamentals.loc[t]
            sector_hash = _hash_to_float(str(row.get("sector"))) if row.get("sector") else np.nan
            industry_hash = _hash_to_float(str(row.get("industry"))) if row.get("industry") else np.nan
            market_cap = row.get("marketCap")
            log_market_cap = float(np.log10(market_cap)) if market_cap and float(market_cap) > 0 else np.nan
            beta = float(row.get("beta")) if row.get("beta") is not None else np.nan
            # Extract new fundamental data
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

        df = pd.DataFrame(
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
                "log_market_cap": log_market_cap,
                "beta": beta,
                "sector_hash": sector_hash,
                "industry_hash": industry_hash,
                "fx_ret_5d": fx_ret_5d_series.values if hasattr(fx_ret_5d_series, "values") else fx_ret_5d_series,
                "fx_ret_20d": fx_ret_20d_series.values if hasattr(fx_ret_20d_series, "values") else fx_ret_20d_series,
                "n_days": n_days.values,
                # Raw fundamental features
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
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        rank_map = {
            "ret_20d": "rank_ret_20d",
            "ret_60d": "rank_ret_60d",
            "vol_60d_ann": "rank_vol_60d",
            "avg_dollar_volume_cad": "rank_avg_dollar_volume",
        }
        for col, out_col in rank_map.items():
            if col in panel.columns:
                panel[out_col] = panel.groupby("date")[col].rank(pct=True)
        
        # Compute composite fundamental features
        def _safe_zscore_panel(series: pd.Series) -> pd.Series:
            """Z-score with NaN handling for panel data."""
            mu = series.mean()
            sd = series.std()
            if sd == 0 or pd.isna(sd):
                return pd.Series(0.0, index=series.index)
            return (series - mu) / sd
        
        # Value score: inverse of valuation ratios (per date)
        def _compute_value_score(group):
            value_components = []
            if "trailing_pe" in group.columns:
                inv_pe = 1.0 / group["trailing_pe"].replace([0, np.inf, -np.inf], np.nan)
                if inv_pe.notna().sum() > 0:
                    value_components.append(_safe_zscore_panel(inv_pe))
            if "price_to_book" in group.columns:
                inv_pb = 1.0 / group["price_to_book"].replace([0, np.inf, -np.inf], np.nan)
                if inv_pb.notna().sum() > 0:
                    value_components.append(_safe_zscore_panel(inv_pb))
            if "price_to_sales" in group.columns:
                inv_ps = 1.0 / group["price_to_sales"].replace([0, np.inf, -np.inf], np.nan)
                if inv_ps.notna().sum() > 0:
                    value_components.append(_safe_zscore_panel(inv_ps))
            if value_components:
                return pd.concat(value_components, axis=1).mean(axis=1)
            return pd.Series(0.0, index=group.index)
        
        panel["value_score"] = panel.groupby("date").apply(_compute_value_score).reset_index(level=0, drop=True)
        
        # Quality score: profitability and efficiency (per date)
        def _compute_quality_score(group):
            quality_components = []
            if "return_on_equity" in group.columns:
                quality_components.append(_safe_zscore_panel(group["return_on_equity"]))
            if "operating_margins" in group.columns:
                quality_components.append(_safe_zscore_panel(group["operating_margins"]))
            if "profit_margins" in group.columns:
                quality_components.append(_safe_zscore_panel(group["profit_margins"]))
            if quality_components:
                return pd.concat(quality_components, axis=1).mean(axis=1)
            return pd.Series(0.0, index=group.index)
        
        panel["quality_score"] = panel.groupby("date").apply(_compute_quality_score).reset_index(level=0, drop=True)
        
        # Growth score: revenue and earnings growth (per date)
        def _compute_growth_score(group):
            growth_components = []
            if "revenue_growth" in group.columns:
                growth_components.append(_safe_zscore_panel(group["revenue_growth"]))
            if "earnings_growth" in group.columns:
                growth_components.append(_safe_zscore_panel(group["earnings_growth"]))
            if growth_components:
                return pd.concat(growth_components, axis=1).mean(axis=1)
            return pd.Series(0.0, index=group.index)
        
        panel["growth_score"] = panel.groupby("date").apply(_compute_growth_score).reset_index(level=0, drop=True)
        
        # Surprise factors: analyst expectations vs current price
        if "target_mean_price" in panel.columns and "last_close_cad" in panel.columns:
            panel["pe_discount"] = (panel["target_mean_price"] - panel["last_close_cad"]) / panel["last_close_cad"].replace(0, np.nan)
            panel["pe_discount"] = panel["pe_discount"].replace([np.inf, -np.inf], np.nan)
        else:
            panel["pe_discount"] = 0.0
        
        # Fundamental momentum (simplified)
        if "earnings_quarterly_growth" in panel.columns:
            panel["roc_growth"] = panel["earnings_quarterly_growth"]
        else:
            panel["roc_growth"] = 0.0
        
        # Interaction features
        if "value_score" in panel.columns and "ret_120d" in panel.columns:
            panel["value_momentum"] = panel["value_score"] * panel["ret_120d"]
        else:
            panel["value_momentum"] = 0.0
        
        if "vol_60d_ann" in panel.columns and "log_market_cap" in panel.columns:
            panel["vol_size"] = panel["vol_60d_ann"] * panel["log_market_cap"]
        else:
            panel["vol_size"] = 0.0
        
        if "quality_score" in panel.columns and "growth_score" in panel.columns:
            panel["quality_growth"] = panel["quality_score"] * panel["growth_score"]
        else:
            panel["quality_growth"] = 0.0
    
    return panel


def train_and_save(cfg: Config, logger) -> TrainResult:
    """Train an ensemble of ML models and write manifest to cfg.model_path."""

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

    fundamentals = fetch_fundamentals(tickers=tickers, cache_dir=data_cache_dir, logger=logger)
    panel = _build_panel_features(prices=prices, fx_usdcad=fx, fundamentals=fundamentals)
    if panel.empty:
        raise RuntimeError("No training panel built")

    horizon = int(cfg.label_horizon_days)
    panel["future_ret"] = panel.groupby("ticker")["last_close_cad"].shift(-horizon) / panel["last_close_cad"] - 1.0

    # Drop rows without labels/features
    panel = panel.dropna(subset=["future_ret"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["last_close_cad", "avg_dollar_volume_cad"])
    panel = panel[panel["n_days"] >= 90]
    panel = panel[panel["last_close_cad"] >= float(cfg.min_price_cad)]
    panel = panel[panel["avg_dollar_volume_cad"] >= float(cfg.min_avg_dollar_volume_cad)]

    # CRITICAL: Cross-sectional winsorization and normalization
    # Financial returns are relative games - normalize within each date
    logger.info("Applying cross-sectional winsorization (MAD-based) and normalization...")
    
    def _winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
        """Clip extreme values using MAD (Median Absolute Deviation)."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or pd.isna(mad):
            return series
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        return series.clip(lower=lower, upper=upper)
    
    def _zscore(series: pd.Series) -> pd.Series:
        """Z-score normalize within group."""
        mu = series.mean()
        sd = series.std()
        if sd == 0 or pd.isna(sd):
            return series * 0.0
        return (series - mu) / sd
    
    # Winsorize and normalize features within each date (cross-sectional)
    feature_cols_to_norm = [
        col for col in FEATURE_COLUMNS 
        if col in panel.columns and col not in ["sector_hash", "industry_hash", "is_tsx"]
    ]
    
    for col in feature_cols_to_norm:
        if panel[col].notna().sum() > 0:
            panel[col] = panel.groupby("date")[col].transform(_winsorize_mad)
            panel[col] = panel.groupby("date")[col].transform(_zscore)
    
    # Also winsorize labels to remove extreme outliers (MAD-based)
    panel["future_ret"] = panel.groupby("date")["future_ret"].transform(_winsorize_mad)


    dates = pd.to_datetime(panel["date"]).dt.normalize().unique().tolist()
    splits, holdout = build_time_splits(dates, embargo_days=horizon)
    val_dates = splits[-1].val_dates if splits else []

    def _subset_by_dates(dates_list: list[pd.Timestamp]) -> pd.DataFrame:
        if not dates_list:
            return panel.iloc[0:0].copy()
        mask = panel["date"].isin(dates_list)
        return panel.loc[mask]

    def _rank_ic_for_preds(df: pd.DataFrame, pred: pd.Series) -> dict[str, object]:
        if df.empty:
            return {"summary": {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}}
        temp = df.copy()
        temp["pred"] = pred
        return evaluate_predictions(temp, date_col="date", label_col="future_ret", pred_col="pred", group_col="is_tsx")


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
                train_df[FEATURE_COLUMNS],
                train_df["future_ret"].astype(float),
                eval_set=[(val_df[FEATURE_COLUMNS], val_df["future_ret"].astype(float))],
                verbose=False,
            )
            preds = model.predict(val_df[FEATURE_COLUMNS])
            metrics = _rank_ic_for_preds(val_df, pd.Series(preds, index=val_df.index))
            scores.append(float(metrics["summary"]["mean_ic"]))
        return float(np.nanmean(scores)) if scores else float("nan")

    regressor_scores = {str(p): _eval_regressor_params(p) for p in regressor_candidates}
    best_regressor_params = max(regressor_candidates, key=lambda p: regressor_scores.get(str(p), float("-inf")))

    train_df = _subset_by_dates(holdout.train_dates)
    val_df = _subset_by_dates(val_dates)
    holdout_df = _subset_by_dates(holdout.holdout_dates)



    seeds = [7, 13, 21, 42, 73, 99, 123]
    model_dir = Path(cfg.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    reg_rel_paths: list[str] = []

    logger.info(
        "Training ML regressor ensemble on %s samples, %s tickers (seeds=%s)",
        len(train_df),
        train_df["ticker"].nunique(),
        seeds,
    )
    for s in seeds:
        m = build_model(random_state=s)
        m.set_params(**best_regressor_params, early_stopping_rounds=50)
        if not val_df.empty:
            m.fit(
                train_df[FEATURE_COLUMNS],
                train_df["future_ret"].astype(float),
                eval_set=[(val_df[FEATURE_COLUMNS], val_df["future_ret"].astype(float))],
                verbose=False,
            )
        else:
            m.fit(train_df[FEATURE_COLUMNS], train_df["future_ret"].astype(float))
        rel = f"regressor_seed_{s}.json"
        save_model(m, model_dir / rel)
        reg_rel_paths.append(rel)

    reg_holdout_preds: list[float] = []
    if not holdout_df.empty:
        preds = np.zeros(len(holdout_df), dtype=float)
        for rel in reg_rel_paths:
            model = build_model()
            model.load_model(str(model_dir / rel))
            preds += model.predict(holdout_df[FEATURE_COLUMNS]).astype(float)
        preds = preds / float(len(reg_rel_paths))
        reg_holdout_preds = preds.tolist()

    reg_holdout_metrics = _rank_ic_for_preds(holdout_df, pd.Series(reg_holdout_preds, index=holdout_df.index))



    # Compute feature importance from first regressor
    feature_importance = {}
    if reg_rel_paths:
        first_model = build_model()
        first_model.load_model(str(model_dir / reg_rel_paths[0]))
        # Use get_booster().get_score() for XGBRegressor
        try:
            importance = first_model.get_booster().get_score(importance_type="gain")
            feature_importance = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]}
            logger.info("Top 10 features by gain: %s", list(feature_importance.keys())[:10])
        except Exception as e:
            logger.warning("Could not extract feature importance: %s", e)

    # Compute per-feature IC on holdout
    feature_ic = {}
    if not holdout_df.empty:
        for col in FEATURE_COLUMNS:
            if col in holdout_df.columns and holdout_df[col].notna().sum() > 10:
                try:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(holdout_df[col].fillna(0), holdout_df["future_ret"].fillna(0), nan_policy="omit")
                    feature_ic[col] = float(ic)
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
                        sector_df, date_col="date", label_col="future_ret", 
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
                        q_df, date_col="date", label_col="future_ret",
                        pred_col="pred_return", group_col=None
                    )
                    mcap_ics[f"quintile_{q+1}"] = q_metrics.get("summary", {})
            if mcap_ics:
                validation_metrics["by_market_cap_quintile"] = mcap_ics
                logger.info("IC by market cap quintile: %d quintiles evaluated", len(mcap_ics))

    metadata = {
        "horizon_days": horizon,
        "feature_columns": FEATURE_COLUMNS,
        "filters": {
            "min_price_cad": float(cfg.min_price_cad),
            "min_avg_dollar_volume_cad": float(cfg.min_avg_dollar_volume_cad),
            "min_history_days": 90,
        },
        "regressor": {
            "params": best_regressor_params,
            "cv_scores": regressor_scores,
            "holdout": reg_holdout_metrics["summary"],
        },
        "date_range": {
            "start": str(pd.to_datetime(panel["date"]).min().date()),
            "end": str(pd.to_datetime(panel["date"]).max().date()),
        },
        "n_samples": int(len(panel)),
        "n_tickers": int(panel["ticker"].nunique()),
        "validation_metrics": validation_metrics,
    }
    metadata_rel = "metrics.json"
    write_json(model_dir / metadata_rel, metadata)

    save_ensemble(cfg.model_path, model_rel_paths=reg_rel_paths, weights=None)
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

    fundamentals = fetch_fundamentals(tickers=tickers, cache_dir=data_cache_dir, logger=logger)
    panel = _build_panel_features(prices=prices, fx_usdcad=fx, fundamentals=fundamentals)
    if panel.empty:
        raise RuntimeError("No evaluation panel built")

    horizon = int(cfg.label_horizon_days)
    panel["future_ret"] = panel.groupby("ticker")["last_close_cad"].shift(-horizon) / panel["last_close_cad"] - 1.0
    panel = panel.dropna(subset=["future_ret"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["last_close_cad", "avg_dollar_volume_cad"])
    panel = panel[panel["n_days"] >= 90]
    panel = panel[panel["last_close_cad"] >= float(cfg.min_price_cad)]
    panel = panel[panel["avg_dollar_volume_cad"] >= float(cfg.min_avg_dollar_volume_cad)]

    # CRITICAL: Cross-sectional winsorization and normalization
    # Financial returns are relative games - normalize within each date
    logger.info("Applying cross-sectional winsorization (MAD-based) and normalization...")
    
    def _winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
        """Clip extreme values using MAD (Median Absolute Deviation)."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or pd.isna(mad):
            return series
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        return series.clip(lower=lower, upper=upper)
    
    def _zscore(series: pd.Series) -> pd.Series:
        """Z-score normalize within group."""
        mu = series.mean()
        sd = series.std()
        if sd == 0 or pd.isna(sd):
            return series * 0.0
        return (series - mu) / sd
    
    # Winsorize and normalize features within each date (cross-sectional)
    feature_cols_to_norm = [
        col for col in FEATURE_COLUMNS 
        if col in panel.columns and col not in ["sector_hash", "industry_hash", "is_tsx"]
    ]
    
    for col in feature_cols_to_norm:
        if panel[col].notna().sum() > 0:
            panel[col] = panel.groupby("date")[col].transform(_winsorize_mad)
            panel[col] = panel.groupby("date")[col].transform(_zscore)
    
    # Also winsorize labels to remove extreme outliers (MAD-based)
    panel["future_ret"] = panel.groupby("date")["future_ret"].transform(_winsorize_mad)


    bundle = load_bundle(Path(cfg.model_path))
    metrics: dict[str, object] = {"ranker": None, "regressor": None}

    if bundle.get("ranker") is not None:
        preds = predict_score(bundle["ranker"], panel)
        metrics["ranker"] = evaluate_predictions(panel.assign(pred=preds), date_col="date", label_col="future_ret", pred_col="pred", group_col="is_tsx")

    reg_models = bundle.get("regressor_models") or []
    if reg_models:
        preds = predict_ensemble(reg_models, bundle.get("regressor_weights"), panel)
        metrics["regressor"] = evaluate_predictions(panel.assign(pred=preds), date_col="date", label_col="future_ret", pred_col="pred", group_col="is_tsx")

    return metrics
