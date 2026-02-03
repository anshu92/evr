from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce noise
except ImportError:
    optuna = None

from stock_screener.config import Config
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.fundamentals import fetch_fundamentals
from stock_screener.data.prices import download_price_history
from stock_screener.features.fundamental_scores import add_fundamental_composites
from stock_screener.modeling.eval import build_time_splits, evaluate_predictions, evaluate_topn_returns, compute_calibration, compute_portfolio_metrics
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
from stock_screener.modeling.transform import normalize_features_cross_section, winsorize_mad
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
            market_cap_cad = None
            if market_cap is not None:
                market_cap_cad = float(market_cap)
                if not is_tsx and fx_factor and not pd.isna(fx_factor):
                    market_cap_cad = market_cap_cad * float(fx_factor)
            log_market_cap = float(np.log10(market_cap_cad)) if market_cap_cad and float(market_cap_cad) > 0 else np.nan
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
    panel = _build_panel_features(prices=prices, fx_usdcad=fx, fundamentals=fundamentals)
    if panel.empty:
        raise RuntimeError("No training panel built")

    horizon = int(cfg.label_horizon_days)
    top_n = max(1, int(cfg.portfolio_size))
    cost_bps = float(getattr(cfg, "trade_cost_bps", 0.0))
    # Shift by (horizon + 1) so predictions made with today's features predict tomorrow onward
    # This ensures we can act on predictions before market open
    panel["future_ret"] = panel.groupby("ticker")["last_close_cad"].shift(-(horizon + 1)) / panel["last_close_cad"] - 1.0

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
    panel = normalize_features_cross_section(panel, date_col="date")

    # Also winsorize labels to remove extreme outliers (MAD-based)
    panel["future_ret"] = panel.groupby("date")["future_ret"].transform(winsorize_mad)
    
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

    def _rank_ic_for_preds(df: pd.DataFrame, pred: pd.Series) -> dict[str, object]:
        if df.empty:
            return {"summary": {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}}
        temp = df.copy()
        temp["pred"] = pred
        return evaluate_predictions(temp, date_col="date", label_col="future_ret", pred_col="pred", group_col="is_tsx")

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
        return evaluate_topn_returns(
            temp, date_col="date", label_col="future_ret", pred_col="pred", top_n=top_n, cost_bps=cost_bps
        )


    # Hyperparameter tuning with Optuna (if available) or fallback to manual search
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
                    train_df["future_ret"].astype(float),
                    eval_set=[(val_df[feature_cols], val_df["future_ret"].astype(float))],
                    verbose=False,
                )
                preds = model.predict(val_df[feature_cols])
                metrics = _topn_for_preds(val_df, pd.Series(preds, index=val_df.index))
                scores.append(float(metrics["summary"]["mean_net_ret"]))
            return float(np.nanmean(scores)) if scores else float("nan")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(_optuna_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best_regressor_params = study.best_params
        logger.info("Optuna best params: %s (score=%.6f)", best_regressor_params, study.best_value)
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
                    train_df["future_ret"].astype(float),
                    eval_set=[(val_df[feature_cols], val_df["future_ret"].astype(float))],
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



    # Use configured ensemble composition or default
    n_xgb = getattr(cfg, 'ensemble_xgb_count', 3)
    n_lgbm = getattr(cfg, 'ensemble_lgbm_count', 3)
    use_lgbm = getattr(cfg, 'use_lightgbm', True)
    
    if not use_lgbm:
        n_xgb = 7  # Fallback to 7 XGBoost models
        n_lgbm = 0
    
    model_dir = Path(cfg.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    reg_rel_paths: list[str] = []
    model_types: list[str] = []

    logger.info(
        "Training mixed ensemble: %d XGBoost + %d LightGBM models on %s samples, %s tickers",
        n_xgb,
        n_lgbm,
        len(train_df),
        train_df["ticker"].nunique(),
    )
    
    # Train XGBoost models
    for i in range(n_xgb):
        seed = 42 + i * 10
        m = build_model(random_state=seed)
        m.set_params(**best_regressor_params, early_stopping_rounds=50)
        if not val_df.empty:
            m.fit(
                train_df[feature_cols],
                train_df["future_ret"].astype(float),
                eval_set=[(val_df[feature_cols], val_df["future_ret"].astype(float))],
                verbose=False,
            )
        else:
            m.fit(train_df[feature_cols], train_df["future_ret"].astype(float))
        rel = f"xgb_model_{i}.json"
        save_model(m, model_dir / rel)
        reg_rel_paths.append(rel)
        model_types.append("xgboost")
        logger.info("Trained XGBoost model %d/%d", i+1, n_xgb)
    
    # Train LightGBM models if enabled
    if use_lgbm and n_lgbm > 0:
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
            if not val_df.empty:
                m.fit(
                    train_df[feature_cols],
                    train_df["future_ret"].astype(float),
                    eval_set=[(val_df[feature_cols], val_df["future_ret"].astype(float))],
                    callbacks=[],  # Disable callbacks to reduce verbosity
                )
            else:
                m.fit(train_df[feature_cols], train_df["future_ret"].astype(float))
            rel = f"lgbm_model_{i}.txt"
            save_model(m, model_dir / rel)
            reg_rel_paths.append(rel)
            model_types.append("lightgbm")
            logger.info("Trained LightGBM model %d/%d", i+1, n_lgbm)

    reg_holdout_preds: list[float] = []
    if not holdout_df.empty:
        preds = np.zeros(len(holdout_df), dtype=float)
        for rel, mtype in zip(reg_rel_paths, model_types):
            model = load_model(model_dir / rel, mtype)
            if mtype == "xgboost":
                preds += model.predict(xgb.DMatrix(holdout_df[feature_cols])).astype(float)
            else:  # lightgbm
                preds += model.predict(holdout_df[feature_cols]).astype(float)
        preds = preds / float(len(reg_rel_paths))
        reg_holdout_preds = preds.tolist()

    reg_holdout_metrics = _rank_ic_for_preds(holdout_df, pd.Series(reg_holdout_preds, index=holdout_df.index))
    reg_holdout_topn = _topn_for_preds(holdout_df, pd.Series(reg_holdout_preds, index=holdout_df.index))



    # Compute feature importance from first XGBoost model
    feature_importance = {}
    xgb_models = [rel for rel, mtype in zip(reg_rel_paths, model_types) if mtype == "xgboost"]
    if xgb_models:
        first_model = load_model(model_dir / xgb_models[0], "xgboost")
        # Use get_score() for xgb.Booster
        try:
            importance = first_model.get_score(importance_type="gain")
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

    # Enhanced metrics including calibration and portfolio stats
    calibration_metrics = {}
    portfolio_metrics = {}
    
    if not holdout_df.empty and len(reg_holdout_preds) > 0:
        # Compute calibration
        calibration_result = compute_calibration(
            pd.Series(reg_holdout_preds, index=holdout_df.index),
            holdout_df["future_ret"],
            n_bins=10
        )
        calibration_metrics = {
            "calibration_error": calibration_result["calibration_error"],
            "n_deciles": len(calibration_result["by_decile"]) if isinstance(calibration_result["by_decile"], pd.DataFrame) else 0,
        }
        logger.info("Calibration error: %.6f", calibration_result["calibration_error"])
        
        # Compute portfolio metrics from top-N daily returns
        if "daily" in reg_holdout_topn and isinstance(reg_holdout_topn["daily"], pd.DataFrame):
            daily_df = reg_holdout_topn["daily"]
            if "mean_ret" in daily_df.columns and len(daily_df) > 0:
                portfolio_metrics = compute_portfolio_metrics(daily_df["mean_ret"])
                logger.info(
                    "Portfolio metrics: Sharpe=%.2f, Sortino=%.2f, MaxDD=%.2f%%",
                    portfolio_metrics.get("sharpe_ratio", 0),
                    portfolio_metrics.get("sortino_ratio", 0),
                    portfolio_metrics.get("max_drawdown", 0) * 100,
                )

    metadata = {
        "horizon_days": horizon,
        "prediction_offset": 1,
        "note": "Model predicts (horizon+1) days forward to enable same-day actionability",
        "feature_columns": FEATURE_COLUMNS,
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
        "feature_importance": feature_importance,
    }
    metadata_rel = "metrics.json"
    write_json(model_dir / metadata_rel, metadata)

    # Save the ensemble manifest with model types
    save_ensemble(cfg.model_path, model_rel_paths=reg_rel_paths, model_types=model_types, weights=None)
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
