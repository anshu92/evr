from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_screener.config import Config
from stock_screener.data.fundamentals import fetch_fundamentals
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.prices import download_price_history
from stock_screener.features.technical import compute_features, apply_target_encodings
from stock_screener.optimization.risk_parity import compute_inverse_vol_weights, compute_correlation_aware_weights, apply_confidence_weighting, apply_volatility_targeting, apply_conviction_sizing, apply_liquidity_adjustment, apply_correlation_limits, apply_beta_adjustment, apply_min_position_filter, apply_max_position_cap, apply_regime_exposure
from stock_screener.reporting.render import render_reports
from stock_screener.screening.screener import score_universe, select_sector_neutral, apply_entry_filters
from stock_screener.universe.tsx import fetch_tsx_universe
from stock_screener.universe.us import fetch_us_universe
from stock_screener.utils import Universe, ensure_dir, read_json, write_json, suppress_external_warnings

# Suppress known external library warnings
suppress_external_warnings()
from stock_screener.modeling.model import load_ensemble, load_model, predict, predict_ensemble, predict_ensemble_with_uncertainty
from stock_screener.modeling.transform import normalize_features_cross_section, calibrate_predictions
from stock_screener.portfolio.manager import PortfolioManager
from stock_screener.portfolio.state import load_portfolio_state, save_portfolio_state, compute_drawdown_scalar


def run_daily(cfg: Config, logger) -> None:
    """Run the daily screener + weights + reporting pipeline."""

    cache_dir = ensure_dir(cfg.cache_dir)
    data_cache_dir = ensure_dir(cfg.data_cache_dir)
    reports_dir = ensure_dir(cfg.reports_dir)
    ensure_dir(reports_dir / "debug")

    run_meta: dict[str, Any] = {
        "run_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config": asdict(cfg),
    }

    us = fetch_us_universe(cfg=cfg, cache_dir=cache_dir, logger=logger)
    tsx = fetch_tsx_universe(cfg=cfg, cache_dir=cache_dir, logger=logger)

    all_tickers = list(dict.fromkeys(us.tickers + tsx.tickers))
    if cfg.max_total_tickers is not None:
        all_tickers = all_tickers[: cfg.max_total_tickers]

    universe = Universe(
        tickers=all_tickers,
        meta={
            "us": us.meta,
            "tsx": tsx.meta,
            "total_requested": len(all_tickers),
        },
    )

    fx = fetch_usdcad(
        fx_ticker=cfg.fx_ticker,
        lookback_days=max(cfg.feature_lookback_days, cfg.liquidity_lookback_days),
        cache_dir=data_cache_dir,
        logger=logger,
    )

    prices = download_price_history(
        tickers=universe.tickers,
        lookback_days=cfg.feature_lookback_days,
        threads=cfg.yfinance_threads,
        batch_size=cfg.batch_size,
        logger=logger,
    )

    fundamentals = fetch_fundamentals(
        tickers=universe.tickers,
        cache_dir=data_cache_dir,
        cache_ttl_days=cfg.fundamentals_cache_ttl_days,
        logger=logger,
    )
    features = compute_features(
        prices=prices,
        fx_usdcad=fx,
        liquidity_lookback_days=cfg.liquidity_lookback_days,
        feature_lookback_days=cfg.feature_lookback_days,
        logger=logger,
        fundamentals=fundamentals,
    )

    if cfg.use_ml:
        try:
            mp = Path(cfg.model_path)
            metadata_path = mp.parent / "metrics.json" if mp.name.lower() == "manifest.json" else None
            
            # Load metadata first to get target encodings
            model_metadata = None
            if metadata_path and metadata_path.is_file():
                model_metadata = read_json(metadata_path)
                run_meta["model"] = {
                    "manifest_path": str(mp),
                    "metadata_path": str(metadata_path),
                    "metadata": model_metadata,
                }
                logger.info("Loaded model metadata from %s", metadata_path)
                
                # Apply target encodings from training
                target_enc = model_metadata.get("target_encodings", {})
                if target_enc:
                    features = apply_target_encodings(features, target_enc, logger)
            
            # Get selected features from metadata (if available)
            selected_features = None
            if model_metadata:
                selected_features = model_metadata.get("feature_columns")
                if selected_features:
                    fs = model_metadata.get("feature_selection", {})
                    if fs.get("dropped_features"):
                        logger.info(
                            "Using %d/%d features (dropped: %s)",
                            len(selected_features), 
                            fs.get("original_count", len(selected_features)),
                            fs.get("dropped_features", [])[:5],
                        )
            
            # Normalize features for ML
            features_ml = normalize_features_cross_section(features, date_col=None)
            
            if mp.name.lower() == "manifest.json":
                ensemble = load_ensemble(mp)
                models, weights = ensemble
                if models:
                    # Use uncertainty-aware predictions with selected features
                    pred_df = predict_ensemble_with_uncertainty(
                        models, weights, features_ml, feature_cols=selected_features
                    )
                    raw_preds = pred_df["pred_return"]
                    
                    # Apply prediction calibration if available
                    calibration_map = model_metadata.get("prediction_calibration") if model_metadata else None
                    if calibration_map and calibration_map.get("values"):
                        calibrated = calibrate_predictions(raw_preds, calibration_map, method="rank_preserve")
                        features["pred_return"] = calibrated
                        features["pred_return_raw"] = raw_preds
                        logger.info(
                            "Calibrated predictions: raw_mean=%.4f -> calibrated_mean=%.4f (training mean=%.4f)",
                            raw_preds.mean(), calibrated.mean(), calibration_map.get("mean", 0)
                        )
                    else:
                        features["pred_return"] = raw_preds
                    
                    features["pred_uncertainty"] = pred_df["pred_uncertainty"]
                    features["pred_confidence"] = pred_df["pred_confidence"]
                    logger.info(
                        "ML predictions: mean=%.4f, confidence range=[%.3f, %.3f]",
                        features["pred_return"].mean(),
                        pred_df["pred_confidence"].min(),
                        pred_df["pred_confidence"].max(),
                    )
                    logger.info("Loaded ML regressor ensemble from %s (%s members)", cfg.model_path, len(models))
            else:
                model = load_model(cfg.model_path)
                logger.info("Loaded ML model from %s", cfg.model_path)
                features["pred_return"] = predict(model, features_ml, feature_cols=selected_features)
        except Exception as e:
            logger.warning("ML enabled but model could not be loaded/used: %s", e)

    scored = score_universe(
        features=features,
        min_price_cad=cfg.min_price_cad,
        min_avg_dollar_volume_cad=cfg.min_avg_dollar_volume_cad,
        logger=logger,
    )
    n = int(cfg.top_n)
    if n <= 0:
        n = 50
    
    # Apply sector-neutral selection if enabled
    sector_neutral = getattr(cfg, "sector_neutral_selection", True)
    if sector_neutral and "sector" in scored.columns:
        screened = select_sector_neutral(scored, top_n=n, sector_col="sector", score_col="score")
        n_sectors = screened["sector"].nunique() if "sector" in screened.columns else 0
        logger.info(
            "Sector-neutral selection: %d tickers from %d sectors (from %d after filters)",
            len(screened), n_sectors, len(scored)
        )
    else:
        screened = scored.head(n).copy()
        logger.info("Screened universe: %s tickers (from %s after filters)", len(screened), len(scored))

    # Apply entry confirmation filters
    entry_filter_stats = {}
    screened, entry_filter_stats = apply_entry_filters(
        screened,
        min_confidence=getattr(cfg, "entry_min_confidence", None),
        min_pred_return=getattr(cfg, "entry_min_pred_return", None),
        max_volatility=getattr(cfg, "entry_max_volatility", None),
        min_momentum_5d=getattr(cfg, "entry_min_momentum_5d", None),
        momentum_alignment=getattr(cfg, "entry_momentum_alignment", True),
        logger=logger,
    )
    if entry_filter_stats.get("rejected_count", 0) > 0:
        run_meta["entry_filters"] = entry_filter_stats

    alpha_col = "pred_return" if "pred_return" in screened.columns else "score"
    
    # Compute portfolio weights with optional correlation awareness
    if cfg.use_correlation_weights:
        logger.info("Using correlation-aware risk parity weights")
        target_weights = compute_correlation_aware_weights(
            features=screened,
            prices=prices,  # Need historical prices for covariance
            portfolio_size=cfg.portfolio_size,
            weight_cap=cfg.weight_cap,
            logger=logger,
        )
    else:
        target_weights = compute_inverse_vol_weights(
            features=screened,
            portfolio_size=cfg.portfolio_size,
            weight_cap=cfg.weight_cap,
            logger=logger,
            alpha_col=alpha_col,
        )
    
    # Apply confidence weighting if available
    if "pred_confidence" in screened.columns:
        confidence = screened["pred_confidence"]
        target_weights = apply_confidence_weighting(
            target_weights,
            confidence,
            cfg.confidence_weight_floor,
            logger,
        )
    
    # Apply conviction-based position sizing if enabled
    conviction_sizing_enabled = getattr(cfg, "conviction_sizing", True)
    if conviction_sizing_enabled:
        target_weights = apply_conviction_sizing(
            target_weights,
            screened,
            pred_col="pred_return",
            confidence_col="pred_confidence",
            vol_col="vol_60d_ann",
            min_weight_scalar=getattr(cfg, "conviction_min_scalar", 0.5),
            max_weight_scalar=getattr(cfg, "conviction_max_scalar", 2.0),
            logger=logger,
        )
        run_meta["conviction_sizing"] = {"enabled": True}
    
    # Apply liquidity adjustment if enabled
    liquidity_adj_enabled = getattr(cfg, "liquidity_adjustment", True)
    if liquidity_adj_enabled:
        target_weights = apply_liquidity_adjustment(
            target_weights,
            screened,
            liquidity_col="avg_dollar_volume_cad",
            min_liquidity=getattr(cfg, "min_liquidity_cad", 100_000),
            target_liquidity=getattr(cfg, "target_liquidity_cad", 1_000_000),
            max_position_pct_of_volume=getattr(cfg, "max_position_pct_of_volume", 0.05),
            portfolio_value=cfg.portfolio_budget_cad,
            logger=logger,
        )
        run_meta["liquidity_adjustment"] = {"enabled": True}
    
    # Apply correlation-based position limits if enabled
    corr_limits_enabled = getattr(cfg, "correlation_limits", True)
    if corr_limits_enabled:
        target_weights = apply_correlation_limits(
            target_weights,
            prices,
            max_corr_weight=getattr(cfg, "max_corr_weight", 0.25),
            corr_threshold=getattr(cfg, "corr_threshold", 0.70),
            lookback_days=60,
            logger=logger,
        )
        run_meta["correlation_limits"] = {"enabled": True}
    
    # Apply beta adjustment if enabled
    beta_adj_enabled = getattr(cfg, "beta_adjustment", True)
    if beta_adj_enabled:
        target_weights = apply_beta_adjustment(
            target_weights,
            screened,
            beta_col="beta",
            target_beta=getattr(cfg, "target_portfolio_beta", 1.0),
            min_weight_scalar=getattr(cfg, "beta_min_scalar", 0.5),
            max_weight_scalar=getattr(cfg, "beta_max_scalar", 1.5),
            logger=logger,
        )
        run_meta["beta_adjustment"] = {"enabled": True}
    
    # Apply maximum position cap (final safety check)
    max_pos_pct = getattr(cfg, "max_position_pct", 0.20)
    if max_pos_pct and max_pos_pct < 1.0:
        target_weights = apply_max_position_cap(
            target_weights,
            max_position_pct=max_pos_pct,
            logger=logger,
        )
        run_meta["max_position_cap"] = {"max_pct": max_pos_pct}
    
    # Apply minimum position filter (remove dust positions)
    min_pos_pct = getattr(cfg, "min_position_pct", 0.02)
    if min_pos_pct and min_pos_pct > 0:
        target_weights = apply_min_position_filter(
            target_weights,
            min_position_pct=min_pos_pct,
            logger=logger,
        )
        run_meta["min_position_filter"] = {"min_pct": min_pos_pct}
    
    # Apply regime-aware exposure scaling
    cash_from_regime = 0.0
    regime_enabled = getattr(cfg, "regime_exposure_enabled", True)
    if regime_enabled:
        target_weights, cash_from_regime, regime_info = apply_regime_exposure(
            target_weights,
            features=screened,
            enabled=True,
            trend_weight=getattr(cfg, "regime_trend_weight", 0.4),
            breadth_weight=getattr(cfg, "regime_breadth_weight", 0.3),
            vol_weight=getattr(cfg, "regime_vol_weight", 0.3),
            min_scalar=getattr(cfg, "regime_min_scalar", 0.5),
            max_scalar=getattr(cfg, "regime_max_scalar", 1.2),
            logger=logger,
        )
        run_meta["regime_exposure"] = regime_info
    
    # Apply volatility targeting if enabled
    cash_from_vol_targeting = 0.0
    vol_targeting_enabled = getattr(cfg, "volatility_targeting", True)
    if vol_targeting_enabled:
        target_vol = getattr(cfg, "target_volatility", 0.15)
        target_weights, cash_from_vol_targeting = apply_volatility_targeting(
            target_weights,
            prices=prices,
            target_vol=target_vol,
            lookback_days=20,
            min_scalar=0.5,
            max_scalar=1.0,
            logger=logger,
        )
        run_meta["volatility_targeting"] = {
            "enabled": True,
            "target_vol": target_vol,
            "cash_allocation": cash_from_vol_targeting,
        }

    # Portfolio actions (stateful)
    # Use full `features` for exits so we can manage holdings even if they are not in today's top-N.
    prices_cad = features["last_close_cad"].astype(float)
    pred_return = features["pred_return"].astype(float) if "pred_return" in features.columns else None
    score = scored["score"].astype(float) if "score" in scored.columns else None
    state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
    # Migration safeguard:
    # Earlier versions created a state file with a large default cash balance and used `shares=1` placeholders,
    # without debiting cash on buys. If we now run with a small configured budget (e.g., 500 CAD), the cached
    # state would show misleading "cash" and P&L. Detect this legacy pattern and reset once so accounting is sane.
    try:
        budget = float(cfg.portfolio_budget_cad)
        open_positions = [p for p in state.positions if p.status == "OPEN"]
        legacy_placeholder = bool(open_positions) and all(int(p.shares) == 1 for p in open_positions)
        if budget > 0 and legacy_placeholder and float(state.cash_cad) >= budget * 25.0:
            logger.warning(
                "Portfolio state appears legacy (cash_cad=%s, budget_cad=%s, open_positions=%s). "
                "Resetting state to configured budget and rebuilding sized positions.",
                state.cash_cad,
                budget,
                len(open_positions),
            )
            state.cash_cad = float(budget)
            state.positions = []
            state.pnl_history = []
            state.last_updated = datetime.now(tz=timezone.utc)
            save_portfolio_state(cfg.portfolio_state_path, state)
    except Exception as e:
        logger.warning("Could not evaluate/reset legacy portfolio state: %s", e)
    
    # Apply drawdown-based position sizing if enabled
    drawdown_scalar = 1.0
    dd_info = {}
    drawdown_mgmt_enabled = getattr(cfg, "drawdown_management", True)
    if drawdown_mgmt_enabled and state.pnl_history:
        max_dd_threshold = getattr(cfg, "max_drawdown_threshold", -0.10)
        dd_min_scalar = getattr(cfg, "drawdown_min_scalar", 0.25)
        
        drawdown_scalar, dd_info = compute_drawdown_scalar(
            state,
            max_drawdown_threshold=max_dd_threshold,
            min_scalar=dd_min_scalar,
            recovery_threshold=-0.02,
        )
        
        if drawdown_scalar < 0.99:
            # Scale down weights due to drawdown
            target_weights["weight"] = target_weights["weight"] * drawdown_scalar
            logger.info(
                "Drawdown management: current_dd=%.1f%%, scalar=%.2f, days_in_dd=%d",
                dd_info.get("current_drawdown", 0) * 100,
                drawdown_scalar,
                dd_info.get("days_in_drawdown", 0),
            )
        
        run_meta["drawdown_management"] = {
            "enabled": True,
            "current_drawdown": dd_info.get("current_drawdown", 0),
            "drawdown_scalar": drawdown_scalar,
            "days_in_drawdown": dd_info.get("days_in_drawdown", 0),
            "max_equity": dd_info.get("max_equity", 0),
            "current_equity": dd_info.get("current_equity", 0),
        }
    
    pm = PortfolioManager(
        state_path=cfg.portfolio_state_path,
        max_holding_days=cfg.max_holding_days,
        max_holding_days_hard=cfg.max_holding_days_hard,
        extend_hold_min_pred_return=cfg.extend_hold_min_pred_return,
        extend_hold_min_score=cfg.extend_hold_min_score,
        max_positions=cfg.portfolio_size,
        stop_loss_pct=cfg.stop_loss_pct,
        take_profit_pct=cfg.take_profit_pct,
        trailing_stop_enabled=getattr(cfg, "trailing_stop_enabled", True),
        trailing_stop_activation_pct=getattr(cfg, "trailing_stop_activation_pct", 0.05),
        trailing_stop_distance_pct=getattr(cfg, "trailing_stop_distance_pct", 0.08),
        signal_decay_exit_enabled=getattr(cfg, "signal_decay_exit_enabled", True),
        signal_decay_threshold=getattr(cfg, "signal_decay_threshold", -0.02),
        dynamic_holding_enabled=getattr(cfg, "dynamic_holding_enabled", True),
        dynamic_holding_vol_scale=getattr(cfg, "dynamic_holding_vol_scale", 0.5),
        vol_adjusted_stop_enabled=getattr(cfg, "vol_adjusted_stop_enabled", True),
        vol_adjusted_stop_base=getattr(cfg, "vol_adjusted_stop_base", 0.08),
        vol_adjusted_stop_min=getattr(cfg, "vol_adjusted_stop_min", 0.04),
        vol_adjusted_stop_max=getattr(cfg, "vol_adjusted_stop_max", 0.15),
        age_urgency_enabled=getattr(cfg, "age_urgency_enabled", True),
        age_urgency_start_day=getattr(cfg, "age_urgency_start_day", 2),
        age_urgency_min_return=getattr(cfg, "age_urgency_min_return", 0.01),
        peak_detection_enabled=cfg.peak_detection_enabled,
        peak_sell_portion_pct=cfg.peak_sell_portion_pct,
        peak_min_gain_pct=cfg.peak_min_gain_pct,
        peak_min_holding_days=cfg.peak_min_holding_days,
        peak_pred_return_threshold=cfg.peak_pred_return_threshold,
        peak_score_percentile_drop=cfg.peak_score_percentile_drop,
        peak_rsi_overbought=cfg.peak_rsi_overbought,
        peak_above_ma_ratio=cfg.peak_above_ma_ratio,
        logger=logger,
    )
    # Extract market volatility regime for dynamic holding period
    market_vol_regime = None
    if "market_vol_regime" in features.columns:
        market_vol_regime = float(features["market_vol_regime"].iloc[0]) if len(features) > 0 else None
    
    exit_actions = pm.apply_exits(
        state, 
        prices_cad=prices_cad, 
        pred_return=pred_return, 
        score=score, 
        features=features,
        market_vol_regime=market_vol_regime,
    )
    if exit_actions:
        sells = len([a for a in exit_actions if a.action in ("SELL", "SELL_PARTIAL")])
        logger.info("Exited %s position(s) (time/stop/target/peak).", sells)

    trade_plan = pm.build_trade_plan(
        state=state,
        screened=screened,
        weights=target_weights,
        prices_cad=prices_cad,
    )

    open_tickers = [p.ticker for p in state.positions if p.status == "OPEN"]
    holdings_features = screened.loc[screened.index.intersection(open_tickers)].copy()
    holdings_features = holdings_features.sort_values("score", ascending=False)
    if holdings_features.empty:
        holdings_weights = holdings_features.copy()
        for col in ["weight", "score", "last_close_cad", "ret_60d", "vol_60d_ann"]:
            if col not in holdings_weights.columns:
                holdings_weights[col] = pd.NA
    else:
        holdings_weights = compute_inverse_vol_weights(
            features=holdings_features,
            portfolio_size=len(holdings_features),
            weight_cap=cfg.weight_cap,
            logger=logger,
            alpha_col=alpha_col,
        )
    # Attach current holdings sizing for reporting (shares + position value).
    shares_by_ticker = {p.ticker: int(p.shares) for p in state.positions if p.status == "OPEN"}
    holdings_weights = holdings_weights.copy()
    holdings_weights["shares"] = [shares_by_ticker.get(str(t), pd.NA) for t in holdings_weights.index.astype(str)]

    render_reports(
        reports_dir=Path(cfg.reports_dir),
        run_meta=run_meta,
        universe_meta=universe.meta,
        screened=screened,
        weights=holdings_weights,
        trade_actions=trade_plan.actions,
        logger=logger,
        portfolio_pnl_history=state.pnl_history,
        fx_usdcad_rate=float(fx.dropna().iloc[-1]) if fx is not None and not fx.dropna().empty else None,
        total_processed=len(features),
    )

    # Persist metadata for debugging/auditing
    write_json(cache_dir / "last_run_meta.json", run_meta)
    logger.info("Wrote reports to %s", reports_dir.resolve())
