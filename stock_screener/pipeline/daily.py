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
from stock_screener.modeling.model import load_ensemble, load_model, predict, predict_ensemble, predict_ensemble_with_uncertainty, predict_peak_days
from stock_screener.modeling.transform import normalize_features_cross_section, calibrate_predictions
from stock_screener.portfolio.manager import PortfolioManager, TradePlan
from stock_screener.portfolio.state import load_portfolio_state, save_portfolio_state, compute_drawdown_scalar
from stock_screener.reward.tracker import RewardEntry, RewardLog, ActionRewardEntry, ActionRewardLog
from stock_screener.reward.feedback import (
    compute_online_ic, compute_ensemble_reward_weights, compute_prediction_bias,
    score_actions, compute_action_quality_summary,
)
from stock_screener.reward.policy import (
    RewardPolicy, build_state_vector, compute_equity_slope, compute_recent_sharpe,
)


def compute_dynamic_portfolio_size(
    screened: pd.DataFrame,
    min_confidence: float,
    min_pred_return: float,
    max_positions: int,
    model_ic: float | None,
    logger,
) -> int:
    """Compute fully dynamic portfolio size based on model metrics and predicted returns.
    
    No base size - portfolio can be 1 to max_positions based entirely on:
    - Number of stocks meeting confidence and return thresholds
    - Model IC (if available) to scale aggressiveness
    - Quality score combining confidence and predicted return
    """
    if screened.empty:
        logger.warning("Dynamic sizing: No stocks to evaluate, defaulting to 1")
        return 1
    
    has_confidence = "pred_confidence" in screened.columns
    has_return = "pred_return" in screened.columns
    
    if not has_confidence and not has_return:
        # Fallback: use score percentile to pick top performers
        if "score" in screened.columns:
            # Take stocks with score > 75th percentile, at least 1
            threshold = screened["score"].quantile(0.75)
            count = max(1, min(max_positions, (screened["score"] >= threshold).sum()))
            logger.info("Dynamic sizing: No ML metrics, using score threshold, selecting %d positions", count)
            return count
        logger.info("Dynamic sizing: No confidence/return/score data, defaulting to 5")
        return 5
    
    # Calculate quality score for each stock
    # Quality = weighted combination of confidence and predicted return percentile
    quality_scores = pd.Series(0.0, index=screened.index)
    
    if has_confidence:
        # Normalize confidence to 0-1 range (it's typically already 0-1)
        conf_normalized = screened["pred_confidence"].clip(0, 1)
        quality_scores += conf_normalized * 0.4  # 40% weight to confidence
    
    if has_return:
        # Normalize predicted return using percentile rank within screened set
        ret_pct = screened["pred_return"].rank(pct=True)
        quality_scores += ret_pct * 0.6  # 60% weight to return (more important)
    
    # Determine quality threshold based on model IC
    # High IC → lower threshold (be more aggressive)
    # Low/negative IC → higher threshold (be conservative)
    if model_ic is not None and model_ic > 0:
        # IC typically ranges 0.01-0.10 for good models
        # Scale threshold: IC=0.10 → threshold=0.4, IC=0.01 → threshold=0.7
        ic_factor = min(1.0, max(0.0, model_ic * 10))  # 0-1 based on IC
        quality_threshold = 0.7 - (ic_factor * 0.3)  # Range: 0.4-0.7
        logger.info("Dynamic sizing: Model IC=%.3f, quality threshold=%.2f", model_ic, quality_threshold)
    else:
        # Conservative default when no IC available
        quality_threshold = 0.6
        logger.info("Dynamic sizing: No model IC, using default threshold=%.2f", quality_threshold)
    
    # Apply hard thresholds first
    qualifying_mask = pd.Series(True, index=screened.index)
    
    if has_confidence:
        qualifying_mask &= (screened["pred_confidence"] >= min_confidence)
    
    if has_return:
        qualifying_mask &= (screened["pred_return"] >= min_pred_return)
    
    # Then apply quality score threshold
    qualifying_mask &= (quality_scores >= quality_threshold)
    
    qualifying_count = int(qualifying_mask.sum())
    
    # Portfolio size: at least 1 (always recommend something), at most max_positions
    dynamic_size = max(1, min(max_positions, qualifying_count))
    
    # Log details
    hard_threshold_count = ((screened.get("pred_confidence", pd.Series(1.0)) >= min_confidence) & 
                            (screened.get("pred_return", pd.Series(1.0)) >= min_pred_return)).sum()
    
    logger.info(
        "Dynamic sizing: %d stocks pass hard thresholds (conf>=%.2f, ret>=%.1f%%), "
        "%d pass quality threshold (>=%.2f), final portfolio size: %d",
        hard_threshold_count, min_confidence, min_pred_return * 100,
        qualifying_count, quality_threshold, dynamic_size
    )
    
    return dynamic_size


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
                models, weights, peak_model = load_ensemble(mp)
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
                    
                    # Predict peak timing (days until optimal sell)
                    max_horizon = model_metadata.get("max_horizon_days", 10) if model_metadata else 10
                    features["pred_peak_days"] = predict_peak_days(
                        peak_model, features_ml, feature_cols=selected_features,
                        min_days=1, max_days=max_horizon
                    )
                    if peak_model:
                        logger.info(
                            "Peak timing predictions: mean=%.1f days, range=[%.0f, %.0f]",
                            features["pred_peak_days"].mean(),
                            features["pred_peak_days"].min(),
                            features["pred_peak_days"].max(),
                        )
                    
                    # Compute return-per-day: ranks stocks by how quickly they spike
                    # pred_return is the predicted peak return (if model trained on peak),
                    # pred_peak_days is the predicted day of peak
                    if "pred_peak_days" in features.columns:
                        safe_days = features["pred_peak_days"].clip(lower=1.0)
                        features["ret_per_day"] = features["pred_return"] / safe_days
                        logger.info(
                            "Return-per-day: mean=%.4f, max=%.4f (optimizing for spike capture)",
                            features["ret_per_day"].mean(),
                            features["ret_per_day"].max(),
                        )
                    
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
    
    # Compute fully dynamic portfolio size based on model metrics and predicted returns
    # No base size - portfolio can range from 1 to max based on opportunity quality
    if getattr(cfg, "dynamic_portfolio_sizing", True):
        # Extract model IC from metadata (used to calibrate aggressiveness)
        model_ic = None
        if run_meta.get("model", {}).get("metadata"):
            holdout_metrics = run_meta["model"]["metadata"].get("holdout", {})
            model_ic = holdout_metrics.get("mean_ic")
            if model_ic is not None:
                logger.info("Model holdout IC: %.4f (used for dynamic sizing)", model_ic)
        
        effective_portfolio_size = compute_dynamic_portfolio_size(
            screened=screened,
            min_confidence=getattr(cfg, "dynamic_size_min_confidence", 0.5),
            min_pred_return=getattr(cfg, "dynamic_size_min_pred_return", 0.03),
            max_positions=getattr(cfg, "dynamic_size_max_positions", 50),
            model_ic=model_ic,
            logger=logger,
        )
        run_meta["dynamic_portfolio_size"] = effective_portfolio_size
    else:
        # Fallback to static portfolio_size when dynamic sizing disabled
        effective_portfolio_size = cfg.portfolio_size
    
    # Compute portfolio weights with optional correlation awareness
    if cfg.use_correlation_weights:
        logger.info("Using correlation-aware risk parity weights")
        target_weights = compute_correlation_aware_weights(
            features=screened,
            prices=prices,  # Need historical prices for covariance
            portfolio_size=effective_portfolio_size,
            weight_cap=cfg.weight_cap,
            logger=logger,
        )
    else:
        target_weights = compute_inverse_vol_weights(
            features=screened,
            portfolio_size=effective_portfolio_size,
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
    
    # ---- Reward model: load log + policy, apply adaptive scaling ----
    reward_log: RewardLog | None = None
    reward_policy: RewardPolicy | None = None
    action_reward_log: ActionRewardLog | None = None
    reward_action: dict[str, float] = {
        "exposure_scalar": 1.0, "conviction_scalar": 1.0,
        "exit_tightness": 1.0, "hold_patience": 1.0,
    }
    if cfg.reward_model_enabled:
        try:
            rlog_path = Path(cfg.cache_dir) / cfg.reward_log_path
            rpol_path = Path(cfg.cache_dir) / cfg.reward_policy_path
            alog_path = Path(cfg.cache_dir) / "action_reward_log.json"
            reward_log = RewardLog.load(rlog_path)
            action_reward_log = ActionRewardLog.load(alog_path)
            reward_policy = RewardPolicy.load(
                rpol_path,
                warmup_days=cfg.reward_warmup_days,
                exposure_min=cfg.reward_exposure_min,
                exposure_max=cfg.reward_exposure_max,
                conviction_min=cfg.reward_conviction_min,
                conviction_max=cfg.reward_conviction_max,
                exit_tightness_min=cfg.reward_exit_tightness_min,
                exit_tightness_max=cfg.reward_exit_tightness_max,
                hold_patience_min=cfg.reward_hold_patience_min,
                hold_patience_max=cfg.reward_hold_patience_max,
                drawdown_penalty=cfg.reward_drawdown_penalty,
            )

            # Back-fill post-action prices and score completed actions
            today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            n_action_bf = action_reward_log.backfill_prices(prices_cad, today_str)
            n_action_scored = score_actions(action_reward_log, window=60)
            if n_action_bf or n_action_scored:
                logger.info(
                    "Action rewards: back-filled %d price fields, scored %d actions",
                    n_action_bf, n_action_scored,
                )
            # Compute and log action quality summary
            action_summary = compute_action_quality_summary(action_reward_log, window=30)
            if action_summary:
                run_meta["action_quality"] = action_summary
                overall = action_summary.get("overall", {})
                logger.info(
                    "Action quality (30d): %d scored, avg_reward=%.4f, positive=%.0f%%",
                    overall.get("total_actions_scored", 0),
                    overall.get("avg_reward", 0),
                    overall.get("positive_action_pct", 0) * 100,
                )

            # Back-fill realized returns from yesterday using today's prices
            n_updated = reward_log.update_realized_returns(prices_cad, date_str=today_str)
            if n_updated:
                logger.info("Reward tracker: back-filled %d realized returns", n_updated)

            # Compute online IC and update policy with yesterday's portfolio return
            online_ic = compute_online_ic(reward_log, window=cfg.reward_ic_window)
            recent_ic = online_ic.get("ensemble_ic") or 0.0
            run_meta["reward_online_ic"] = online_ic

            # Compute portfolio daily return from last two pnl_history entries
            prev_daily_return = 0.0
            if len(state.pnl_history) >= 2:
                eq_prev = float(state.pnl_history[-2].get("equity_cad", 0))
                eq_cur = float(state.pnl_history[-1].get("equity_cad", 0))
                if eq_prev > 0:
                    prev_daily_return = (eq_cur - eq_prev) / eq_prev

            # Build state vector for the policy
            regime_composite = 0.0
            if "market_trend_20d" in features.columns and len(features) > 0:
                regime_composite = float(features["market_trend_20d"].iloc[0])
            avg_confidence = float(screened["pred_confidence"].mean()) if "pred_confidence" in screened.columns and not screened.empty else 0.5
            pred_spread = float(screened["pred_return"].std()) if "pred_return" in screened.columns and not screened.empty else 0.0

            state_vec = build_state_vector(
                portfolio_drawdown=dd_info.get("current_drawdown", 0.0),
                equity_slope_5d=compute_equity_slope(state.pnl_history),
                regime_composite=regime_composite,
                model_avg_confidence=avg_confidence,
                prediction_spread=pred_spread,
                n_positions=len([p for p in state.positions if p.status == "OPEN"]),
                recent_sharpe_5d=compute_recent_sharpe(state.pnl_history),
                reward_ic_recent=recent_ic,
            )

            # Update policy with yesterday's reward (if we have data)
            if reward_policy.state.n_updates > 0 or prev_daily_return != 0.0:
                # Use the last action stored in policy history, or default
                last_action = reward_action
                if reward_policy.state.history:
                    h = reward_policy.state.history[-1]
                    last_action = {
                        "exposure_scalar": h.get("exposure_scalar", h.get("exposure", 1.0)),
                        "conviction_scalar": h.get("conviction_scalar", h.get("conviction", 1.0)),
                        "exit_tightness": h.get("exit_tightness", 1.0),
                        "hold_patience": h.get("hold_patience", 1.0),
                    }
                reward_policy.update(state_vec, last_action, prev_daily_return)

            # Select today's action
            reward_action = reward_policy.select_action(state_vec)

            # Modulate conviction based on recent action quality
            # If many recent actions were bad, dampen conviction
            if action_reward_log is not None:
                aq = compute_action_quality_summary(action_reward_log, window=30)
                overall_aq = aq.get("overall", {})
                positive_pct = overall_aq.get("positive_action_pct", 0.5)
                if overall_aq.get("total_actions_scored", 0) >= 10:
                    # Scale conviction: 0.5 at 0% positive, 1.0 at 50%, up to policy max
                    aq_scalar = max(0.5, min(1.5, positive_pct * 2.0))
                    reward_action["conviction_scalar"] = float(
                        max(cfg.reward_conviction_min,
                            min(cfg.reward_conviction_max,
                                reward_action["conviction_scalar"] * aq_scalar))
                    )

            # Apply exposure scalar to target weights
            exp_s = reward_action["exposure_scalar"]
            if abs(exp_s - 1.0) > 0.01 and not target_weights.empty:
                target_weights["weight"] = target_weights["weight"] * exp_s

            # Apply conviction scalar: amplify/dampen spread between positions
            conv_s = reward_action["conviction_scalar"]
            if abs(conv_s - 1.0) > 0.01 and not target_weights.empty and len(target_weights) > 1:
                mean_w = float(target_weights["weight"].mean())
                target_weights["weight"] = mean_w + (target_weights["weight"] - mean_w) * conv_s
                target_weights["weight"] = target_weights["weight"].clip(lower=1e-4)
                total_w = target_weights["weight"].sum()
                if total_w > 0:
                    target_weights["weight"] = target_weights["weight"] / total_w

            logger.info(
                "Reward policy: exposure=%.2f conviction=%.2f exit_tight=%.2f hold_pat=%.2f (updates=%d)",
                exp_s, conv_s, reward_action["exit_tightness"],
                reward_action["hold_patience"], reward_policy.state.n_updates,
            )

            # Compute prediction bias and log it
            bias_info = compute_prediction_bias(reward_log, window=cfg.reward_ic_window)
            run_meta["reward_prediction_bias"] = bias_info
            run_meta["reward_policy"] = reward_policy.summary()
            run_meta["reward_action"] = reward_action
        except Exception as e:
            logger.warning("Reward model error (non-fatal): %s", e)

    # Safeguard: if all weights were eliminated by the scaling chain
    # (regime + vol targeting + drawdown can compound to push every position
    # below the min-position threshold), rebuild weights for the top picks
    # so the portfolio always has recommendations when screened stocks exist.
    if (target_weights.empty or target_weights["weight"].sum() <= 0) and not screened.empty:
        fallback_n = min(3, len(screened))
        logger.warning(
            "All target weights eliminated after scaling chain; "
            "rebuilding for top %d screened stocks",
            fallback_n,
        )
        target_weights = compute_inverse_vol_weights(
            features=screened,
            portfolio_size=fallback_n,
            weight_cap=cfg.weight_cap,
            logger=logger,
            alpha_col=alpha_col,
        )
        run_meta["target_weights_fallback"] = True
    
    # Scale exit/holding parameters by the bandit's adaptive scalars.
    # exit_tightness > 1 → tighter stops (protective), < 1 → looser stops (patient)
    # hold_patience > 1 → hold longer, < 1 → exit sooner
    exit_t = reward_action.get("exit_tightness", 1.0)
    hold_p = reward_action.get("hold_patience", 1.0)
    adaptive_trailing_dist = getattr(cfg, "trailing_stop_distance_pct", 0.08) / max(exit_t, 0.1)
    adaptive_vol_stop_base = getattr(cfg, "vol_adjusted_stop_base", 0.08) / max(exit_t, 0.1)
    adaptive_vol_stop_min = getattr(cfg, "vol_adjusted_stop_min", 0.04) / max(exit_t, 0.1)
    adaptive_vol_stop_max = getattr(cfg, "vol_adjusted_stop_max", 0.15) / max(exit_t, 0.1)
    adaptive_max_hold = max(1, round(cfg.max_holding_days * hold_p))
    adaptive_max_hold_hard = max(adaptive_max_hold, round(cfg.max_holding_days_hard * hold_p))
    adaptive_quick_profit_pct = getattr(cfg, "quick_profit_pct", 0.05) / max(hold_p, 0.1)
    adaptive_min_daily_return = getattr(cfg, "min_daily_return", 0.005) / max(hold_p, 0.1)
    if abs(exit_t - 1.0) > 0.05 or abs(hold_p - 1.0) > 0.05:
        logger.info(
            "Adaptive PM params: trailing_dist=%.3f vol_stop=%.3f max_hold=%d quick_profit=%.3f",
            adaptive_trailing_dist, adaptive_vol_stop_base,
            adaptive_max_hold, adaptive_quick_profit_pct,
        )

    pm = PortfolioManager(
        state_path=cfg.portfolio_state_path,
        max_holding_days=adaptive_max_hold,
        max_holding_days_hard=adaptive_max_hold_hard,
        extend_hold_min_pred_return=cfg.extend_hold_min_pred_return,
        extend_hold_min_score=cfg.extend_hold_min_score,
        max_positions=effective_portfolio_size,
        stop_loss_pct=cfg.stop_loss_pct,
        take_profit_pct=cfg.take_profit_pct,
        trailing_stop_enabled=getattr(cfg, "trailing_stop_enabled", True),
        trailing_stop_activation_pct=getattr(cfg, "trailing_stop_activation_pct", 0.05),
        trailing_stop_distance_pct=adaptive_trailing_dist,
        peak_based_exit=getattr(cfg, "peak_based_exit", True),
        twr_optimization=getattr(cfg, "twr_optimization", True),
        quick_profit_pct=adaptive_quick_profit_pct,
        quick_profit_days=getattr(cfg, "quick_profit_days", 3),
        min_daily_return=adaptive_min_daily_return,
        momentum_decay_exit=getattr(cfg, "momentum_decay_exit", True),
        signal_decay_exit_enabled=getattr(cfg, "signal_decay_exit_enabled", True),
        signal_decay_threshold=getattr(cfg, "signal_decay_threshold", -0.02),
        dynamic_holding_enabled=getattr(cfg, "dynamic_holding_enabled", True),
        dynamic_holding_vol_scale=getattr(cfg, "dynamic_holding_vol_scale", 0.5),
        vol_adjusted_stop_enabled=getattr(cfg, "vol_adjusted_stop_enabled", True),
        vol_adjusted_stop_base=adaptive_vol_stop_base,
        vol_adjusted_stop_min=adaptive_vol_stop_min,
        vol_adjusted_stop_max=adaptive_vol_stop_max,
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

    # Add pred_return and pred_peak_days to target_weights for email reporting
    if "pred_return" in screened.columns:
        for t in target_weights.index:
            if t in screened.index:
                target_weights.loc[t, "pred_return"] = screened.loc[t, "pred_return"]
    if "pred_peak_days" in screened.columns:
        for t in target_weights.index:
            if t in screened.index:
                target_weights.loc[t, "pred_peak_days"] = screened.loc[t, "pred_peak_days"]
    
    trade_plan = pm.build_trade_plan(
        state=state,
        screened=screened,
        weights=target_weights,
        prices_cad=prices_cad,
        scored=scored,
    )

    # Merge exit-based SELL actions (PEAK_TARGET, STOP_LOSS, etc.) into the
    # trade plan so they appear in the report and email.  Place sells first.
    if exit_actions:
        # Avoid duplicates: build_trade_plan may also generate ROTATION sells
        # for the same tickers that apply_exits already closed.
        existing_sell_tickers = {
            a.ticker for a in trade_plan.actions
            if a.action in ("SELL", "SELL_PARTIAL")
        }
        new_sells = [
            a for a in exit_actions
            if a.ticker not in existing_sell_tickers
        ]
        if new_sells:
            trade_plan = TradePlan(
                actions=new_sells + trade_plan.actions,
                holdings=trade_plan.holdings,
            )

    # Build holdings weights from ALL open positions using the full features
    # DataFrame (not just 'screened'), so positions that have fallen out of
    # today's top-N screening still appear in the portfolio report.
    open_tickers = [p.ticker for p in state.positions if p.status == "OPEN"]
    holdings_features = features.loc[features.index.intersection(open_tickers)].copy()
    # Merge in the score column from scored so all holdings have scores for
    # the report, even tickers that dropped out of the screened top-N.
    if "score" not in holdings_features.columns and "score" in scored.columns:
        holdings_features["score"] = scored["score"].reindex(holdings_features.index)
    holdings_features = holdings_features.sort_values("score" if "score" in holdings_features.columns else "last_close_cad", ascending=False)
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
    # Attach current holdings sizing for reporting (shares + position value). Keep fractional shares.
    shares_by_ticker = {p.ticker: float(p.shares) for p in state.positions if p.status == "OPEN"}
    holdings_weights = holdings_weights.copy()
    holdings_weights["shares"] = [shares_by_ticker.get(str(t), pd.NA) for t in holdings_weights.index.astype(str)]

    render_reports(
        reports_dir=Path(cfg.reports_dir),
        run_meta=run_meta,
        universe_meta=universe.meta,
        screened=screened,
        weights=holdings_weights,
        target_weights=target_weights,
        trade_actions=trade_plan.actions,
        logger=logger,
        portfolio_pnl_history=state.pnl_history,
        fx_usdcad_rate=float(fx.dropna().iloc[-1]) if fx is not None and not fx.dropna().empty else None,
        total_processed=len(features),
    )

    # ---- Reward model: record today's predictions and save ----
    if cfg.reward_model_enabled and reward_log is not None:
        try:
            today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            new_entries: list[RewardEntry] = []

            # Record predictions for all screened tickers
            for t in screened.index:
                pred_ret = float(screened.loc[t, "pred_return"]) if "pred_return" in screened.columns else 0.0
                pred_alpha = float(screened.loc[t, "pred_return_raw"]) if "pred_return_raw" in screened.columns else None
                conf = float(screened.loc[t, "pred_confidence"]) if "pred_confidence" in screened.columns else None
                sc = float(scored.loc[t, "score"]) if t in scored.index and "score" in scored.columns else None
                tw = float(target_weights.loc[t, "weight"]) if t in target_weights.index else None
                px = float(prices_cad.get(t, float("nan")))
                if pd.isna(px) or px <= 0:
                    continue
                new_entries.append(RewardEntry(
                    date=today_str,
                    ticker=str(t),
                    predicted_return=pred_ret,
                    predicted_alpha=pred_alpha,
                    model_score=sc,
                    confidence=conf,
                    weight_assigned=tw,
                    price_at_prediction=px,
                ))

            # Record closed positions from today's exits
            for action in trade_plan.actions:
                if action.action in ("SELL", "SELL_PARTIAL"):
                    pos = next(
                        (p for p in state.positions if p.ticker == action.ticker and p.status != "OPEN"),
                        None,
                    )
                    if pos and pos.entry_price and pos.entry_price > 0 and action.price_cad:
                        cum_ret = (action.price_cad - pos.entry_price) / pos.entry_price
                        # Update or create an entry for the closed trade
                        new_entries.append(RewardEntry(
                            date=today_str,
                            ticker=action.ticker,
                            predicted_return=float(screened.loc[action.ticker, "pred_return"]) if action.ticker in screened.index and "pred_return" in screened.columns else 0.0,
                            realized_cumulative_return=cum_ret,
                            days_held=action.days_held,
                            exit_reason=action.reason,
                            price_at_prediction=pos.entry_price,
                        ))

            if new_entries:
                reward_log.append_batch(new_entries)

            rlog_path = Path(cfg.cache_dir) / cfg.reward_log_path
            reward_log.save(rlog_path)
            logger.info("Reward tracker: logged %d entries (%d total)", len(new_entries), len(reward_log.entries))
        except Exception as e:
            logger.warning("Reward tracker save error (non-fatal): %s", e)

    # ---- Action-level reward logging (with rotation pairs + context) ----
    if cfg.reward_model_enabled and action_reward_log is not None:
        try:
            today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

            # Pre-compute screened context for selection quality
            screened_avg_pred = None
            screened_top_pred = None
            if "pred_return" in screened.columns:
                preds = screened["pred_return"].dropna()
                if len(preds):
                    screened_avg_pred = float(preds.mean())
                    screened_top_pred = float(preds.max())

            # Identify rotation pairs: SELLs and BUYs on the same day
            sells_today = [
                a for a in trade_plan.actions
                if a.action in ("SELL", "SELL_PARTIAL")
                and a.price_cad and not pd.isna(a.price_cad) and float(a.price_cad) > 0
            ]
            buys_today = [
                a for a in trade_plan.actions
                if a.action == "BUY"
                and a.price_cad and not pd.isna(a.price_cad) and float(a.price_cad) > 0
            ]
            # Build rotation linkage: pair sells to buys (order-matched)
            sell_to_buy: dict[str, str] = {}  # sold_ticker -> bought_ticker
            buy_to_sell: dict[str, str] = {}  # bought_ticker -> sold_ticker
            for i, sell_a in enumerate(sells_today):
                if i < len(buys_today):
                    sell_to_buy[sell_a.ticker] = buys_today[i].ticker
                    buy_to_sell[buys_today[i].ticker] = sell_a.ticker

            action_entries: list[ActionRewardEntry] = []

            for action in trade_plan.actions:
                if action.action not in ("BUY", "SELL", "SELL_PARTIAL", "HOLD"):
                    continue
                px = action.price_cad
                if not px or pd.isna(px) or float(px) <= 0:
                    continue
                pred_ret = float(action.pred_return) if action.pred_return is not None else 0.0
                conf = float(screened.loc[action.ticker, "pred_confidence"]) if action.ticker in screened.index and "pred_confidence" in screened.columns else 0.0
                entry_px = float(action.entry_price) if action.entry_price else None

                # Look up stock volatility from features
                stock_vol = None
                if action.ticker in features.index and "vol_20d_ann" in features.columns:
                    v = features.loc[action.ticker, "vol_20d_ann"]
                    if not pd.isna(v):
                        stock_vol = float(v)

                # Rotation pair tracking
                replaced_by = None
                replaced_by_px = None
                replaced = None
                replaced_px = None

                if action.action in ("SELL", "SELL_PARTIAL") and action.ticker in sell_to_buy:
                    rpl_ticker = sell_to_buy[action.ticker]
                    replaced_by = rpl_ticker
                    rpl_px = prices_cad.get(rpl_ticker)
                    if rpl_px and not pd.isna(rpl_px) and float(rpl_px) > 0:
                        replaced_by_px = float(rpl_px)

                if action.action == "BUY" and action.ticker in buy_to_sell:
                    sold_ticker = buy_to_sell[action.ticker]
                    replaced = sold_ticker
                    rpl_px = prices_cad.get(sold_ticker)
                    if rpl_px and not pd.isna(rpl_px) and float(rpl_px) > 0:
                        replaced_px = float(rpl_px)

                action_entries.append(ActionRewardEntry(
                    date=today_str,
                    ticker=action.ticker,
                    action=action.action,
                    reason=action.reason,
                    price_at_action=float(px),
                    shares=float(action.shares),
                    predicted_return=pred_ret,
                    confidence=conf,
                    entry_price=entry_px,
                    replaced_by_ticker=replaced_by,
                    replaced_by_price=replaced_by_px,
                    replaced_ticker=replaced,
                    replaced_price=replaced_px,
                    screened_avg_pred_return=screened_avg_pred,
                    screened_top_pred_return=screened_top_pred,
                    stock_volatility=stock_vol,
                ))

            if action_entries:
                action_reward_log.append_batch(action_entries)

            alog_path = Path(cfg.cache_dir) / "action_reward_log.json"
            action_reward_log.save(alog_path)
            n_by_type: dict[str, int] = {}
            for ae in action_entries:
                n_by_type[ae.action] = n_by_type.get(ae.action, 0) + 1
            n_rotations = len(sell_to_buy)
            logger.info(
                "Action reward tracker: logged %d actions %s, %d rotation pairs (%d total)",
                len(action_entries), dict(n_by_type), n_rotations,
                len(action_reward_log.entries),
            )
        except Exception as e:
            logger.warning("Action reward tracker save error (non-fatal): %s", e)

    if cfg.reward_model_enabled and reward_policy is not None:
        try:
            rpol_path = Path(cfg.cache_dir) / cfg.reward_policy_path
            reward_policy.save(rpol_path)
        except Exception as e:
            logger.warning("Reward policy save error (non-fatal): %s", e)

    # Persist metadata for debugging/auditing
    write_json(cache_dir / "last_run_meta.json", run_meta)
    logger.info("Wrote reports to %s", reports_dir.resolve())
