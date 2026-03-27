from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_screener.config import Config
from stock_screener.data.fundamentals import fetch_fundamentals
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.macro import fetch_macro_indicators
from stock_screener.data.prices import download_price_history
from stock_screener.features.technical import compute_features, apply_target_encodings
from stock_screener.optimization.risk_parity import compute_adaptive_vol_target, compute_inverse_vol_weights, compute_hrp_weights, compute_correlation_aware_weights, optimize_unified_portfolio, apply_confidence_weighting, apply_volatility_targeting, apply_conviction_sizing, apply_liquidity_adjustment, apply_correlation_limits, apply_beta_adjustment, apply_min_position_filter, apply_max_position_cap, apply_regime_exposure
from stock_screener.reporting.render import render_reports
from stock_screener.screening.screener import score_universe, select_sector_neutral, apply_entry_filters
from stock_screener.universe.tsx import fetch_tsx_universe
from stock_screener.universe.us import fetch_us_universe
from stock_screener.utils import Universe, ensure_dir, read_json, write_json, suppress_external_warnings

# Suppress known external library warnings
suppress_external_warnings()
from stock_screener.modeling.model import (
    compute_regime_gate_weights,
    load_regime_ensembles,
    load_ensemble,
    load_model,
    load_quantile_ensembles,
    predict,
    predict_ensemble,
    predict_ensemble_with_uncertainty,
    predict_peak_days,
    predict_regime_gated,
    predict_quantile_lcb,
    compute_feature_schema_hash,
)
from stock_screener.modeling.transform import normalize_features_cross_section, calibrate_predictions
from stock_screener.portfolio.manager import PortfolioManager, TradeAction, TradePlan
from stock_screener.portfolio.state import (
    append_portfolio_events,
    compute_drawdown_scalar,
    load_portfolio_state,
    resolve_portfolio_event_log_path,
    save_portfolio_state,
)
from stock_screener.reward.tracker import RewardEntry, RewardLog, ActionRewardEntry, ActionRewardLog
from stock_screener.reward.feedback import (
    compute_online_ic, compute_ensemble_reward_weights, compute_prediction_bias,
    score_actions, compute_action_quality_summary,
)
from stock_screener.reward.policy import (
    RewardPolicy, build_state_vector, compute_equity_slope, compute_recent_sharpe,
)


# ---------------------------------------------------------------------------
# Extracted helpers — canonical sources in sibling modules.
# Re-exported here for backward compatibility with existing imports/tests.
# ---------------------------------------------------------------------------
from stock_screener.pipeline.helpers import (  # noqa: E402
    compute_dynamic_portfolio_size,
    _lookup_frame_metric,
    _lookup_metric_from_sources,
    _extract_model_holdout_ic,
    _check_runtime_budget,
    _series_distribution_stats,
    _compute_ret_per_day_signal,
    _compute_effective_entry_thresholds,
    _infer_instrument_type_row,
    _infer_instrument_types,
    _apply_instrument_sleeve_constraints,
    _validate_feature_parity,
    _apply_uncertainty_weighting,
    _compute_exposure_metrics,
    _weights_to_series,
    _compute_weight_distance_metrics,
    _enforce_exposure_policy,
)
from stock_screener.pipeline.rebalance import _apply_rebalance_controls  # noqa: E402
from stock_screener.pipeline.state_utils import (  # noqa: E402
    _compute_open_position_values,
    _build_hold_only_target_weights,
    _resolve_portfolio_state_path,
    _build_price_history_cad,
    _sanitize_trade_actions,
    _position_changing_actions,
    _trade_actions_to_event_payloads,
    _append_pnl_snapshot,
    _persist_state_transition_or_fail,
    _check_kill_switch,
)


def run_daily(cfg: Config, logger) -> None:
    """Run the daily screener + weights + reporting pipeline."""

    started_utc = datetime.now(tz=timezone.utc)
    if _check_kill_switch(logger):
        return
    state_path = _resolve_portfolio_state_path(cfg.portfolio_state_path)
    event_log_path = resolve_portfolio_event_log_path(state_path)
    cache_dir = ensure_dir(cfg.cache_dir)
    data_cache_dir = ensure_dir(cfg.data_cache_dir)
    reports_dir = ensure_dir(cfg.reports_dir)
    ensure_dir(reports_dir / "debug")

    run_meta: dict[str, Any] = {
        "run_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config": asdict(cfg),
        "portfolio_state_path": str(state_path),
        "portfolio_event_log_path": str(event_log_path),
    }
    previous_run_meta: dict[str, Any] | None = None
    prev_meta_path = cache_dir / "last_run_meta.json"
    if prev_meta_path.is_file():
        try:
            loaded_prev = read_json(prev_meta_path)
            if isinstance(loaded_prev, dict):
                previous_run_meta = loaded_prev
                run_meta["previous_run_meta_loaded"] = True
        except Exception as e:
            logger.warning("Could not load previous run metadata (%s): %s", prev_meta_path, e)
    if "previous_run_meta_loaded" not in run_meta:
        run_meta["previous_run_meta_loaded"] = False
    ml_expected = bool(getattr(cfg, "use_ml", False))
    allow_baseline_trading = bool(getattr(cfg, "allow_baseline_trading", False))
    ml_available = False
    ml_failure_reason: str | None = None

    us = fetch_us_universe(cfg=cfg, cache_dir=cache_dir, logger=logger)
    tsx = fetch_tsx_universe(cfg=cfg, cache_dir=cache_dir, logger=logger)
    _check_runtime_budget(started_utc, cfg, logger, "universe")

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
    _check_runtime_budget(started_utc, cfg, logger, "price_download")

    fundamentals = fetch_fundamentals(
        tickers=universe.tickers,
        cache_dir=data_cache_dir,
        cache_ttl_days=cfg.fundamentals_cache_ttl_days,
        logger=logger,
    )
    macro = fetch_macro_indicators(
        lookback_days=max(cfg.feature_lookback_days, cfg.liquidity_lookback_days),
        logger=logger,
    )
    features = compute_features(
        prices=prices,
        fx_usdcad=fx,
        liquidity_lookback_days=cfg.liquidity_lookback_days,
        feature_lookback_days=cfg.feature_lookback_days,
        logger=logger,
        fundamentals=fundamentals,
        macro=macro,
    )
    _check_runtime_budget(started_utc, cfg, logger, "feature_build")

    # ── News sentiment + insider features (fail-soft) ────────────
    try:
        from stock_screener.data.news import fetch_news_sentiment_features
        _news_tickers = list(features.index[:cfg.top_n])  # Top screened only (not all 4000)
        news_df = fetch_news_sentiment_features(
            _news_tickers, cache_dir=Path(cfg.cache_dir), logger=logger,
        )
        if not news_df.empty:
            for col in ["news_sentiment_avg", "news_volume_5d", "news_sentiment_pos_ratio"]:
                if col in news_df.columns:
                    features[col] = news_df[col].reindex(features.index)
            logger.info("Merged news sentiment features: %d tickers", len(news_df))
    except Exception as e:
        logger.warning("News sentiment failed (continuing without): %s", e)
    for _nc in ["news_sentiment_avg", "news_volume_5d", "news_sentiment_pos_ratio"]:
        if _nc not in features.columns:
            features[_nc] = float("nan")

    try:
        from stock_screener.data.insiders import fetch_insider_features
        _insider_tickers = list(features.index[:cfg.top_n])
        insider_df = fetch_insider_features(
            _insider_tickers, cache_dir=Path(cfg.cache_dir), logger=logger,
        )
        if not insider_df.empty:
            for col in ["insider_net_buys_90d", "insider_buy_ratio_90d", "insider_activity_recency"]:
                if col in insider_df.columns:
                    features[col] = insider_df[col].reindex(features.index)
            logger.info("Merged insider features: %d tickers", len(insider_df))
    except Exception as e:
        logger.warning("Insider features failed (continuing without): %s", e)
    for _ic in ["insider_net_buys_90d", "insider_buy_ratio_90d", "insider_activity_recency"]:
        if _ic not in features.columns:
            features[_ic] = float("nan")

    _check_runtime_budget(started_utc, cfg, logger, "news_insider_features")

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
                expected_schema_hash = model_metadata.get("feature_schema_hash")
                if selected_features and expected_schema_hash:
                    actual_schema_hash = compute_feature_schema_hash(list(selected_features))
                    if str(actual_schema_hash) != str(expected_schema_hash):
                        raise RuntimeError(
                            "Model metadata schema hash mismatch; artifact may be corrupted "
                            f"(expected={expected_schema_hash}, actual={actual_schema_hash})"
                        )
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
            missing_schema = _validate_feature_parity(
                features_ml,
                selected_features,
                strict=bool(getattr(cfg, "strict_feature_parity", True)),
                logger=logger,
            )
            if missing_schema:
                run_meta["feature_schema_missing"] = missing_schema
            
            if mp.name.lower() == "manifest.json":
                models, weights, peak_model = load_ensemble(mp)
                quantile_ensembles = {}
                regime_ensembles = {}
                if bool(getattr(cfg, "quantile_models_enabled", True)):
                    try:
                        quantile_ensembles = load_quantile_ensembles(mp)
                    except Exception as qe:
                        logger.warning("Could not load quantile ensembles: %s", qe)
                if bool(getattr(cfg, "regime_specialist_enabled", True)):
                    try:
                        regime_ensembles = load_regime_ensembles(mp)
                    except Exception as re:
                        logger.warning("Could not load regime specialists: %s", re)
                if models:
                    # Use uncertainty-aware predictions with selected features
                    pred_df = predict_ensemble_with_uncertainty(
                        models, weights, features_ml, feature_cols=selected_features
                    )
                    raw_preds = pred_df["pred_return"]
                    regime_gate_used = False

                    if regime_ensembles:
                        try:
                            gate_weights = compute_regime_gate_weights(features_ml)
                            regime_preds: dict[str, pd.Series] = {}
                            for regime_name, (r_models, r_weights) in regime_ensembles.items():
                                regime_preds[regime_name] = predict_ensemble(
                                    r_models,
                                    r_weights,
                                    features_ml,
                                    feature_cols=selected_features,
                                )
                            gate_blend = float(getattr(cfg, "regime_gating_base_blend", 0.25))
                            gated_df = predict_regime_gated(
                                raw_preds,
                                regime_preds=regime_preds,
                                gate_weights=gate_weights,
                                base_blend=gate_blend,
                            )
                            raw_preds = gated_df["pred_return"]
                            for c in (
                                "regime",
                                "regime_confidence",
                                "regime_gate_bull",
                                "regime_gate_neutral",
                                "regime_gate_bear",
                                "pred_return_regime_mix",
                                "pred_return_base",
                            ):
                                features[c] = gated_df[c]
                            run_meta["regime_gating"] = {
                                "enabled": True,
                                "n_experts": len(regime_ensembles),
                                "base_blend": gate_blend,
                                "dominant_regime": str(gated_df["regime"].mode().iloc[0]) if len(gated_df) else "neutral",
                                "mean_confidence": float(gated_df["regime_confidence"].mean()) if len(gated_df) else float("nan"),
                            }
                            regime_gate_used = True
                            logger.info(
                                "Applied regime specialist gating: experts=%d, dominant=%s, confidence=%.3f",
                                len(regime_ensembles),
                                run_meta["regime_gating"]["dominant_regime"],
                                run_meta["regime_gating"]["mean_confidence"],
                            )
                        except Exception as ge:
                            logger.warning("Regime specialist gating failed; falling back to base ensemble: %s", ge)
                    
                    # Apply prediction calibration if available
                    recalib_payload = model_metadata.get("prediction_recalibration") if model_metadata else None
                    if (
                        bool(getattr(cfg, "apply_prediction_recalibration", True))
                        and isinstance(recalib_payload, dict)
                        and bool(recalib_payload.get("enabled", False))
                    ):
                        try:
                            slope = float(recalib_payload.get("slope", 1.0))
                            intercept = float(recalib_payload.get("intercept", 0.0))
                            recalibrated = (raw_preds * slope) + intercept
                            features["pred_return_linear_recal"] = recalibrated
                            raw_preds = recalibrated
                            logger.info(
                                "Applied linear prediction recalibration: y=%.4f*x + %.4f",
                                slope,
                                intercept,
                            )
                        except Exception as re:
                            logger.warning("Prediction recalibration payload invalid; skipping: %s", re)

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
                    if "pred_return_base" not in features.columns:
                        features["pred_return_base"] = features["pred_return"]

                    # Optional quantile + LCB override.
                    if quantile_ensembles:
                        lcb_lambda = float(getattr(cfg, "lcb_risk_aversion", 0.5))
                        q_df = predict_quantile_lcb(
                            quantile_ensembles,
                            features_ml,
                            feature_cols=selected_features,
                            lcb_risk_aversion=lcb_lambda,
                        )
                        for c in q_df.columns:
                            features[c] = q_df[c]

                        # If regime specialists are active, keep their alpha signal and use quantiles
                        # for uncertainty only. Otherwise, preserve existing LCB-as-primary behavior.
                        if not regime_gate_used:
                            features["pred_return"] = q_df["pred_return_lcb"]

                        # Quantile spread is a more interpretable uncertainty proxy than model disagreement.
                        features["pred_uncertainty"] = q_df["pred_quantile_spread"]
                        features["pred_confidence"] = 1.0 / (
                            1.0 + features["pred_uncertainty"].fillna(features["pred_uncertainty"].median()).clip(lower=0.0)
                        )
                        if regime_gate_used:
                            logger.info(
                                "Quantile spread attached (regime-gated alpha retained): lambda=%.2f, spread mean=%.4f",
                                lcb_lambda,
                                float(features["pred_quantile_spread"].mean()),
                            )
                        else:
                            logger.info(
                                "Using quantile LCB signal: lambda=%.2f, spread mean=%.4f",
                                lcb_lambda,
                                float(features["pred_quantile_spread"].mean()),
                            )
                    
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
                    
                    # Compute guarded return-per-day signal from predicted return
                    # and predicted peak day. This applies clamp + smoothing and
                    # emits distribution diagnostics + shift alerts.
                    ret_per_day_info = _compute_ret_per_day_signal(
                        features,
                        cfg=cfg,
                        logger=logger,
                        previous_run_meta=previous_run_meta,
                    )
                    run_meta["ret_per_day_signal"] = ret_per_day_info
                    if bool(ret_per_day_info.get("shift_alert_triggered", False)):
                        alerts = run_meta.setdefault("alerts", [])
                        if isinstance(alerts, list):
                            alerts.append(
                                {
                                    "type": "ret_per_day_distribution_shift",
                                    "details": ret_per_day_info.get("shift_alerts", []),
                                }
                            )
                    
                    logger.info(
                        "ML predictions: mean=%.4f, confidence range=[%.3f, %.3f]",
                        features["pred_return"].mean(),
                        pred_df["pred_confidence"].min(),
                        pred_df["pred_confidence"].max(),
                    )
                    logger.info("Loaded ML regressor ensemble from %s (%s members)", cfg.model_path, len(models))
                    ml_available = True
                else:
                    ml_failure_reason = "empty_ensemble"
                    logger.warning(
                        "ML manifest loaded but ensemble contains no models; treating ML as unavailable.",
                    )
            else:
                model = load_model(cfg.model_path)
                logger.info("Loaded ML model from %s", cfg.model_path)
                features["pred_return"] = predict(model, features_ml, feature_cols=selected_features)
                ml_available = True
        except Exception as e:
            ml_failure_reason = str(e)
            logger.warning("ML enabled but model could not be loaded/used: %s", e)

    if ml_expected and not ml_available:
        # Last-resort availability check in case predictions were injected by an
        # alternative ML path without toggling the explicit flag above.
        try:
            ml_available = bool(
                "pred_return" in features.columns
                and pd.to_numeric(features["pred_return"], errors="coerce").notna().any()
            )
        except Exception:
            ml_available = False

    if ml_expected:
        if ml_available:
            strategy_mode = "ML"
            strategy_reason = "ml_inference_available"
        elif allow_baseline_trading:
            strategy_mode = "BASELINE"
            strategy_reason = "ml_unavailable_baseline_allowed"
            logger.warning(
                "ML expected but unavailable (%s); ALLOW_BASELINE_TRADING=1, continuing in BASELINE mode.",
                ml_failure_reason or "unknown_reason",
            )
        else:
            strategy_mode = "HOLD_ONLY"
            strategy_reason = "ml_unavailable_hold_only"
            logger.warning(
                "ML expected but unavailable (%s); entering HOLD_ONLY mode (no new buys).",
                ml_failure_reason or "unknown_reason",
            )
    else:
        strategy_mode = "BASELINE"
        strategy_reason = "ml_disabled"

    run_meta["strategy_mode"] = {
        "mode": strategy_mode,
        "reason": strategy_reason,
        "ml_expected": ml_expected,
        "ml_available": ml_available,
        "allow_baseline_trading": allow_baseline_trading,
        "ml_failure_reason": ml_failure_reason,
    }

    scored = score_universe(
        features=features,
        min_price_cad=cfg.min_price_cad,
        min_avg_dollar_volume_cad=cfg.min_avg_dollar_volume_cad,
        logger=logger,
        max_volatility=getattr(cfg, "max_screen_volatility", None),
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

    # Apply entry confirmation filters with optional dynamic threshold relaxation.
    entry_filter_stats = {}
    entry_thresholds = _compute_effective_entry_thresholds(
        screened,
        cfg=cfg,
        logger=logger,
    )
    if bool(entry_thresholds.get("hold_only_recommended", False)) and strategy_mode != "HOLD_ONLY":
        strategy_mode = "HOLD_ONLY"
        strategy_reason = "entry_stress_guard_hold_only"
        logger.warning(
            "Entry stress guard escalated strategy mode to HOLD_ONLY (vol/breadth stress).",
        )
    if isinstance(run_meta.get("strategy_mode"), dict):
        run_meta["strategy_mode"]["mode"] = strategy_mode
        run_meta["strategy_mode"]["reason"] = strategy_reason
        run_meta["strategy_mode"]["entry_stress_guard_triggered"] = bool(
            entry_thresholds.get("stress_guard_triggered", False)
        )
        run_meta["strategy_mode"]["entry_stress_hold_only_recommended"] = bool(
            entry_thresholds.get("hold_only_recommended", False)
        )
    screened, entry_filter_stats = apply_entry_filters(
        screened,
        min_confidence=entry_thresholds.get("min_confidence"),
        min_pred_return=entry_thresholds.get("min_pred_return"),
        max_volatility=getattr(cfg, "entry_max_volatility", None),
        min_momentum_5d=getattr(cfg, "entry_min_momentum_5d", None),
        momentum_alignment=getattr(cfg, "entry_momentum_alignment", True),
        logger=logger,
    )
    if (
        entry_thresholds.get("dynamic_applied")
        or entry_thresholds.get("stress_guard_triggered")
        or entry_thresholds.get("dynamic_blocked_reasons")
    ):
        run_meta["entry_thresholds"] = entry_thresholds
    if entry_filter_stats.get("rejected_count", 0) > 0:
        run_meta["entry_filters"] = entry_filter_stats
    _check_runtime_budget(started_utc, cfg, logger, "screening")

    # ── LLM trading agent layer (fail-soft) ──────────────────────
    if cfg.llm_agent_enabled and not screened.empty:
        try:
            from stock_screener.agents.trading_agent import (
                analyze_candidates, blend_llm_scores, build_agent_candidates, build_portfolio_context,
            )
            from stock_screener.agents.config import get_agent_config as _get_agent_config

            _agent_cfg = _get_agent_config()
            _has_key = bool(_agent_cfg.get("api_key"))
            logger.info(
                "LLM agent: provider=%s, model=%s, api_key=%s, candidates=%d",
                _agent_cfg.get("provider"), _agent_cfg.get("model"),
                "set" if _has_key else "MISSING", min(len(screened), cfg.dynamic_size_max_positions),
            )

            if not _has_key:
                run_meta["llm_agent"] = {"status": "skipped", "reason": "no API key configured"}
                logger.warning("LLM agent: GROQ_API_KEY not set; skipping")
            else:
                from stock_screener.data.news import fetch_ticker_news
                _llm_tickers = list(screened.index[:cfg.dynamic_size_max_positions])
                _news_by_ticker: dict[str, list[dict]] = {}
                for _nt in _llm_tickers:
                    try:
                        _news_by_ticker[str(_nt)] = fetch_ticker_news(str(_nt), logger=logger)[:5]
                    except Exception:
                        _news_by_ticker[str(_nt)] = []

                _agent_candidates = build_agent_candidates(
                    screened, max_tickers=cfg.dynamic_size_max_positions,
                    news_by_ticker=_news_by_ticker,
                )

                from stock_screener.portfolio.state import load_portfolio_state as _load_ps
                _early_state = _load_ps(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
                _portfolio_ctx = build_portfolio_context(
                    _early_state.positions, _early_state.cash_cad,
                )

                _llm_pm = bool(getattr(cfg, "llm_decision_primary", False))
                _decisions = analyze_candidates(
                    _agent_candidates, portfolio_context=_portfolio_ctx, log=logger,
                    primary_mode=_llm_pm,
                )

                if _decisions:
                    screened = blend_llm_scores(
                        screened, _decisions, score_col="score",
                        ml_weight=cfg.llm_agent_ml_weight, llm_weight=cfg.llm_agent_llm_weight,
                        log=logger,
                    )
                    screened = screened.sort_values("score", ascending=False)
                    run_meta["llm_agent"] = {
                        "status": "success",
                        "n_analyzed": len(_decisions),
                        "decisions": {
                            t: {
                                "rating": d.rating,
                                "score": d.score,
                                "reasoning": d.reasoning,
                                "bull_thesis": d.bull_thesis,
                                "bear_thesis": d.bear_thesis,
                                "risk_assessment": d.risk_assessment,
                                "debate_rounds": getattr(d, "debate_rounds", 1),
                                "debate_history": getattr(d, "debate_history", []),
                                "risk_debate": getattr(d, "risk_debate", None),
                                "analyst_reports": getattr(d, "analyst_reports", None),
                                "position_size": getattr(d, "position_size", "MEDIUM"),
                                "target_weight": getattr(d, "target_weight", None),
                                "suggested_stop_loss": getattr(d, "suggested_stop_loss", None),
                                "expected_hold_days": getattr(d, "expected_hold_days", None),
                            }
                            for t, d in _decisions.items()
                        },
                    }
                    logger.info("LLM agent: blended scores for %d tickers", len(_decisions))
                else:
                    run_meta["llm_agent"] = {"status": "no_results", "reason": "API returned no decisions (check logs for warnings)"}
                    logger.warning("LLM agent: analyze_candidates returned empty")
        except Exception as e:
            run_meta["llm_agent"] = {"status": "error", "reason": str(e)}
            logger.warning("LLM agent layer failed (continuing without): %s", e)
        _check_runtime_budget(started_utc, cfg, logger, "llm_agent")

        # ── Phase 4: Portfolio-level LLM reasoning (fail-soft) ────────
        if cfg.agent_portfolio_reasoning and _decisions and not screened.empty:
            try:
                from stock_screener.agents.trading_agent import analyze_portfolio
                _market_cond = {
                    "vol_regime": float(screened["market_vol_regime"].iloc[0]) if "market_vol_regime" in screened.columns and len(screened) > 0 else 1.0,
                    "market_trend": float(screened["market_trend_20d"].iloc[0]) if "market_trend_20d" in screened.columns and len(screened) > 0 else 0.0,
                    "breadth": float(screened["market_breadth"].iloc[0]) if "market_breadth" in screened.columns and len(screened) > 0 else 0.5,
                }
                _portfolio_llm = analyze_portfolio(
                    _agent_candidates, _decisions, _portfolio_ctx,
                    market_conditions=_market_cond,
                    max_positions=cfg.dynamic_size_max_positions,
                    budget_cad=cfg.portfolio_budget_cad,
                    log=logger,
                    primary_mode=_llm_pm,
                )
                if _portfolio_llm:
                    if isinstance(run_meta.get("llm_agent"), dict):
                        run_meta["llm_agent"]["portfolio_reasoning"] = _portfolio_llm
                    # Apply score adjustments if LLM suggests weight changes
                    adj_raw = _portfolio_llm.get("adjustments", "")
                    if adj_raw and adj_raw.upper() != "NONE" and "score" in screened.columns:
                        logger.info("Portfolio LLM adjustments: %s", adj_raw)
                    logger.info("Portfolio reasoning complete: %s", _portfolio_llm.get("overall", ""))
            except Exception as e:
                logger.warning("Portfolio LLM reasoning failed (continuing): %s", e)

    elif cfg.llm_agent_enabled and screened.empty:
        run_meta["llm_agent"] = {"status": "skipped", "reason": "no screened tickers"}
    elif not cfg.llm_agent_enabled:
        run_meta["llm_agent"] = {"status": "disabled"}

    # ── LLM-primary decision mode: LLM drives ticker selection + weighting ──
    _llm_primary_mode = bool(getattr(cfg, "llm_decision_primary", False))
    _llm_primary_success = False

    if _llm_primary_mode and cfg.llm_agent_enabled and not screened.empty:
        _llm_agent_status = run_meta.get("llm_agent", {}).get("status") if isinstance(run_meta.get("llm_agent"), dict) else None
        if _llm_agent_status == "success" and _decisions:
            try:
                from stock_screener.agents.trading_agent import (
                    select_tickers_llm_primary as _select_llm,
                    compute_llm_primary_weights as _compute_llm_w,
                    apply_portfolio_reasoning_enforced as _enforce_pr,
                )
                _llm_selected = _select_llm(screened, _decisions, max_positions=cfg.dynamic_size_max_positions, log=logger)
                if _llm_selected:
                    target_weights = _compute_llm_w(
                        _llm_selected, _decisions, screened,
                        small_range=(cfg.llm_weight_small_min, cfg.llm_weight_small_max),
                        medium_range=(cfg.llm_weight_medium_min, cfg.llm_weight_medium_max),
                        full_range=(cfg.llm_weight_full_min, cfg.llm_weight_full_max),
                        max_position_pct=float(getattr(cfg, "max_position_pct", 0.20)),
                        min_position_pct=float(getattr(cfg, "min_position_pct", 0.02)),
                        log=logger,
                    )
                    # Enforce portfolio reasoning if available
                    _pr = run_meta.get("llm_agent", {}).get("portfolio_reasoning")
                    if cfg.llm_portfolio_enforce and isinstance(_pr, dict):
                        target_weights = _enforce_pr(
                            target_weights, _pr, screened,
                            sector_cap=cfg.llm_sector_cap,
                            regime_reduce_scalar=cfg.llm_regime_reduce_scalar,
                            max_position_pct=float(getattr(cfg, "max_position_pct", 0.20)),
                            log=logger,
                        )
                    effective_portfolio_size = len(target_weights)
                    _llm_primary_success = True
                    run_meta["llm_primary"] = {
                        "mode": "active",
                        "selected_tickers": _llm_selected,
                        "n_selected": len(_llm_selected),
                    }
                    logger.info("LLM-primary mode ACTIVE: %d tickers selected, weights assigned by LLM", len(_llm_selected))
                else:
                    logger.warning("LLM-primary: no BUY/OVERWEIGHT tickers; falling back to ML pipeline")
            except Exception as e:
                logger.warning("LLM-primary decision failed (%s); falling back to ML pipeline", e)
                run_meta["llm_primary"] = {"mode": "fallback", "reason": str(e)}
        else:
            logger.warning("LLM-primary: LLM analysis not successful (status=%s); falling back to ML", _llm_agent_status)
            run_meta["llm_primary"] = {"mode": "fallback", "reason": f"LLM status: {_llm_agent_status}"}
    elif _llm_primary_mode:
        run_meta["llm_primary"] = {"mode": "fallback", "reason": "LLM not enabled or screened empty"}

    # ── ML-primary fallback (runs when LLM-primary is off or failed) ──────
    # When LLM-primary succeeded, skip the entire ML weight computation block.
    # The quantitative guardrails (regime, vol targeting, drawdown) still run
    # on LLM-assigned target_weights downstream.
    if _llm_primary_success:
        alpha_col = "score"
        # target_weights and effective_portfolio_size already set by LLM-primary
    elif screened.empty:
        alpha_col = "pred_return" if "pred_return" in screened.columns else "score"
        logger.warning("No screened tickers remain after entry filters; skipping new-entry weight construction.")
        effective_portfolio_size = 0
        target_weights = screened.copy()
        if "weight" not in target_weights.columns:
            target_weights["weight"] = pd.Series(dtype=float)
    else:
        alpha_col = "pred_return" if "pred_return" in screened.columns else "score"
        # Compute fully dynamic portfolio size based on model metrics and predicted returns
        # No base size - portfolio can range from 1 to max based on opportunity quality
        if getattr(cfg, "dynamic_portfolio_sizing", True):
            # Extract model IC from metadata (used to calibrate aggressiveness)
            model_meta = run_meta.get("model", {}).get("metadata")
            model_ic = _extract_model_holdout_ic(model_meta)
            if model_ic is not None:
                logger.info("Model holdout IC: %.4f (used for dynamic sizing)", model_ic)

            dyn_min_conf = float(getattr(cfg, "dynamic_size_min_confidence", 0.5))
            dyn_min_pred = float(getattr(cfg, "dynamic_size_min_pred_return", 0.01))
            if entry_thresholds.get("min_confidence") is not None:
                dyn_min_conf = min(dyn_min_conf, float(entry_thresholds["min_confidence"]))
            if entry_thresholds.get("min_pred_return") is not None:
                dyn_min_pred = min(dyn_min_pred, float(entry_thresholds["min_pred_return"]))

            effective_portfolio_size = compute_dynamic_portfolio_size(
                screened=screened,
                min_confidence=dyn_min_conf,
                min_pred_return=dyn_min_pred,
                max_positions=getattr(cfg, "dynamic_size_max_positions", 50),
                model_ic=model_ic,
                logger=logger,
            )
            run_meta["dynamic_portfolio_size"] = effective_portfolio_size
            run_meta["dynamic_sizing_thresholds"] = {
                "min_confidence": dyn_min_conf,
                "min_pred_return": dyn_min_pred,
            }
        else:
            # Fallback to static portfolio_size when dynamic sizing disabled
            effective_portfolio_size = cfg.portfolio_size

        # Compute portfolio weights with optional correlation awareness
        if getattr(cfg, "use_hrp_weights", False):
            from stock_screener.optimization.risk_parity import SCIPY_AVAILABLE as _SCIPY_OK
            if _SCIPY_OK:
                logger.info("Using Hierarchical Risk Parity (HRP) weights")
                # Build returns DataFrame from prices for HRP tickers
                _hrp_tickers = list(screened["ticker"].unique()) if "ticker" in screened.columns else list(screened.index)
                _hrp_prices = prices[[t for t in _hrp_tickers if t in prices.columns]]
                _hrp_returns = _hrp_prices.pct_change().dropna(how="all")
                _hrp_alpha = None
                if alpha_col and alpha_col in screened.columns:
                    _idx = screened["ticker"] if "ticker" in screened.columns else screened.index
                    _hrp_alpha = pd.Series(screened[alpha_col].values, index=_idx)
                hrp_w = compute_hrp_weights(
                    returns=_hrp_returns,
                    logger=logger,
                    alpha_series=_hrp_alpha,
                    weight_cap=cfg.weight_cap,
                )
                # Build target_weights DataFrame matching inverse-vol format
                _top = screened.head(effective_portfolio_size).copy()
                _tkrs = _top["ticker"].values if "ticker" in _top.columns else _top.index.values
                _top["weight"] = [float(hrp_w.get(t, 0.0)) for t in _tkrs]
                # Re-normalize to selected positions only
                _wsum = _top["weight"].sum()
                if _wsum > 0:
                    _top["weight"] = _top["weight"] / _wsum
                target_weights = _top.sort_values("weight", ascending=False)
            else:
                logger.warning("HRP requested but scipy unavailable; falling back to inverse-vol")
                target_weights = compute_inverse_vol_weights(
                    features=screened,
                    portfolio_size=effective_portfolio_size,
                    weight_cap=cfg.weight_cap,
                    logger=logger,
                    alpha_col=alpha_col,
                )
        elif cfg.use_correlation_weights:
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
    
    # Load portfolio state once before optimization so optimizer and execution
    # share the same migrated/current holdings view.
    state = load_portfolio_state(
        state_path, initial_cash_cad=cfg.portfolio_budget_cad
    )
    run_meta["portfolio_state_loaded"] = {
        "path": str(state_path),
        "open_positions": int(len([p for p in state.positions if p.status == "OPEN"])),
        "closed_positions": int(len([p for p in state.positions if p.status != "OPEN"])),
        "last_updated_utc": (
            state.last_updated.isoformat()
            if getattr(state, "last_updated", None) is not None
            else None
        ),
    }
    if getattr(state, "last_updated", None) is not None:
        try:
            staleness_hours = (
                datetime.now(tz=timezone.utc) - state.last_updated
            ).total_seconds() / 3600.0
            run_meta["portfolio_state_loaded"]["staleness_hours"] = float(staleness_hours)
            if staleness_hours > 36 and state.positions:
                logger.warning(
                    "Portfolio state appears stale (last_updated=%s, staleness=%.1fh). "
                    "Cross-run persistence may be misconfigured.",
                    state.last_updated.isoformat(),
                    staleness_hours,
                )
        except Exception:
            pass
    # Migration safeguard:
    # Earlier versions created a state file with a large default cash balance and used `shares=1` placeholders,
    # without debiting cash on buys. If we now run with a small configured budget (e.g., 500 CAD), the cached
    # state would show misleading "cash" and P&L. Detect this legacy pattern and reset once so accounting is sane.
    try:
        budget = float(cfg.portfolio_budget_cad)
        open_positions = [p for p in state.positions if p.status == "OPEN"]
        legacy_placeholder = bool(open_positions) and all(
            abs(float(p.shares) - 1.0) < 1e-9 for p in open_positions
        )
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
            save_portfolio_state(state_path, state)
    except Exception as e:
        logger.warning("Could not evaluate/reset legacy portfolio state: %s", e)

    px_now = (
        features["last_close_cad"].astype(float)
        if "last_close_cad" in features.columns
        else pd.Series(dtype=float)
    )
    open_values = _compute_open_position_values(state, px_now)
    live_equity_cad = float(state.cash_cad) + float(sum(open_values.values()))
    liquidity_portfolio_value = (
        float(live_equity_cad)
        if live_equity_cad > 0
        else float(cfg.portfolio_budget_cad)
    )

    # Optional single-pass constrained optimizer (replaces sequential transforms).
    unified_opt_enabled = bool(getattr(cfg, "unified_optimizer_enabled", True))
    optimizer_weights_snapshot: pd.DataFrame | None = None
    if unified_opt_enabled:
        current_weights = None
        try:
            if live_equity_cad > 0 and open_values:
                current_weights = pd.Series(
                    {t: (v / live_equity_cad) for t, v in open_values.items()},
                    dtype=float,
                )
        except Exception as e:
            logger.warning("Could not build current weights for unified optimizer: %s", e)

        target_weights = optimize_unified_portfolio(
            target_weights,
            features=screened,
            prices=prices,
            current_weights=current_weights,
            alpha_col=alpha_col,
            vol_col="vol_60d_ann",
            beta_col="beta",
            max_position_pct=getattr(cfg, "max_position_pct", 0.20),
            max_corr_weight=getattr(cfg, "max_corr_weight", 0.25),
            corr_threshold=getattr(cfg, "corr_threshold", 0.70),
            target_beta=getattr(cfg, "target_portfolio_beta", 1.0),
            beta_tolerance=getattr(cfg, "optimizer_beta_tolerance", 0.25),
            risk_penalty=getattr(cfg, "optimizer_risk_penalty", 1.0),
            turnover_penalty=getattr(cfg, "optimizer_turnover_penalty", 1.0),
            cost_penalty=getattr(cfg, "optimizer_cost_penalty", 1.0),
            lookback_days=60,
            use_shrinkage_cov=bool(getattr(cfg, "optimizer_use_shrinkage_cov", True)),
            shrinkage_min_obs=int(getattr(cfg, "optimizer_shrinkage_min_obs", 40)),
            allow_cash=True,
            logger=logger,
        )
        optimizer_weights_snapshot = target_weights.copy()
        run_meta["unified_optimizer"] = {
            "enabled": True,
            "risk_penalty": float(getattr(cfg, "optimizer_risk_penalty", 1.0)),
            "turnover_penalty": float(getattr(cfg, "optimizer_turnover_penalty", 1.0)),
            "cost_penalty": float(getattr(cfg, "optimizer_cost_penalty", 1.0)),
            "beta_tolerance": float(getattr(cfg, "optimizer_beta_tolerance", 0.25)),
            "use_shrinkage_cov": bool(getattr(cfg, "optimizer_use_shrinkage_cov", True)),
            "shrinkage_min_obs": int(getattr(cfg, "optimizer_shrinkage_min_obs", 40)),
        }
    else:
        # Legacy sequential transforms.
        if "pred_confidence" in screened.columns:
            confidence = screened["pred_confidence"]
            target_weights = apply_confidence_weighting(
                target_weights,
                confidence,
                cfg.confidence_weight_floor,
                logger,
            )
        if "pred_uncertainty" in screened.columns:
            target_weights = _apply_uncertainty_weighting(target_weights, screened, logger)
        
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
        
        liquidity_adj_enabled = getattr(cfg, "liquidity_adjustment", True)
        if liquidity_adj_enabled:
            target_weights = apply_liquidity_adjustment(
                target_weights,
                screened,
                liquidity_col="avg_dollar_volume_cad",
                min_liquidity=getattr(cfg, "min_liquidity_cad", 100_000),
                target_liquidity=getattr(cfg, "target_liquidity_cad", 1_000_000),
                max_position_pct_of_volume=getattr(cfg, "max_position_pct_of_volume", 0.05),
                portfolio_value=liquidity_portfolio_value,
                logger=logger,
            )
            run_meta["liquidity_adjustment"] = {"enabled": True}
        
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
        
        max_pos_pct = getattr(cfg, "max_position_pct", 0.20)
        if max_pos_pct and max_pos_pct < 1.0:
            target_weights = apply_max_position_cap(
                target_weights,
                max_position_pct=max_pos_pct,
                logger=logger,
            )
            run_meta["max_position_cap"] = {"max_pct": max_pos_pct}
        
        min_pos_pct = getattr(cfg, "min_position_pct", 0.02)
        if min_pos_pct and min_pos_pct > 0:
            target_weights = apply_min_position_filter(
                target_weights,
                min_position_pct=min_pos_pct,
                logger=logger,
            )
            run_meta["min_position_filter"] = {"min_pct": min_pos_pct}

        # Re-apply pair limits after any renormalization.
        if corr_limits_enabled:
            target_weights = apply_correlation_limits(
                target_weights,
                prices,
                max_corr_weight=getattr(cfg, "max_corr_weight", 0.25),
                corr_threshold=getattr(cfg, "corr_threshold", 0.70),
                lookback_days=60,
                logger=logger,
            )
    
    # Apply instrument sleeve constraints (equity vs fund cap/floor) before
    # top-level exposure scaling so sleeve proportions persist through scalars.
    target_weights, sleeve_info = _apply_instrument_sleeve_constraints(
        target_weights,
        screened=screened,
        cfg=cfg,
        logger=logger,
    )
    run_meta["instrument_sleeves"] = sleeve_info

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

        # Dynamic risk budgeting: adjust vol target based on model IC and confidence
        effective_vol_target = target_vol
        if getattr(cfg, "adaptive_vol_target", False):
            try:
                # Get recent IC from reward log if available
                recent_ic = 0.0
                avg_conf = 0.5
                if "pred_confidence" in screened.columns:
                    avg_conf = float(screened["pred_confidence"].mean())
                # Try to get IC from reward log (load lightweight copy)
                try:
                    _rlog_path = Path(cfg.cache_dir) / cfg.reward_log_path
                    _rlog = RewardLog.load(_rlog_path)
                    _online_ic = compute_online_ic(_rlog, window=cfg.reward_ic_window)
                    recent_ic = float(_online_ic.get("ensemble_ic") or 0.0)
                except Exception:
                    pass

                effective_vol_target = compute_adaptive_vol_target(
                    target_vol,
                    recent_ic=recent_ic,
                    ic_baseline=cfg.adaptive_vol_ic_baseline,
                    ic_sensitivity=cfg.adaptive_vol_ic_sensitivity,
                    avg_confidence=avg_conf,
                    vol_min=cfg.adaptive_vol_min,
                    vol_max=cfg.adaptive_vol_max,
                    logger=logger,
                )
            except Exception as e:
                logger.warning("Adaptive vol target failed: %s; using base target", e)

        target_weights, cash_from_vol_targeting = apply_volatility_targeting(
            target_weights,
            prices=prices,
            target_vol=effective_vol_target,
            lookback_days=20,
            min_scalar=0.5,
            max_scalar=1.0,
            logger=logger,
        )
        run_meta["volatility_targeting"] = {
            "enabled": True,
            "target_vol": effective_vol_target,
            "base_vol_target": target_vol,
            "adaptive_vol_enabled": getattr(cfg, "adaptive_vol_target", False),
            "cash_allocation": cash_from_vol_targeting,
        }

    # Portfolio actions (stateful)
    # Use full `features` for exits so we can manage holdings even if they are not in today's top-N.
    prices_cad = features["last_close_cad"].astype(float)
    pred_return = features["pred_return"].astype(float) if "pred_return" in features.columns else None
    score = scored["score"].astype(float) if "score" in scored.columns else None
    # `state` is loaded and migration-normalized before optimizer sizing.
    
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
            price_history_cad = _build_price_history_cad(prices, fx)
            n_action_bf = action_reward_log.backfill_prices(
                prices_cad,
                today_str,
                price_history_cad=price_history_cad,
            )
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
    adaptive_quick_profit_pct = getattr(cfg, "quick_profit_pct", 0.05) / max(hold_p, 0.1)
    adaptive_min_daily_return = getattr(cfg, "min_daily_return", 0.005) / max(hold_p, 0.1)
    if abs(exit_t - 1.0) > 0.05 or abs(hold_p - 1.0) > 0.05:
        logger.info(
            "Adaptive PM params: trailing_dist=%.3f vol_stop=%.3f quick_profit=%.3f",
            adaptive_trailing_dist, adaptive_vol_stop_base,
            adaptive_quick_profit_pct,
        )

    pm = PortfolioManager(
        state_path=str(state_path),
        event_log_path=str(event_log_path),
        max_holding_days=cfg.max_holding_days,
        max_holding_days_hard=cfg.max_holding_days_hard,
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
        low_daily_return_hold_min_pred_return=getattr(cfg, "low_daily_return_hold_min_pred_return", 0.01),
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
        min_trade_notional_cad=getattr(cfg, "min_trade_notional_cad", 15.0),
        min_rebalance_weight_delta=getattr(cfg, "min_rebalance_weight_delta", 0.015),
        rotate_on_missing_data=getattr(cfg, "rotate_on_missing_data", False),
        rotation_cooldown_days=getattr(cfg, "rotation_cooldown_days", 2),
        logger=logger,
    )
    # Extract market volatility regime for dynamic holding period
    market_vol_regime = None
    if "market_vol_regime" in features.columns:
        market_vol_regime = float(features["market_vol_regime"].iloc[0]) if len(features) > 0 else None
    
    # ── LLM-primary exit review (runs BEFORE mechanical exits) ─────────
    _llm_daily_exit_actions: list = []
    if _llm_primary_mode and cfg.agent_exit_review_enabled and cfg.llm_agent_enabled:
        try:
            from stock_screener.agents.trading_agent import review_exits as _review_daily_exits
            _open_for_exit = [p for p in state.positions if getattr(p, "status", "OPEN") == "OPEN" and p.ticker]
            if _open_for_exit:
                _exit_news_d: dict[str, list] = {}
                try:
                    from stock_screener.data.news import fetch_ticker_news as _fn_d
                    for _p in _open_for_exit[:6]:
                        try:
                            _exit_news_d[_p.ticker] = _fn_d(str(_p.ticker), logger=logger)[:3]
                        except Exception:
                            _exit_news_d[_p.ticker] = []
                except Exception:
                    pass
                _exit_mkt_d = {
                    "vol_regime": float(market_vol_regime) if market_vol_regime is not None else 1.0,
                    "market_trend": float(features["market_trend_20d"].iloc[0]) if "market_trend_20d" in features.columns and len(features) > 0 else 0.0,
                }
                _reviews_d = _review_daily_exits(
                    _open_for_exit, prices_cad,
                    features=features if not features.empty else None,
                    news_by_ticker=_exit_news_d,
                    market_conditions=_exit_mkt_d,
                    max_hold_days=cfg.max_holding_days,
                    log=logger,
                )
                for _t, _rv in _reviews_d.items():
                    if _rv.action != "EXIT":
                        continue
                    _veto = False
                    if _rv.urgency == "MEDIUM" and pred_return is not None:
                        _pr_val = float(pred_return.get(_t, float("nan"))) if hasattr(pred_return, "get") else 0.0
                        if not pd.isna(_pr_val) and _pr_val > cfg.llm_exit_ml_veto_threshold:
                            _veto = True
                            logger.info("LLM EXIT vetoed by ML: %s pred_return=%.2f%% > %.2f%%", _t, _pr_val * 100, cfg.llm_exit_ml_veto_threshold * 100)
                    if _rv.urgency == "LOW":
                        _veto = True  # LOW urgency exits are advisory only
                    if not _veto:
                        _pos = next((p for p in _open_for_exit if p.ticker == _t), None)
                        if _pos and _t in prices_cad.index:
                            _px = float(prices_cad[_t])
                            if _px > 0:
                                _llm_daily_exit_actions.append(TradeAction(
                                    ticker=_t, action="SELL",
                                    reason=f"LLM_EXIT_{_rv.urgency}:{_rv.reason[:50]}",
                                    shares=float(_pos.shares), price_cad=_px,
                                ))
                if _llm_daily_exit_actions:
                    logger.info("LLM-primary daily exit review: %d exit(s)", len(_llm_daily_exit_actions))
        except Exception as e:
            logger.warning("LLM daily exit review failed (continuing with mechanical): %s", e)

    # Mechanical exits (circuit breakers — always run)
    exit_actions = pm.apply_exits(
        state,
        prices_cad=prices_cad,
        pred_return=pred_return,
        score=score,
        features=features,
        market_vol_regime=market_vol_regime,
    )
    # Merge LLM exits (avoid duplicates)
    if _llm_daily_exit_actions:
        _mech_tickers = {str(a.ticker).upper() for a in exit_actions if a.action in ("SELL", "SELL_PARTIAL")}
        for _a in _llm_daily_exit_actions:
            if str(_a.ticker).upper() not in _mech_tickers:
                exit_actions.append(_a)
    exited_sell_tickers = {
        str(a.ticker).upper()
        for a in exit_actions
        if a.action in ("SELL", "SELL_PARTIAL") and getattr(a, "ticker", None)
    }
    if exit_actions:
        sells = len([a for a in exit_actions if a.action in ("SELL", "SELL_PARTIAL")])
        logger.info("Exited %s position(s) (time/stop/target/peak).", sells)
        if exited_sell_tickers:
            shown = sorted(exited_sell_tickers)
            preview = ", ".join(shown[:8]) + ("..." if len(shown) > 8 else "")
            logger.info("Blocking same-run re-entry for exited tickers: %s", preview)
        # Persist state immediately after exits so position closures and cash
        # updates are not lost if the pipeline crashes before build_trade_plan.
        # Persistence failures are fail-stop: continuing would desynchronize
        # reported actions from recorded portfolio state.
        n_exit_events = _persist_state_transition_or_fail(
            state_path=state_path,
            state=state,
            event_log_path=event_log_path,
            actions=exit_actions,
            source="apply_exits",
        )
        run_meta["exit_persistence"] = {
            "actions": int(len(_position_changing_actions(exit_actions))),
            "events_appended": int(n_exit_events),
        }

    # Apply rebalance hysteresis and trade-size guards. If unified optimizer is
    # active, it already penalizes turnover in the objective, so avoid a second
    # post-target turnover shrinkage layer.
    apply_post_turnover_shrink = not unified_opt_enabled
    if not apply_post_turnover_shrink and float(getattr(cfg, "turnover_penalty_bps", 0.0)) > 0.0:
        logger.info(
            "Unified optimizer active: disabling post-optimizer turnover shrinkage to avoid duplicate turnover penalties.",
        )
    exposure_policy = str(getattr(cfg, "exposure_policy", "allow_cash_no_upscale"))
    target_gross_exposure = float(getattr(cfg, "target_gross_exposure", 1.0))
    allow_leverage = bool(getattr(cfg, "allow_leverage", False))
    pre_rebalance_weights = target_weights.copy()
    run_meta["exposure_control_before_rebalance"] = _compute_exposure_metrics(
        pre_rebalance_weights,
        target_gross=target_gross_exposure,
    )
    rebalance_result = _apply_rebalance_controls(
        target_weights,
        state=state,
        screened=screened,
        prices_cad=prices_cad,
        market_vol_regime=market_vol_regime,
        min_rebalance_weight_delta=getattr(cfg, "min_rebalance_weight_delta", 0.015),
        min_trade_notional_cad=getattr(cfg, "min_trade_notional_cad", 15.0),
        turnover_penalty_bps=getattr(cfg, "turnover_penalty_bps", 15.0),
        dynamic_band_enabled=getattr(cfg, "dynamic_no_trade_band_enabled", True),
        uncertainty_weight=getattr(cfg, "dynamic_no_trade_uncertainty_weight", 1.2),
        liquidity_weight=getattr(cfg, "dynamic_no_trade_liquidity_weight", 0.8),
        vol_regime_weight=getattr(cfg, "dynamic_no_trade_vol_regime_weight", 0.8),
        band_mult_min=getattr(cfg, "dynamic_no_trade_multiplier_min", 1.0),
        band_mult_max=getattr(cfg, "dynamic_no_trade_multiplier_max", 3.0),
        exposure_policy=exposure_policy,
        target_gross_exposure=target_gross_exposure,
        allow_leverage=allow_leverage,
        logger=logger,
        apply_turnover_shrinkage=apply_post_turnover_shrink,
        return_diagnostics=True,
    )
    if isinstance(rebalance_result, tuple):
        target_weights, rebalance_diag = rebalance_result
    else:
        target_weights = rebalance_result
        rebalance_diag = {}
    run_meta["rebalance_gate_audit"] = rebalance_diag

    pre_rebalance_drift = _compute_weight_distance_metrics(pre_rebalance_weights, target_weights)
    if isinstance(rebalance_diag, dict):
        reason_map = {
            str(evt.get("ticker", "")).upper(): str(evt.get("reason_code", "unknown"))
            for evt in rebalance_diag.get("dropped_tickers", [])
            if isinstance(evt, dict)
        }
        pre_rebalance_drift["dropped_tickers_with_reasons"] = [
            {"ticker": str(t), "reason_code": reason_map.get(str(t).upper(), "unknown")}
            for t in pre_rebalance_drift.get("dropped_tickers", [])
        ]

    optimizer_vs_final = None
    if optimizer_weights_snapshot is not None:
        optimizer_vs_final = _compute_weight_distance_metrics(optimizer_weights_snapshot, target_weights)

    run_meta["optimizer_projection_audit"] = {
        "optimizer_enabled": bool(unified_opt_enabled),
        "pre_rebalance_vs_final": pre_rebalance_drift,
        "optimizer_vs_final": optimizer_vs_final,
        "notional_drops_with_reasons": (
            rebalance_diag.get("dropped_by_notional_gate", [])
            if isinstance(rebalance_diag, dict)
            else []
        ),
    }

    logger.info(
        "Weight projection audit: pre->final L1=%.4f L2=%.4f dropped=%d notional_drops=%d",
        float(pre_rebalance_drift.get("l1_distance", 0.0)),
        float(pre_rebalance_drift.get("l2_distance", 0.0)),
        int(pre_rebalance_drift.get("dropped_count", 0)),
        int(len(run_meta["optimizer_projection_audit"]["notional_drops_with_reasons"])),
    )
    if isinstance(optimizer_vs_final, dict):
        logger.info(
            "Weight projection audit (optimizer->final): L1=%.4f L2=%.4f dropped=%d",
            float(optimizer_vs_final.get("l1_distance", 0.0)),
            float(optimizer_vs_final.get("l2_distance", 0.0)),
            int(optimizer_vs_final.get("dropped_count", 0)),
        )

    run_meta["exposure_control_after_rebalance"] = {
        **_compute_exposure_metrics(target_weights, target_gross=target_gross_exposure),
        "policy": exposure_policy,
        "target_gross_exposure": target_gross_exposure,
        "allow_leverage": allow_leverage,
        "unified_optimizer_enabled": unified_opt_enabled,
        "post_rebalance_turnover_shrinkage_applied": bool(apply_post_turnover_shrink),
    }

    # Hard strategy gate: when ML is expected but unavailable and baseline
    # trading is not explicitly allowed, freeze entries and only manage current
    # holdings via exit/risk logic.
    if strategy_mode == "HOLD_ONLY":
        target_weights = _build_hold_only_target_weights(state, prices_cad, logger)
        run_meta["hold_only"] = {
            "enabled": True,
            "open_positions": int(
                len([p for p in state.positions if p.status == "OPEN" and p.ticker and p.shares > 0])
            ),
        }

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
        features=features,
        blocked_buys=exited_sell_tickers,
    )

    # Merge exit-based SELL actions (PEAK_TARGET, STOP_LOSS, etc.) into the
    # trade plan so they appear in the report and email.  Place sells first.
    if exit_actions:
        exit_sell_tickers = {
            a.ticker
            for a in exit_actions
            if a.action in ("SELL", "SELL_PARTIAL")
        }
        base_actions = [
            a
            for a in trade_plan.actions
            if not (a.action == "HOLD" and a.ticker in exit_sell_tickers)
        ]
        # Avoid duplicates: build_trade_plan may also generate ROTATION sells
        # for the same tickers that apply_exits already closed.
        existing_sell_tickers = {
            a.ticker for a in base_actions
            if a.action in ("SELL", "SELL_PARTIAL")
        }
        new_sells = [
            a for a in exit_actions
            if a.ticker not in existing_sell_tickers
        ]
        if new_sells:
            trade_plan = TradePlan(
                actions=new_sells + base_actions,
                holdings=trade_plan.holdings,
            )
        else:
            trade_plan = TradePlan(
                actions=base_actions,
                holdings=trade_plan.holdings,
            )
    trade_plan = TradePlan(
        actions=_sanitize_trade_actions(trade_plan.actions, logger),
        holdings=trade_plan.holdings,
    )

    # Build holdings weights from ALL open positions using the full features
    # DataFrame (not just 'screened'), so positions that have fallen out of
    # today's top-N screening still appear in the portfolio report.
    open_positions = [p for p in state.positions if p.status == "OPEN" and p.ticker and p.shares > 0]
    open_tickers = list(dict.fromkeys(str(p.ticker).upper() for p in open_positions))
    holdings_features = features.loc[features.index.intersection(open_tickers)].copy()
    # Merge in the score column from scored so all holdings have scores for
    # the report, even tickers that dropped out of the screened top-N.
    if "score" not in holdings_features.columns and "score" in scored.columns:
        holdings_features["score"] = scored["score"].reindex(holdings_features.index)
    # Positions filtered out by quality gates (vol cap, price, liquidity) won't
    # appear in 'scored' and will have NaN score.  Compute a simple proxy so the
    # report never shows NaN in the score column.
    if "score" in holdings_features.columns:
        import numpy as np
        missing_score = holdings_features["score"].isna()
        if missing_score.any():
            proxy = pd.Series(0.0, index=holdings_features.index)
            if "ret_60d" in holdings_features.columns:
                proxy += pd.to_numeric(holdings_features["ret_60d"], errors="coerce").fillna(0)
            if "ret_120d" in holdings_features.columns:
                proxy += 0.5 * pd.to_numeric(holdings_features["ret_120d"], errors="coerce").fillna(0)
            holdings_features.loc[missing_score, "score"] = proxy[missing_score]
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
    # If some open positions are missing from today's feature frame (e.g., transient data gap),
    # keep them in the holdings report with best-effort market-value weights.
    holdings_index_upper = {str(t).upper() for t in holdings_weights.index.astype(str)}
    missing_holdings = [t for t in open_tickers if t not in holdings_index_upper]
    if missing_holdings:
        open_mkt_value = 0.0
        market_value_by_ticker: dict[str, float] = {}
        for p in open_positions:
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.isna(px) or px <= 0:
                continue
            mv = float(px) * float(p.shares)
            open_mkt_value += mv
            market_value_by_ticker[str(p.ticker).upper()] = mv
        equity_cad = float(state.cash_cad) + float(open_mkt_value)
        fallback_rows: list[dict[str, Any]] = []
        for t in missing_holdings:
            px = float(prices_cad.get(t, float("nan")))
            px_val = pd.NA if pd.isna(px) or px <= 0 else float(px)
            mv = market_value_by_ticker.get(t)
            w = float(mv / equity_cad) if mv is not None and equity_cad > 0 else pd.NA
            fallback_rows.append(
                {
                    "ticker": t,
                    "weight": w,
                    "score": pd.NA,
                    "last_close_cad": px_val,
                    "ret_60d": pd.NA,
                    "vol_60d_ann": pd.NA,
                    "avg_dollar_volume_cad": pd.NA,
                }
            )
        if fallback_rows:
            fallback_df = pd.DataFrame(fallback_rows).set_index("ticker")
            holdings_weights = pd.concat([holdings_weights, fallback_df], axis=0)
            shown = ", ".join(missing_holdings[:8]) + ("..." if len(missing_holdings) > 8 else "")
            logger.warning(
                "Included %d open holding(s) without feature rows in report: %s",
                len(missing_holdings),
                shown,
            )
    # Attach current holdings sizing for reporting (shares + actual/live weights).
    # Keep model/optimizer recommendation as target_weight so the report can show
    # both "what we hold now" and "what model currently prefers".
    shares_by_ticker = {str(p.ticker).upper(): float(p.shares) for p in open_positions}
    market_value_by_ticker: dict[str, float] = {}
    open_mkt_value_total = 0.0
    for p in open_positions:
        px = float(prices_cad.get(p.ticker, float("nan")))
        if pd.isna(px) or px <= 0:
            continue
        mv = float(px) * float(p.shares)
        key = str(p.ticker).upper()
        market_value_by_ticker[key] = market_value_by_ticker.get(key, 0.0) + mv
        open_mkt_value_total += mv
    equity_cad_live = float(state.cash_cad) + float(open_mkt_value_total)

    holdings_weights = holdings_weights.copy()
    if "weight" in holdings_weights.columns:
        holdings_weights["target_weight"] = holdings_weights["weight"]
    else:
        holdings_weights["target_weight"] = pd.NA
    holdings_weights["shares"] = [
        shares_by_ticker.get(str(t).upper(), pd.NA)
        for t in holdings_weights.index.astype(str)
    ]
    holdings_weights["position_value_cad"] = [
        market_value_by_ticker.get(str(t).upper(), pd.NA)
        for t in holdings_weights.index.astype(str)
    ]
    if equity_cad_live > 0:
        holdings_weights["actual_weight"] = pd.to_numeric(
            holdings_weights["position_value_cad"], errors="coerce"
        ) / float(equity_cad_live)
    else:
        holdings_weights["actual_weight"] = pd.NA
    # Keep backwards compatibility for downstream code that expects `weight`:
    # prefer actual holdings weight when available.
    holdings_weights["weight"] = holdings_weights["actual_weight"].where(
        pd.notna(holdings_weights["actual_weight"]),
        holdings_weights["target_weight"],
    )
    run_meta["portfolio_state_after_planning"] = {
        "open_positions": int(len(open_positions)),
        "closed_positions": int(len([p for p in state.positions if p.status != "OPEN"])),
    }

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
    _check_runtime_budget(started_utc, cfg, logger, "reporting")

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
                    event_type="PREDICTION",
                ))

            # Index closed positions by ticker (newest first) so each sell action
            # can be matched to the most recent close instead of a stale old lot.
            closed_positions_by_ticker: dict[str, list[Any]] = {}
            dt_min_utc = datetime.min.replace(tzinfo=timezone.utc)
            for p in state.positions:
                if p.status == "OPEN" or not p.ticker:
                    continue
                key = str(p.ticker).upper()
                closed_positions_by_ticker.setdefault(key, []).append(p)
            for key in list(closed_positions_by_ticker.keys()):
                closed_positions_by_ticker[key] = sorted(
                    closed_positions_by_ticker[key],
                    key=lambda p: p.exit_date if p.exit_date is not None else dt_min_utc,
                    reverse=True,
                )

            # Record closed positions from today's exits
            for action in trade_plan.actions:
                if action.action in ("SELL", "SELL_PARTIAL"):
                    ticker_key = str(action.ticker).upper()
                    candidates = closed_positions_by_ticker.get(ticker_key, [])
                    pos = None
                    if candidates:
                        # Prefer matching exit reason to avoid mismatching lots.
                        reason_idx = next(
                            (i for i, candidate in enumerate(candidates) if candidate.exit_reason == action.reason),
                            None,
                        )
                        if reason_idx is not None:
                            pos = candidates.pop(reason_idx)
                        else:
                            pos = candidates.pop(0)

                    entry_price = None
                    if pos and pos.entry_price and pos.entry_price > 0:
                        entry_price = float(pos.entry_price)
                    elif action.entry_price and action.entry_price > 0:
                        entry_price = float(action.entry_price)

                    if entry_price and action.price_cad:
                        cum_ret = (action.price_cad - entry_price) / entry_price
                        close_pred_return = _lookup_metric_from_sources(
                            action.ticker,
                            "pred_return",
                            [screened, scored, features],
                        )
                        if close_pred_return is None and action.pred_return is not None:
                            close_pred_return = float(action.pred_return)
                        if (
                            close_pred_return is None
                            and pos is not None
                            and getattr(pos, "entry_pred_return", None) is not None
                        ):
                            try:
                                close_pred_return = float(pos.entry_pred_return)
                            except (TypeError, ValueError):
                                close_pred_return = None
                        close_confidence = _lookup_metric_from_sources(
                            action.ticker,
                            "pred_confidence",
                            [screened, scored, features],
                        )
                        # Update or create an entry for the closed trade
                        new_entries.append(RewardEntry(
                            date=today_str,
                            ticker=action.ticker,
                            predicted_return=float(close_pred_return) if close_pred_return is not None else 0.0,
                            confidence=float(close_confidence) if close_confidence is not None else None,
                            realized_cumulative_return=cum_ret,
                            days_held=action.days_held,
                            exit_reason=action.reason,
                            price_at_prediction=entry_price,
                            event_type="CLOSE",
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

            # Build explicit rotation linkage from BUY actions that declare
            # which ticker they replaced (set in PortfolioManager.build_trade_plan).
            sell_to_buy: dict[str, str] = {}  # sold_ticker -> bought_ticker
            buy_to_sell: dict[str, str] = {}  # bought_ticker -> sold_ticker
            for a in trade_plan.actions:
                if a.action != "BUY":
                    continue
                replaced = getattr(a, "replaces_ticker", None)
                if not replaced:
                    continue
                sold_ticker = str(replaced).upper()
                bought_ticker = str(a.ticker).upper()
                buy_to_sell[bought_ticker] = sold_ticker
                if sold_ticker not in sell_to_buy:
                    sell_to_buy[sold_ticker] = bought_ticker

            action_entries: list[ActionRewardEntry] = []

            for action in trade_plan.actions:
                if action.action not in ("BUY", "SELL", "SELL_PARTIAL", "HOLD"):
                    continue
                px = action.price_cad
                if not px or pd.isna(px) or float(px) <= 0:
                    continue
                pred_ret_val = float(action.pred_return) if action.pred_return is not None else None
                if pred_ret_val is None:
                    pred_ret_val = _lookup_metric_from_sources(
                        action.ticker,
                        "pred_return",
                        [screened, scored, features],
                    )
                pred_ret = float(pred_ret_val) if pred_ret_val is not None else 0.0
                conf_val = _lookup_metric_from_sources(
                    action.ticker,
                    "pred_confidence",
                    [screened, scored, features],
                )
                conf = float(conf_val) if conf_val is not None else 0.0
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

                action_ticker = str(action.ticker).upper()
                if action.action in ("SELL", "SELL_PARTIAL") and action_ticker in sell_to_buy:
                    rpl_ticker = sell_to_buy[action_ticker]
                    replaced_by = rpl_ticker
                    rpl_px = prices_cad.get(rpl_ticker)
                    if rpl_px and not pd.isna(rpl_px) and float(rpl_px) > 0:
                        replaced_by_px = float(rpl_px)

                if action.action == "BUY" and action_ticker in buy_to_sell:
                    sold_ticker = buy_to_sell[action_ticker]
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
            n_rotations = len(buy_to_sell)
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

    # Persist lightweight cache for intraday monitoring runs
    try:
        _open_tickers = [
            p.ticker for p in state.positions if getattr(p, "status", "OPEN") == "OPEN"
        ]
        _screened_tickers = list(screened.index[:cfg.intraday_watchlist_top_n]) if not screened.empty else []

        # Save screened features and target weights as parquet for fast intraday reload.
        # This gives the intraday pipeline full access to the daily signal without recomputation.
        _intraday_data_dir = Path(cache_dir) / "intraday"
        _intraday_data_dir.mkdir(parents=True, exist_ok=True)
        if not screened.empty:
            screened.to_parquet(_intraday_data_dir / "screened.parquet", engine="auto")
        if target_weights is not None and not target_weights.empty:
            target_weights.to_parquet(_intraday_data_dir / "target_weights.parquet", engine="auto")

        _intraday_cache = {
            "run_utc": datetime.now(tz=timezone.utc).isoformat(),
            "screened_tickers": _screened_tickers,
            "held_tickers": _open_tickers,
            "fx_usdcad": float(fx.iloc[-1]) if fx is not None and not fx.empty else 1.35,
            "portfolio_size": int(effective_portfolio_size),
        }
        # Cache market regime scalar (single value, same for all tickers)
        if "market_vol_regime" in screened.columns and not screened.empty:
            _intraday_cache["market_vol_regime"] = float(screened["market_vol_regime"].iloc[0])
        write_json(cache_dir / "intraday_cache.json", _intraday_cache)
        logger.info("Saved intraday cache: %d screened tickers, %d held, %d target weights",
                   len(screened), len(_open_tickers),
                   len(target_weights) if target_weights is not None else 0)
    except Exception as _ic_err:
        logger.warning("Failed to save intraday cache: %s", _ic_err)

    # Persist metadata for debugging/auditing
    write_json(cache_dir / "last_run_meta.json", run_meta)
    logger.info("Wrote reports to %s", reports_dir.resolve())


def run_intraday(cfg, logger) -> None:
    """Intraday trading pipeline: scan, trade, and manage the portfolio.

    Loads the daily run's screened universe and target weights, downloads
    fresh intraday prices, then runs the full portfolio management cycle —
    exits, rotation, new entries, and rebalancing — using live prices with
    the daily ML signal as the alpha source.

    Designed to complete in <3 minutes.
    """
    if _check_kill_switch(logger):
        return

    from pathlib import Path
    from stock_screener.data.prices import download_intraday_prices, get_latest_prices
    from stock_screener.portfolio.state import load_portfolio_state, save_portfolio_state
    from stock_screener.portfolio.manager import PortfolioManager
    from stock_screener.reporting.render import render_reports
    from stock_screener.utils import read_json, write_json, ensure_dir

    started_utc = datetime.now(tz=timezone.utc)
    cache_dir = ensure_dir(cfg.cache_dir)
    reports_dir = ensure_dir(cfg.reports_dir)
    intraday_data_dir = Path(cache_dir) / "intraday"

    # ── Load caches first (needed by _empty_report) ────────────────
    try:
        last_meta = read_json(cache_dir / "last_run_meta.json")
    except (FileNotFoundError, OSError):
        last_meta = None
    try:
        intraday_cache = read_json(cache_dir / "intraday_cache.json") or {}
    except (FileNotFoundError, OSError):
        intraday_cache = {}

    def _empty_report(status: str, msg: str) -> None:
        """Render a report with current positions so the email is never blank."""
        _holdings = pd.DataFrame()
        _pnl_history = None
        try:
            _state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
            _pnl_history = getattr(_state, "pnl_history", None)
            _rows = []
            for _p in _state.positions:
                if getattr(_p, "status", "OPEN") != "OPEN":
                    continue
                _rows.append({
                    "ticker": _p.ticker,
                    "weight": 0.0,
                    "last_close_cad": _p.entry_price,
                    "ret_60d": float("nan"),
                    "vol_60d_ann": float("nan"),
                    "score": float("nan"),
                })
            if _rows:
                _holdings = pd.DataFrame(_rows).set_index("ticker")
        except Exception:
            pass
        render_reports(
            reports_dir=Path(reports_dir),
            run_meta={"intraday": True, "status": status, "message": msg, "started_utc": started_utc.isoformat()},
            universe_meta={"us": {}, "tsx": {}, "total_requested": 0},
            screened=_holdings,
            weights=_holdings,
            trade_actions=[],
            logger=logger,
            portfolio_pnl_history=_pnl_history,
            fx_usdcad_rate=intraday_cache.get("fx_usdcad"),
        )
        logger.warning(msg)

    # ── Staleness guard ──────────────────────────────────────────────
    if not last_meta:
        _empty_report("no_daily_run", "No daily run metadata found; skipping intraday run")
        return
    last_run_ts = last_meta.get("started_utc") or last_meta.get("run_utc")
    if last_run_ts:
        try:
            last_dt = datetime.fromisoformat(str(last_run_ts).replace("Z", "+00:00"))
            age_hours = (started_utc - last_dt).total_seconds() / 3600.0
            if age_hours > cfg.intraday_stale_threshold_hours:
                _empty_report("stale", f"Daily run {age_hours:.1f}h old (threshold {cfg.intraday_stale_threshold_hours:.1f}h)")
                return
            logger.info("Daily run age: %.1fh — OK", age_hours)
        except Exception as e:
            logger.warning("Could not parse daily run timestamp: %s", e)

    # Load full screened features and target weights from parquet
    screened = pd.DataFrame()
    target_weights = pd.DataFrame()
    try:
        screened_path = intraday_data_dir / "screened.parquet"
        if screened_path.exists():
            screened = pd.read_parquet(screened_path)
            logger.info("Loaded screened cache: %d tickers, %d columns", len(screened), len(screened.columns))
    except Exception as e:
        logger.warning("Could not load screened cache: %s", e)
    try:
        tw_path = intraday_data_dir / "target_weights.parquet"
        if tw_path.exists():
            target_weights = pd.read_parquet(tw_path)
            logger.info("Loaded target weights cache: %d tickers", len(target_weights))
    except Exception as e:
        logger.warning("Could not load target weights cache: %s", e)

    if screened.empty and target_weights.empty:
        _empty_report("no_cache", "No cached screened data or target weights; skipping intraday run")
        return

    fx_rate = intraday_cache.get("fx_usdcad", 1.35)
    market_vol_regime = intraday_cache.get("market_vol_regime")

    # ── Load portfolio state ─────────────────────────────────────────
    state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
    open_positions = [p for p in state.positions if getattr(p, "status", "OPEN") == "OPEN"]
    held_tickers = [p.ticker for p in open_positions]

    # ── Build ticker list: held + target + screened watchlist ─────────
    target_tickers = list(target_weights.index) if not target_weights.empty else []
    watchlist = list(screened.index[:cfg.intraday_watchlist_top_n]) if not screened.empty else []
    all_tickers = list(dict.fromkeys(held_tickers + target_tickers + watchlist))[:cfg.intraday_ticker_limit]
    logger.info("Intraday scan: %d held, %d targets, %d watchlist, %d total",
                len(held_tickers), len(target_tickers), len(watchlist), len(all_tickers))

    if not all_tickers:
        _empty_report("no_tickers", "No tickers to scan; skipping")
        return

    # ── Download intraday prices ─────────────────────────────────────
    intraday_prices = download_intraday_prices(
        tickers=all_tickers, period="5d", interval="1h",
        threads=True, batch_size=50, logger=logger,
    )
    if intraday_prices.empty:
        _empty_report("no_intraday_data", "No intraday data (market may be closed)")
        return

    # ── Derive current prices in CAD ─────────────────────────────────
    latest_prices = get_latest_prices(intraday_prices)
    if latest_prices.empty:
        _empty_report("no_prices", "Could not extract latest prices")
        return

    prices_cad = pd.Series(dtype=float)
    for ticker in latest_prices.index:
        px = float(latest_prices[ticker])
        is_tsx = str(ticker).upper().endswith(".TO") or str(ticker).upper().endswith(".V")
        prices_cad[str(ticker)] = px * float(fx_rate) if not is_tsx else px

    # Update screened DataFrame with live prices so scoring uses current data
    if not screened.empty and "last_close_cad" in screened.columns:
        for t in screened.index:
            if str(t) in prices_cad.index:
                screened.loc[t, "last_close_cad"] = prices_cad[str(t)]

    # ── Intraday LLM agent analysis (fail-soft) ─────────────────────
    run_meta_intraday = {
        "intraday": True,
        "started_utc": started_utc.isoformat(),
        "label_horizon_days": getattr(cfg, "label_horizon_days", 5),
    }
    if cfg.llm_agent_enabled and not screened.empty:
        try:
            from stock_screener.agents.trading_agent import (
                analyze_candidates, blend_llm_scores, build_agent_candidates, build_portfolio_context,
            )
            from stock_screener.agents.config import get_agent_config as _get_agent_config
            from stock_screener.data.news import fetch_ticker_news

            _agent_cfg = _get_agent_config()
            _has_key = bool(_agent_cfg.get("api_key"))
            _llm_tickers = list(screened.index[:cfg.dynamic_size_max_positions])
            logger.info("Intraday LLM agent: provider=%s, key=%s, tickers=%d",
                        _agent_cfg.get("provider"), "set" if _has_key else "MISSING", len(_llm_tickers))

            if not _has_key:
                run_meta_intraday["llm_agent"] = {"status": "skipped", "reason": "no API key"}
            else:
                # Compute intraday price action from 1h bars for each ticker
                _intraday_context: dict[str, dict] = {}
                for _t in _llm_tickers:
                    _ctx: dict = {}
                    try:
                        if (str(_t), "Close") in intraday_prices.columns:
                            _bars = intraday_prices[(str(_t), "Close")].dropna()
                            if len(_bars) >= 2:
                                _ctx["intraday_last"] = float(_bars.iloc[-1])
                                _ctx["intraday_open_today"] = float(_bars.iloc[-min(7, len(_bars))])
                                _ctx["intraday_high"] = float(_bars.iloc[-min(7, len(_bars)):].max())
                                _ctx["intraday_low"] = float(_bars.iloc[-min(7, len(_bars)):].min())
                                _ctx["intraday_change"] = float(_bars.iloc[-1] / _bars.iloc[-2] - 1.0)
                                _ctx["bars_today"] = min(7, len(_bars))
                    except Exception:
                        pass
                    _intraday_context[str(_t)] = _ctx

                _news_by_ticker: dict[str, list] = {}
                for _nt in _llm_tickers:
                    try:
                        _news_by_ticker[str(_nt)] = fetch_ticker_news(str(_nt), logger=logger)[:5]
                    except Exception:
                        _news_by_ticker[str(_nt)] = []

                _agent_candidates = build_agent_candidates(
                    screened, max_tickers=cfg.dynamic_size_max_positions,
                    news_by_ticker=_news_by_ticker,
                    intraday_context=_intraday_context,
                    prices_cad=prices_cad,
                )

                _portfolio_ctx = build_portfolio_context(
                    state.positions, state.cash_cad, prices_cad=prices_cad,
                )

                _decisions = analyze_candidates(
                    _agent_candidates, portfolio_context=_portfolio_ctx, log=logger,
                )

                if _decisions:
                    screened = blend_llm_scores(
                        screened, _decisions, score_col="score",
                        ml_weight=cfg.llm_agent_ml_weight, llm_weight=cfg.llm_agent_llm_weight,
                        log=logger,
                    )
                    screened = screened.sort_values("score", ascending=False)
                    run_meta_intraday["llm_agent"] = {
                        "status": "success",
                        "n_analyzed": len(_decisions),
                        "decisions": {
                            t: {
                                "rating": d.rating,
                                "score": d.score,
                                "reasoning": d.reasoning,
                                "bull_thesis": d.bull_thesis,
                                "bear_thesis": d.bear_thesis,
                                "risk_assessment": d.risk_assessment,
                                "debate_rounds": getattr(d, "debate_rounds", 1),
                                "debate_history": getattr(d, "debate_history", []),
                                "risk_debate": getattr(d, "risk_debate", None),
                                "analyst_reports": getattr(d, "analyst_reports", None),
                            }
                            for t, d in _decisions.items()
                        },
                    }
                    logger.info("Intraday LLM: blended scores for %d tickers", len(_decisions))
                else:
                    run_meta_intraday["llm_agent"] = {"status": "no_results", "reason": "API returned empty"}
        except Exception as e:
            run_meta_intraday["llm_agent"] = {"status": "error", "reason": str(e)}
            logger.warning("Intraday LLM agent failed: %s", e)

    # ── Build PortfolioManager ───────────────────────────────────────
    effective_portfolio_size = intraday_cache.get("portfolio_size", cfg.dynamic_size_max_positions)
    pm = PortfolioManager(
        state_path=str(cfg.portfolio_state_path),
        max_holding_days=cfg.max_holding_days,
        max_holding_days_hard=cfg.max_holding_days_hard,
        extend_hold_min_pred_return=cfg.extend_hold_min_pred_return,
        extend_hold_min_score=cfg.extend_hold_min_score,
        max_positions=effective_portfolio_size,
        stop_loss_pct=cfg.stop_loss_pct,
        take_profit_pct=cfg.take_profit_pct,
        trailing_stop_enabled=cfg.trailing_stop_enabled,
        trailing_stop_activation_pct=cfg.trailing_stop_activation_pct,
        trailing_stop_distance_pct=cfg.trailing_stop_distance_pct,
        peak_based_exit=cfg.peak_based_exit,
        twr_optimization=cfg.twr_optimization,
        quick_profit_pct=cfg.quick_profit_pct,
        quick_profit_days=cfg.quick_profit_days,
        min_daily_return=cfg.min_daily_return,
        low_daily_return_hold_min_pred_return=cfg.low_daily_return_hold_min_pred_return,
        momentum_decay_exit=cfg.momentum_decay_exit,
        signal_decay_exit_enabled=cfg.signal_decay_exit_enabled,
        signal_decay_threshold=cfg.signal_decay_threshold,
        dynamic_holding_enabled=cfg.dynamic_holding_enabled,
        dynamic_holding_vol_scale=cfg.dynamic_holding_vol_scale,
        vol_adjusted_stop_enabled=cfg.vol_adjusted_stop_enabled,
        vol_adjusted_stop_base=cfg.vol_adjusted_stop_base,
        vol_adjusted_stop_min=cfg.vol_adjusted_stop_min,
        vol_adjusted_stop_max=cfg.vol_adjusted_stop_max,
        age_urgency_enabled=cfg.age_urgency_enabled,
        age_urgency_start_day=cfg.age_urgency_start_day,
        age_urgency_min_return=cfg.age_urgency_min_return,
        peak_detection_enabled=cfg.peak_detection_enabled,
        peak_sell_portion_pct=cfg.peak_sell_portion_pct,
        peak_min_gain_pct=cfg.peak_min_gain_pct,
        peak_min_holding_days=cfg.peak_min_holding_days,
        peak_pred_return_threshold=cfg.peak_pred_return_threshold,
        peak_score_percentile_drop=cfg.peak_score_percentile_drop,
        peak_rsi_overbought=cfg.peak_rsi_overbought,
        peak_above_ma_ratio=cfg.peak_above_ma_ratio,
        min_trade_notional_cad=cfg.min_trade_notional_cad,
        min_rebalance_weight_delta=cfg.min_rebalance_weight_delta,
        rotate_on_missing_data=cfg.rotate_on_missing_data,
        rotation_cooldown_days=cfg.rotation_cooldown_days,
        logger=logger,
    )

    # ── 1. Exits — evaluate all held positions with live prices ──────
    pred_return_series = screened["pred_return"] if "pred_return" in screened.columns else None
    score_series = screened["score"] if "score" in screened.columns else None

    exit_actions = pm.apply_exits(
        state, prices_cad,
        pred_return=pred_return_series,
        score=score_series,
        features=screened if not screened.empty else None,
        market_vol_regime=float(market_vol_regime) if market_vol_regime is not None else None,
    )
    logger.info("Exits: %d actions from %d positions", len(exit_actions), len(held_tickers))

    # ── Phase 5: LLM exit review for positions mechanical rules say HOLD ──
    if cfg.agent_exit_review_enabled and cfg.llm_agent_enabled:
        try:
            from stock_screener.agents.trading_agent import review_exits as _review_exits
            _exited_by_mechanical = {
                str(getattr(a, "ticker", "") or "").upper()
                for a in exit_actions
                if getattr(a, "action", "") in ("SELL", "SELL_PARTIAL")
            }
            _held_after_exits = [
                p for p in state.positions
                if getattr(p, "status", "OPEN") == "OPEN"
                and str(getattr(p, "ticker", "")).upper() not in _exited_by_mechanical
            ]
            if _held_after_exits:
                _exit_news: dict[str, list] = {}
                try:
                    from stock_screener.data.news import fetch_ticker_news as _fetch_news
                    for _p in _held_after_exits[:6]:
                        try:
                            _exit_news[_p.ticker] = _fetch_news(str(_p.ticker), logger=logger)[:3]
                        except Exception:
                            _exit_news[_p.ticker] = []
                except Exception:
                    pass

                _exit_market = {
                    "vol_regime": float(market_vol_regime) if market_vol_regime is not None else 1.0,
                    "market_trend": float(screened["market_trend_20d"].iloc[0]) if "market_trend_20d" in screened.columns and len(screened) > 0 else 0.0,
                }
                _exit_reviews = _review_exits(
                    _held_after_exits, prices_cad,
                    features=screened if not screened.empty else None,
                    news_by_ticker=_exit_news,
                    market_conditions=_exit_market,
                    max_hold_days=cfg.max_holding_days,
                    log=logger,
                )
                _llm_exits_added = 0
                for _ticker, _review in _exit_reviews.items():
                    if _review.action == "EXIT" and _review.urgency in ("MEDIUM", "HIGH"):
                        _pos = next((p for p in _held_after_exits if p.ticker == _ticker), None)
                        if _pos and _ticker in prices_cad:
                            _px = float(prices_cad[_ticker])
                            if _px > 0:
                                exit_actions.append(TradeAction(
                                    ticker=_ticker, action="SELL",
                                    reason=f"LLM_EXIT:{_review.reason[:50]}",
                                    shares=float(_pos.shares), price_cad=_px,
                                    days_held=None, pred_return=None,
                                ))
                                _llm_exits_added += 1
                if _llm_exits_added > 0:
                    logger.info("LLM exit review: added %d exit(s)", _llm_exits_added)
        except Exception as e:
            logger.warning("LLM exit review failed (continuing): %s", e)

    # Update trailing stops with intraday peaks
    for p in [pp for pp in state.positions if getattr(pp, "status", "OPEN") == "OPEN"]:
        if p.ticker in prices_cad and hasattr(p, "update_highest_price"):
            p.update_highest_price(float(prices_cad[p.ticker]))

    # ── 2. Entries + rotation — full trade plan with live prices ─────
    exited_tickers = {
        str(getattr(a, "ticker", "") or (a.get("ticker", "") if isinstance(a, dict) else "")).upper()
        for a in exit_actions
    }

    entry_actions = []
    hold_actions = []
    rotation_actions = []
    _pre_trade_positions = list(state.positions)
    _pre_trade_cash = state.cash_cad

    if not target_weights.empty:
        try:
            trade_plan = pm.build_trade_plan(
                state=state,
                screened=screened,
                weights=target_weights,
                prices_cad=prices_cad,
                scored=screened if not screened.empty else None,
                features=screened if not screened.empty else None,
                blocked_buys=exited_tickers,
            )
            for a in trade_plan.actions:
                act = getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else None)
                if act == "BUY":
                    entry_actions.append(a)
                elif act == "HOLD":
                    hold_actions.append(a)
                elif act in ("SELL", "SELL_PARTIAL"):
                    rotation_actions.append(a)
            logger.info("Trade plan: %d buys, %d holds, %d rotations",
                        len(entry_actions), len(hold_actions), len(rotation_actions))
        except Exception as e:
            logger.warning("Trade plan failed: %s; rolling back", e)
            state.positions = _pre_trade_positions
            state.cash_cad = _pre_trade_cash
    else:
        logger.info("No target weights cached; skipping entries/rotation")

    # ── 3. Persist state ─────────────────────────────────────────────
    _append_pnl_snapshot(state, prices_cad=prices_cad, now=started_utc)
    save_portfolio_state(cfg.portfolio_state_path, state)

    # ── 4. Report — use same render_reports() as daily pipeline ─────
    open_after = [p for p in state.positions if getattr(p, "status", "OPEN") == "OPEN"]
    shares_by_ticker = {str(p.ticker).upper(): float(p.shares) for p in open_after}
    market_value_by_ticker: dict[str, float] = {}
    open_mkt_value_total = 0.0
    for p in open_after:
        px = float(prices_cad.get(p.ticker, float("nan")))
        if pd.isna(px) or px <= 0:
            continue
        mv = float(px) * float(p.shares)
        market_value_by_ticker[str(p.ticker).upper()] = market_value_by_ticker.get(str(p.ticker).upper(), 0.0) + mv
        open_mkt_value_total += mv
    equity_cad_live = float(state.cash_cad) + open_mkt_value_total

    # Build holdings DataFrame matching what the daily pipeline passes to render_reports
    holdings_weights = screened.copy() if not screened.empty else pd.DataFrame()
    # Keep only tickers that are currently held
    held_set = {str(p.ticker) for p in open_after}
    if not holdings_weights.empty:
        holdings_weights = holdings_weights[holdings_weights.index.isin(held_set)].copy()
    # Add any held tickers missing from screened
    for t in held_set:
        if t not in holdings_weights.index:
            holdings_weights.loc[t] = float("nan")

    if "weight" in holdings_weights.columns:
        holdings_weights["target_weight"] = holdings_weights["weight"]
    else:
        holdings_weights["target_weight"] = pd.NA
    holdings_weights["shares"] = holdings_weights.index.astype(str).map(
        lambda t: shares_by_ticker.get(str(t).upper(), pd.NA)
    )
    holdings_weights["position_value_cad"] = holdings_weights.index.astype(str).map(
        lambda t: market_value_by_ticker.get(str(t).upper(), pd.NA)
    )
    if equity_cad_live > 0:
        holdings_weights["actual_weight"] = pd.to_numeric(holdings_weights["position_value_cad"], errors="coerce") / equity_cad_live
    else:
        holdings_weights["actual_weight"] = pd.NA
    holdings_weights["weight"] = holdings_weights["actual_weight"].where(
        pd.notna(holdings_weights["actual_weight"]),
        holdings_weights.get("target_weight", pd.NA),
    )

    all_trade_actions = exit_actions + rotation_actions + entry_actions + hold_actions

    render_reports(
        reports_dir=Path(reports_dir),
        run_meta=run_meta_intraday,
        universe_meta={"us": {}, "tsx": {}, "total_requested": len(all_tickers)},
        screened=screened,
        weights=holdings_weights,
        trade_actions=all_trade_actions,
        logger=logger,
        target_weights=target_weights if not target_weights.empty else None,
        portfolio_pnl_history=state.pnl_history if hasattr(state, "pnl_history") else None,
        fx_usdcad_rate=fx_rate,
        total_processed=len(all_tickers),
    )

    # ── 5. Metadata ──────────────────────────────────────────────────
    elapsed = (datetime.now(tz=timezone.utc) - started_utc).total_seconds()
    def _ticker(a):
        return str(getattr(a, "ticker", "") or (a.get("ticker", "") if isinstance(a, dict) else ""))
    write_json(cache_dir / "last_intraday_meta.json", {
        "started_utc": started_utc.isoformat(),
        "elapsed_seconds": elapsed,
        "n_positions": len(open_after),
        "n_exits": len(exit_actions),
        "n_rotations": len(rotation_actions),
        "n_entries": len(entry_actions),
        "n_holds": len(hold_actions),
        "exit_tickers": [_ticker(a) for a in exit_actions],
        "entry_tickers": [_ticker(a) for a in entry_actions],
    })
    logger.info("Intraday complete: %d exits, %d rotations, %d entries, %d holds, %d open (%.1fs)",
                len(exit_actions), len(rotation_actions), len(entry_actions),
                len(hold_actions), len(open_after), elapsed)


# Backward compatibility alias
run_intraday_monitor = run_intraday
