from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_screener.config import Config
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.prices import download_price_history
from stock_screener.features.technical import compute_features
from stock_screener.optimization.risk_parity import compute_inverse_vol_weights
from stock_screener.reporting.render import render_reports
from stock_screener.screening.screener import screen_universe
from stock_screener.universe.tsx import fetch_tsx_universe
from stock_screener.universe.us import fetch_us_universe
from stock_screener.utils import Universe, ensure_dir, write_json
from stock_screener.modeling.model import load_ensemble, load_model, predict, predict_ensemble
from stock_screener.portfolio.manager import PortfolioManager
from stock_screener.portfolio.state import load_portfolio_state


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

    features = compute_features(
        prices=prices,
        fx_usdcad=fx,
        liquidity_lookback_days=cfg.liquidity_lookback_days,
        feature_lookback_days=cfg.feature_lookback_days,
        logger=logger,
    )

    model = None
    if cfg.use_ml:
        try:
            mp = Path(cfg.model_path)
            if mp.name.lower() == "manifest.json":
                models, weights = load_ensemble(mp)
                logger.info("Loaded ML ensemble from %s (%s members)", cfg.model_path, len(models))
                features["pred_return"] = predict_ensemble(models, weights, features)
            else:
                model = load_model(cfg.model_path)
                logger.info("Loaded ML model from %s", cfg.model_path)
                features["pred_return"] = predict(model, features)
        except Exception as e:
            logger.warning("ML enabled but model could not be loaded/used: %s", e)

    screened = screen_universe(
        features=features,
        min_price_cad=cfg.min_price_cad,
        min_avg_dollar_volume_cad=cfg.min_avg_dollar_volume_cad,
        top_n=cfg.top_n,
        logger=logger,
    )

    target_weights = compute_inverse_vol_weights(
        features=screened,
        portfolio_size=cfg.portfolio_size,
        weight_cap=cfg.weight_cap,
        logger=logger,
    )

    # Portfolio actions (stateful)
    # Use full `features` for exits so we can manage holdings even if they are not in today's top-N.
    prices_cad = features["last_close_cad"].astype(float)
    pred_return = features["pred_return"].astype(float) if "pred_return" in features.columns else None
    score = screened["score"].astype(float) if "score" in screened.columns else None
    state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
    pm = PortfolioManager(
        state_path=cfg.portfolio_state_path,
        max_holding_days=cfg.max_holding_days,
        max_holding_days_hard=cfg.max_holding_days_hard,
        extend_hold_min_pred_return=cfg.extend_hold_min_pred_return,
        extend_hold_min_score=cfg.extend_hold_min_score,
        max_positions=cfg.portfolio_size,
        stop_loss_pct=cfg.stop_loss_pct,
        take_profit_pct=cfg.take_profit_pct,
        logger=logger,
    )
    exit_actions = pm.apply_exits(state, prices_cad=prices_cad, pred_return=pred_return, score=score)
    if exit_actions:
        logger.info("Exited %s position(s) (time/stop/target).", len([a for a in exit_actions if a.action == "SELL"]))

    trade_plan = pm.build_trade_plan(
        state=state,
        screened=screened,
        weights=target_weights,
        prices_cad=prices_cad,
    )

    open_tickers = [p.ticker for p in state.positions if p.status == "OPEN"]
    holdings_features = screened.loc[screened.index.intersection(open_tickers)].copy()
    holdings_features = holdings_features.sort_values("score", ascending=False)
    holdings_weights = compute_inverse_vol_weights(
        features=holdings_features,
        portfolio_size=len(holdings_features),
        weight_cap=cfg.weight_cap,
        logger=logger,
    )

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
    )

    # Persist metadata for debugging/auditing
    write_json(cache_dir / "last_run_meta.json", run_meta)
    logger.info("Wrote reports to %s", reports_dir.resolve())


