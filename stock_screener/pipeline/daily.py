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

    weights = compute_inverse_vol_weights(
        features=screened,
        portfolio_size=cfg.portfolio_size,
        weight_cap=cfg.weight_cap,
        logger=logger,
    )

    render_reports(
        reports_dir=Path(cfg.reports_dir),
        run_meta=run_meta,
        universe_meta=universe.meta,
        screened=screened,
        weights=weights,
        logger=logger,
    )

    # Persist metadata for debugging/auditing
    write_json(cache_dir / "last_run_meta.json", run_meta)
    logger.info("Wrote reports to %s", reports_dir.resolve())


