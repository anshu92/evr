from __future__ import annotations

import argparse
from datetime import datetime, timezone

from stock_screener.config import Config
from stock_screener.utils import get_logger


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily stock screener + portfolio weights")
    sub = p.add_subparsers(dest="cmd", required=True)

    daily = sub.add_parser("daily", help="Run daily screener + portfolio weights + reports")
    daily.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    train = sub.add_parser("train-model", help="Train and save ML screening model")
    train.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    eval_model = sub.add_parser("eval-model", help="Evaluate current ML model")
    eval_model.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    intraday = sub.add_parser("intraday", help="Lightweight intraday portfolio monitoring")
    intraday.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    macro = sub.add_parser("macro-insights", help="Macro news + standalone macro portfolio + reports")
    macro.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    c2 = sub.add_parser("collective2-copy", help="Run Collective2 USD paper copy-trade workflow")
    c2.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logger = get_logger(level=args.log_level)

    if args.cmd == "daily":
        from stock_screener.pipeline.daily import run_daily

        cfg = Config.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting daily pipeline at %s", started.isoformat())
        run_daily(cfg=cfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished daily pipeline at %s", finished.isoformat())
        return 0

    if args.cmd == "train-model":
        from stock_screener.modeling.train import train_and_save

        cfg = Config.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting model training at %s", started.isoformat())
        res = train_and_save(cfg=cfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished model training at %s", finished.isoformat())
        logger.info("TrainResult: samples=%s tickers=%s horizon_days=%s", res.n_samples, res.n_tickers, res.horizon_days)
        return 0

    if args.cmd == "eval-model":
        from stock_screener.modeling.train import evaluate_model

        cfg = Config.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting model evaluation at %s", started.isoformat())
        metrics = evaluate_model(cfg=cfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished model evaluation at %s", finished.isoformat())
        ranker = metrics.get("ranker")
        regressor = metrics.get("regressor")
        if ranker:
            logger.info("Ranker IC summary: %s", ranker.get("summary"))
            topn = ranker.get("topn", {}).get("summary") if isinstance(ranker, dict) else None
            if topn:
                logger.info("Ranker Top-N returns: %s", topn)
        if regressor:
            logger.info("Regressor IC summary: %s", regressor.get("summary"))
            topn = regressor.get("topn", {}).get("summary") if isinstance(regressor, dict) else None
            if topn:
                logger.info("Regressor Top-N returns: %s", topn)
        return 0

    if args.cmd == "intraday":
        from stock_screener.pipeline.daily import run_intraday
        cfg = Config.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting intraday trading pipeline at %s", started.isoformat())
        run_intraday(cfg=cfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        elapsed = (finished - started).total_seconds()
        logger.info("Finished intraday trading pipeline in %.1fs", elapsed)
        return 0

    if args.cmd == "macro-insights":
        from stock_screener.macro.config import MacroConfig as MacroCfg
        from stock_screener.pipeline.macro_insights import run_macro_insights

        mcfg = MacroCfg.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting macro insights pipeline at %s", started.isoformat())
        run_macro_insights(cfg=mcfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished macro insights pipeline at %s", finished.isoformat())
        return 0

    if args.cmd == "collective2-copy":
        from stock_screener.pipeline.collective2_copy import (
            Collective2CopyConfig,
            run_collective2_copy,
        )

        ccfg = Collective2CopyConfig.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting Collective2 copy-trade paper pipeline at %s", started.isoformat())
        run_collective2_copy(cfg=ccfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished Collective2 copy-trade paper pipeline at %s", finished.isoformat())
        return 0

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
