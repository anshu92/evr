from __future__ import annotations

import argparse
from datetime import datetime, timezone

from stock_screener.config import Config
from stock_screener.pipeline.daily import run_daily
from stock_screener.modeling.train import evaluate_model, train_and_save
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
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logger = get_logger(level=args.log_level)

    if args.cmd == "daily":
        cfg = Config.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting daily pipeline at %s", started.isoformat())
        run_daily(cfg=cfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished daily pipeline at %s", finished.isoformat())
        return 0

    if args.cmd == "train-model":
        cfg = Config.from_env()
        started = datetime.now(tz=timezone.utc)
        logger.info("Starting model training at %s", started.isoformat())
        res = train_and_save(cfg=cfg, logger=logger)
        finished = datetime.now(tz=timezone.utc)
        logger.info("Finished model training at %s", finished.isoformat())
        logger.info("TrainResult: samples=%s tickers=%s horizon_days=%s", res.n_samples, res.n_tickers, res.horizon_days)
        return 0

    if args.cmd == "eval-model":
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
        if regressor:
            logger.info("Regressor IC summary: %s", regressor.get("summary"))
        return 0

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())


