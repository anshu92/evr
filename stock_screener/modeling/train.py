from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_screener.config import Config
from stock_screener.data.fx import fetch_usdcad
from stock_screener.data.prices import download_price_history
from stock_screener.modeling.model import FEATURE_COLUMNS, build_model, save_ensemble, save_model
from stock_screener.universe.tsx import fetch_tsx_universe
from stock_screener.universe.us import fetch_us_universe
from stock_screener.utils import Universe, ensure_dir


@dataclass(frozen=True)
class TrainResult:
    n_samples: int
    n_tickers: int
    horizon_days: int


def _is_tsx_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return t.endswith(".TO") or t.endswith(".V")


def _build_panel_features(prices: pd.DataFrame, fx_usdcad: pd.Series) -> pd.DataFrame:
    # prices: columns (ticker, field). We compute rolling features per ticker per date.
    idx = pd.to_datetime(prices.index).sort_values()
    fx = fx_usdcad.copy()
    fx.index = pd.to_datetime(fx.index)
    fx = fx.reindex(idx).ffill()

    frames: list[pd.DataFrame] = []
    tickers = list(prices.columns.levels[0])
    for t in tickers:
        if (t, "Close") not in prices.columns:
            continue
        close = prices[(t, "Close")].astype(float)
        vol = prices[(t, "Volume")].astype(float) if (t, "Volume") in prices.columns else pd.Series(index=idx, dtype=float)
        is_tsx = _is_tsx_ticker(t)
        fx_factor = 1.0 if is_tsx else fx
        close_cad = close * fx_factor

        rets = close.pct_change()
        vol_20 = rets.rolling(20).std(ddof=0) * np.sqrt(252.0)
        vol_60 = rets.rolling(60).std(ddof=0) * np.sqrt(252.0)

        # RSI(14)
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / roll_down.replace(0.0, np.nan)
        rsi_14 = 100.0 - (100.0 / (1.0 + rs))

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        df = pd.DataFrame(
            {
                "date": idx,
                "ticker": t,
                "is_tsx": int(is_tsx),
                "last_close_cad": close_cad.values,
                "avg_dollar_volume_cad": (close_cad * vol).rolling(30).mean().values,
                "ret_20d": close.pct_change(20).values,
                "ret_60d": close.pct_change(60).values,
                "ret_120d": close.pct_change(120).values,
                "vol_20d_ann": vol_20.values,
                "vol_60d_ann": vol_60.values,
                "rsi_14": rsi_14.values,
                "ma20_ratio": (close / ma20 - 1.0).values,
                "ma50_ratio": (close / ma50 - 1.0).values,
                "ma200_ratio": (close / ma200 - 1.0).values,
            }
        )
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel


def train_and_save(cfg: Config, logger) -> TrainResult:
    """Train an ensemble of ML models and write manifest to cfg.model_path."""

    cache_dir = ensure_dir(cfg.cache_dir)
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

    panel = _build_panel_features(prices=prices, fx_usdcad=fx)
    if panel.empty:
        raise RuntimeError("No training panel built")

    horizon = int(cfg.label_horizon_days)
    panel["future_ret"] = panel.groupby("ticker")["last_close_cad"].shift(-horizon) / panel["last_close_cad"] - 1.0

    # Drop rows without labels/features
    panel = panel.dropna(subset=["future_ret"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["last_close_cad", "avg_dollar_volume_cad"])

    x = panel[FEATURE_COLUMNS]
    y = panel["future_ret"].astype(float)

    # Ensemble: same architecture, different random seeds. This is a robust SOTA baseline for tabular alpha models.
    # Using 7 members for better stability without blowing up GitHub Actions runtime.
    seeds = [7, 13, 21, 42, 73, 99, 123]
    model_dir = Path(cfg.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []

    logger.info(
        "Training ML ensemble on %s samples, %s tickers (seeds=%s)",
        len(panel),
        panel["ticker"].nunique(),
        seeds,
    )
    for s in seeds:
        m = build_model(random_state=s)
        m.fit(x, y)
        rel = f"model_seed_{s}.json"
        save_model(m, model_dir / rel)
        rel_paths.append(rel)

    save_ensemble(cfg.model_path, rel_paths, weights=None)
    logger.info("Saved ensemble manifest to %s", cfg.model_path)
    return TrainResult(n_samples=int(len(panel)), n_tickers=int(panel["ticker"].nunique()), horizon_days=horizon)


