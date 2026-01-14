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
from stock_screener.modeling.model import FEATURE_COLUMNS, build_model, build_ranker, save_bundle, save_model
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



def _make_ranker_labels(df: pd.DataFrame, *, bins: int = 5) -> pd.Series:
    """Convert continuous returns into integer relevance labels per date."""

    if df.empty:
        return pd.Series(dtype=int)

    def _bin_group(group: pd.DataFrame) -> pd.Series:
        if len(group) <= 1:
            return pd.Series([0] * len(group), index=group.index, dtype=int)
        ranks = group["future_ret"].rank(pct=True, method="average")
        labels = (ranks * bins).fillna(0).astype(float)
        labels = labels.clip(lower=0, upper=bins - 1).astype(int)
        return pd.Series(labels, index=group.index, dtype=int)

    return df.groupby("date", group_keys=False).apply(_bin_group)


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
        if fundamentals is not None and t in fundamentals.index:
            row = fundamentals.loc[t]
            sector_hash = _hash_to_float(str(row.get("sector"))) if row.get("sector") else np.nan
            industry_hash = _hash_to_float(str(row.get("industry"))) if row.get("industry") else np.nan
            market_cap = row.get("marketCap")
            log_market_cap = float(np.log10(market_cap)) if market_cap and float(market_cap) > 0 else np.nan
            beta = float(row.get("beta")) if row.get("beta") is not None else np.nan

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

    ranker_candidates = [
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5},
        {"max_depth": 5, "learning_rate": 0.07, "min_child_weight": 8},
    ]
    regressor_candidates = [
        {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 5},
        {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 8},
    ]

    def _eval_ranker_params(params: dict[str, float]) -> float:
        scores: list[float] = []
        for split in splits:
            train_df = _subset_by_dates(split.train_dates)
            val_df = _subset_by_dates(split.val_dates)
            if train_df.empty or val_df.empty:
                continue
            model = build_ranker(random_state=42)
            model.set_params(**params, early_stopping_rounds=50)
            train_groups = train_df.groupby("date").size().to_numpy()
            val_groups = val_df.groupby("date").size().to_numpy()
            train_labels = _make_ranker_labels(train_df)
            val_labels = _make_ranker_labels(val_df)
            model.fit(
                train_df[FEATURE_COLUMNS],
                train_labels,
                group=train_groups,
                eval_set=[(val_df[FEATURE_COLUMNS], val_labels)],
                eval_group=[val_groups],
                verbose=False,
            )
            preds = model.predict(val_df[FEATURE_COLUMNS])
            metrics = _rank_ic_for_preds(val_df, pd.Series(preds, index=val_df.index))
            scores.append(float(metrics["summary"]["mean_ic"]))
        return float(np.nanmean(scores)) if scores else float("nan")

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

    ranker_scores = {str(p): _eval_ranker_params(p) for p in ranker_candidates}
    regressor_scores = {str(p): _eval_regressor_params(p) for p in regressor_candidates}
    best_ranker_params = max(ranker_candidates, key=lambda p: ranker_scores.get(str(p), float("-inf")))
    best_regressor_params = max(regressor_candidates, key=lambda p: regressor_scores.get(str(p), float("-inf")))

    train_df = _subset_by_dates(holdout.train_dates)
    val_df = _subset_by_dates(val_dates)
    holdout_df = _subset_by_dates(holdout.holdout_dates)

    ranker = build_ranker(random_state=42)
    ranker.set_params(**best_ranker_params, early_stopping_rounds=50)
    ranker_groups = train_df.groupby("date").size().to_numpy()
    if not val_df.empty:
        val_groups = val_df.groupby("date").size().to_numpy()
        train_labels = _make_ranker_labels(train_df)
        val_labels = _make_ranker_labels(val_df)
        ranker.fit(
            train_df[FEATURE_COLUMNS],
            train_labels,
            group=ranker_groups,
            eval_set=[(val_df[FEATURE_COLUMNS], val_labels)],
            eval_group=[val_groups],
            verbose=False,
        )
    else:
        train_labels = _make_ranker_labels(train_df)
        ranker.fit(train_df[FEATURE_COLUMNS], train_labels, group=ranker_groups)

    ranker_holdout_preds = ranker.predict(holdout_df[FEATURE_COLUMNS]) if not holdout_df.empty else []
    ranker_holdout_metrics = _rank_ic_for_preds(holdout_df, pd.Series(ranker_holdout_preds, index=holdout_df.index))

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

    ranker_rel = "ranker.json"
    save_model(ranker, model_dir / ranker_rel)

    metadata = {
        "horizon_days": horizon,
        "feature_columns": FEATURE_COLUMNS,
        "filters": {
            "min_price_cad": float(cfg.min_price_cad),
            "min_avg_dollar_volume_cad": float(cfg.min_avg_dollar_volume_cad),
            "min_history_days": 90,
        },
        "ranker": {"params": best_ranker_params, "cv_scores": ranker_scores, "holdout": ranker_holdout_metrics["summary"]},
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
    }
    metadata_rel = "metrics.json"
    write_json(model_dir / metadata_rel, metadata)

    save_bundle(
        cfg.model_path,
        ranker_rel_path=ranker_rel,
        regressor_rel_paths=reg_rel_paths,
        regressor_weights=None,
        metadata_rel_path=metadata_rel,
    )
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

