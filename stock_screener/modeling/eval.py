from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_dates: list[pd.Timestamp]
    val_dates: list[pd.Timestamp]


@dataclass(frozen=True)
class HoldoutSplit:
    train_dates: list[pd.Timestamp]
    holdout_dates: list[pd.Timestamp]


def _as_dates(dates: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    out = [pd.to_datetime(d).normalize() for d in dates]
    return sorted(list(dict.fromkeys(out)))


def build_time_splits(
    dates: Iterable[pd.Timestamp],
    *,
    n_splits: int = 3,
    val_window: int = 60,
    embargo_days: int = 5,
    min_train: int = 252,
) -> tuple[list[TimeSplit], HoldoutSplit]:
    """Create expanding-window time splits with an embargo and final holdout."""

    uniq = _as_dates(dates)
    if len(uniq) < min_train + val_window + embargo_days + 5:
        raise ValueError("Not enough dates to build time splits")

    splits: list[TimeSplit] = []
    for i in range(n_splits):
        train_end = min_train + i * val_window
        val_start = train_end + embargo_days
        val_end = val_start + val_window
        if val_end > len(uniq):
            break
        splits.append(TimeSplit(train_dates=uniq[:train_end], val_dates=uniq[val_start:val_end]))

    holdout_end = len(uniq)
    holdout_start = max(0, holdout_end - val_window)
    train_end = max(0, holdout_start - embargo_days)
    holdout = HoldoutSplit(train_dates=uniq[:train_end], holdout_dates=uniq[holdout_start:holdout_end])
    return splits, holdout


@dataclass(frozen=True)
class WalkForwardPeriod:
    """A single walk-forward test period."""
    period_id: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]


def build_walk_forward_periods(
    dates: Iterable[pd.Timestamp],
    *,
    n_periods: int = 4,
    test_window: int = 60,
    embargo_days: int = 5,
    min_train: int = 252,
) -> list[WalkForwardPeriod]:
    """Create multiple walk-forward test periods for robust out-of-sample evaluation.
    
    Unlike single holdout, this creates N non-overlapping test periods, each with
    its own training set (all data before the test period). This gives:
    1. More robust evaluation across different market regimes
    2. Larger total out-of-sample data
    3. Detection of regime-specific failures
    """
    uniq = _as_dates(dates)
    total_days = len(uniq)
    
    # Calculate how much space we need for all periods
    space_needed = min_train + (n_periods * test_window) + (n_periods * embargo_days)
    if total_days < space_needed:
        # Reduce number of periods if not enough data
        n_periods = max(1, (total_days - min_train) // (test_window + embargo_days))
    
    if n_periods < 1:
        return []
    
    periods = []
    
    # Work backwards from the end to create non-overlapping periods
    # This ensures the most recent data is always tested
    for i in range(n_periods):
        # Calculate test period boundaries (working backwards)
        test_end_idx = total_days - (i * (test_window + embargo_days))
        test_start_idx = test_end_idx - test_window
        train_end_idx = test_start_idx - embargo_days
        
        if train_end_idx < min_train:
            break  # Not enough training data
        
        period = WalkForwardPeriod(
            period_id=n_periods - i,  # 1-indexed, oldest first
            train_end=uniq[train_end_idx - 1],
            test_start=uniq[test_start_idx],
            test_end=uniq[test_end_idx - 1],
            train_dates=uniq[:train_end_idx],
            test_dates=uniq[test_start_idx:test_end_idx],
        )
        periods.append(period)
    
    # Reverse so oldest period is first
    return list(reversed(periods))


def aggregate_walk_forward_results(period_results: list[dict]) -> dict[str, object]:
    """Aggregate metrics across walk-forward periods.
    
    Returns mean, std, and per-period breakdown for key metrics.
    """
    if not period_results:
        return {"n_periods": 0, "aggregate": {}, "per_period": []}
    
    # Key metrics to aggregate
    metrics_to_agg = [
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "total_return",
        "ann_return", "return_per_day", "cost_adjusted_sharpe",
        "alpha_ann", "alpha_sharpe", "avg_turnover", "turnover_efficiency",
        "mean_ic", "ic_ir",
    ]
    
    aggregated = {}
    for metric in metrics_to_agg:
        values = []
        for result in period_results:
            val = result.get(metric)
            if val is not None and not np.isnan(val):
                values.append(val)
        
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }
    
    # Calculate consistency score (% of periods with positive Sharpe)
    sharpe_values = [r.get("sharpe_ratio", float("nan")) for r in period_results]
    sharpe_positive = sum(1 for s in sharpe_values if s and not np.isnan(s) and s > 0)
    consistency = sharpe_positive / len(sharpe_values) if sharpe_values else 0.0

    # PBO-style robustness proxy:
    # - failure_rate: share of periods with non-positive Sharpe
    # - instability_score: variability of return/day vs absolute mean
    # Lower is better; 0.0 means robust.
    failure_rate = 1.0 - consistency
    return_day_values = [
        float(r.get("return_per_day"))
        for r in period_results
        if r.get("return_per_day") is not None and np.isfinite(float(r.get("return_per_day")))
    ]
    instability_ratio = float("nan")
    if len(return_day_values) >= 2:
        rp = np.array(return_day_values, dtype=float)
        mu_abs = abs(float(np.mean(rp)))
        sigma = float(np.std(rp))
        instability_ratio = sigma / (mu_abs + 1e-8)
    if np.isfinite(instability_ratio):
        instability_score = float(np.clip(1.0 - np.exp(-instability_ratio), 0.0, 1.0))
    else:
        instability_score = 0.5
    pbo_proxy = float(np.clip((0.6 * failure_rate) + (0.4 * instability_score), 0.0, 1.0))

    return {
        "n_periods": len(period_results),
        "consistency": consistency,  # % of periods with positive Sharpe
        "oos_failure_rate": failure_rate,
        "instability_ratio": instability_ratio,
        "pbo_proxy": pbo_proxy,
        "aggregate": aggregated,
        "per_period": period_results,
    }


def evaluate_model_promotion_gates(
    *,
    realistic_metrics: dict[str, object] | None,
    walk_forward_results: dict[str, object] | None,
    calibration_metrics: dict[str, object] | None = None,
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, object]:
    """Evaluate statistical/business gates required for model promotion."""
    realistic_metrics = realistic_metrics or {}
    walk_forward_results = walk_forward_results or {}
    calibration_metrics = calibration_metrics or {}
    thr = {
        "min_return_per_day": 0.0002,
        "min_cost_adjusted_sharpe": 0.5,
        "max_drawdown": -0.25,
        "min_consistency": 0.55,
        "min_turnover_efficiency": 0.20,
        "max_avg_turnover": 0.80,
        "min_periods": 2,
        # Optional calibration gates (disabled unless finite thresholds supplied).
        "max_calibration_error": float("inf"),
        "min_calibration_slope": float("-inf"),
        # Optional overfit robustness gate (disabled unless finite threshold supplied).
        "max_pbo_proxy": float("inf"),
    }
    if thresholds:
        thr.update(thresholds)

    wf_agg = walk_forward_results.get("aggregate", {}) if isinstance(walk_forward_results, dict) else {}

    def _metric_with_wf_mean(metric_name: str, fallback: float) -> float:
        maybe = wf_agg.get(metric_name, {})
        if isinstance(maybe, dict):
            mean_val = maybe.get("mean")
            if mean_val is not None and np.isfinite(float(mean_val)):
                return float(mean_val)
        return float(fallback)

    return_per_day = _metric_with_wf_mean(
        "return_per_day",
        realistic_metrics.get("return_per_day", float("nan")),
    )
    cost_adj_sharpe = _metric_with_wf_mean(
        "cost_adjusted_sharpe",
        realistic_metrics.get("cost_adjusted_sharpe", realistic_metrics.get("sharpe_ratio", float("nan"))),
    )
    max_dd = _metric_with_wf_mean(
        "max_drawdown",
        realistic_metrics.get("max_drawdown", float("nan")),
    )
    consistency = float(walk_forward_results.get("consistency", float("nan")))
    turnover_eff = _metric_with_wf_mean(
        "turnover_efficiency",
        realistic_metrics.get("turnover_efficiency", float("nan")),
    )
    avg_turnover = _metric_with_wf_mean(
        "avg_turnover",
        realistic_metrics.get("avg_turnover", float("nan")),
    )
    n_periods = int(walk_forward_results.get("n_periods", 0))
    calibration_error = float(calibration_metrics.get("calibration_error", float("nan")))
    calibration_slope = float(calibration_metrics.get("calibration_slope", float("nan")))
    pbo_proxy = float(walk_forward_results.get("pbo_proxy", float("nan")))

    gates = [
        {
            "name": "net_return_per_day",
            "operator": ">=",
            "actual": return_per_day,
            "threshold": float(thr["min_return_per_day"]),
            "passed": bool(np.isfinite(return_per_day) and return_per_day >= float(thr["min_return_per_day"])),
        },
        {
            "name": "cost_adjusted_sharpe",
            "operator": ">=",
            "actual": cost_adj_sharpe,
            "threshold": float(thr["min_cost_adjusted_sharpe"]),
            "passed": bool(np.isfinite(cost_adj_sharpe) and cost_adj_sharpe >= float(thr["min_cost_adjusted_sharpe"])),
        },
        {
            "name": "max_drawdown",
            "operator": ">=",
            "actual": max_dd,
            "threshold": float(thr["max_drawdown"]),
            "passed": bool(np.isfinite(max_dd) and max_dd >= float(thr["max_drawdown"])),
        },
        {
            "name": "consistency_positive_sharpe_periods",
            "operator": ">=",
            "actual": consistency,
            "threshold": float(thr["min_consistency"]),
            "passed": bool(np.isfinite(consistency) and consistency >= float(thr["min_consistency"])),
        },
        {
            "name": "turnover_efficiency",
            "operator": ">=",
            "actual": turnover_eff,
            "threshold": float(thr["min_turnover_efficiency"]),
            "passed": bool(np.isfinite(turnover_eff) and turnover_eff >= float(thr["min_turnover_efficiency"])),
        },
        {
            "name": "avg_turnover_cap",
            "operator": "<=",
            "actual": avg_turnover,
            "threshold": float(thr["max_avg_turnover"]),
            "passed": bool(np.isfinite(avg_turnover) and avg_turnover <= float(thr["max_avg_turnover"])),
        },
        {
            "name": "walk_forward_period_count",
            "operator": ">=",
            "actual": float(n_periods),
            "threshold": float(thr["min_periods"]),
            "passed": bool(n_periods >= int(thr["min_periods"])),
        },
    ]

    max_calib_err_thr = float(thr["max_calibration_error"])
    if np.isfinite(max_calib_err_thr):
        gates.append(
            {
                "name": "calibration_error_cap",
                "operator": "<=",
                "actual": calibration_error,
                "threshold": max_calib_err_thr,
                "passed": bool(np.isfinite(calibration_error) and calibration_error <= max_calib_err_thr),
            }
        )

    min_calib_slope_thr = float(thr["min_calibration_slope"])
    if np.isfinite(min_calib_slope_thr):
        gates.append(
            {
                "name": "calibration_slope_floor",
                "operator": ">=",
                "actual": calibration_slope,
                "threshold": min_calib_slope_thr,
                "passed": bool(np.isfinite(calibration_slope) and calibration_slope >= min_calib_slope_thr),
            }
        )

    max_pbo_thr = float(thr["max_pbo_proxy"])
    if np.isfinite(max_pbo_thr):
        gates.append(
            {
                "name": "pbo_proxy_cap",
                "operator": "<=",
                "actual": pbo_proxy,
                "threshold": max_pbo_thr,
                "passed": bool(np.isfinite(pbo_proxy) and pbo_proxy <= max_pbo_thr),
            }
        )

    passed = all(g["passed"] for g in gates)
    return {
        "passed": passed,
        "gates": gates,
        "thresholds": thr,
        "summary": {
            "return_per_day": return_per_day,
            "cost_adjusted_sharpe": cost_adj_sharpe,
            "max_drawdown": max_dd,
            "consistency": consistency,
            "turnover_efficiency": turnover_eff,
            "avg_turnover": avg_turnover,
            "n_periods": n_periods,
            "calibration_error": calibration_error,
            "calibration_slope": calibration_slope,
            "pbo_proxy": pbo_proxy,
        },
    }


def compute_daily_rank_ic(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """Compute per-date Spearman rank IC for predictions."""

    if df.empty:
        return pd.DataFrame(columns=["date", "n", "rank_ic"])

    frame = df[[date_col, label_col, pred_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame = frame.dropna(subset=[label_col, pred_col])

    rows: list[dict[str, float]] = []
    for dt, grp in frame.groupby(date_col):
        if len(grp) < 2:
            continue
        ic = grp[label_col].corr(grp[pred_col], method="spearman")
        if pd.isna(ic):
            continue
        rows.append({"date": dt, "n": float(len(grp)), "rank_ic": float(ic)})

    return pd.DataFrame(rows)


def compute_daily_topn_returns(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
    top_n: int,
    cost_bps: float = 0.0,
    market_ret_col: str | None = None,
    beta_col: str | None = None,
) -> pd.DataFrame:
    """Compute per-date top-N mean returns with optional per-day cost in bps.
    
    If market_ret_col and beta_col are provided, also computes beta-adjusted alpha.
    """

    if df.empty:
        return pd.DataFrame(columns=["date", "n", "mean_ret", "net_ret"])

    cols_needed = [date_col, label_col, pred_col]
    if market_ret_col and market_ret_col in df.columns:
        cols_needed.append(market_ret_col)
    if beta_col and beta_col in df.columns:
        cols_needed.append(beta_col)
    
    frame = df[cols_needed].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame = frame.dropna(subset=[label_col, pred_col])

    n = max(1, int(top_n))
    cost = float(cost_bps) * 1e-4

    rows: list[dict[str, float]] = []
    for dt, grp in frame.groupby(date_col):
        if len(grp) < 1:
            continue
        top = grp.sort_values(pred_col, ascending=False).head(n)
        mean_ret = float(top[label_col].mean())
        net_ret = float(mean_ret - cost)
        
        row = {"date": dt, "n": float(len(top)), "mean_ret": mean_ret, "net_ret": net_ret}
        
        # Compute beta-adjusted alpha if market data available
        if market_ret_col and market_ret_col in top.columns:
            market_ret = float(top[market_ret_col].mean())
            row["market_ret"] = market_ret
            
            if beta_col and beta_col in top.columns:
                portfolio_beta = float(top[beta_col].fillna(1.0).mean())
                # Alpha = portfolio return - beta * market return
                beta_adj_alpha = mean_ret - portfolio_beta * market_ret
                row["portfolio_beta"] = portfolio_beta
                row["beta_adj_alpha"] = beta_adj_alpha
            else:
                # Simple alpha (assumes beta = 1)
                row["simple_alpha"] = mean_ret - market_ret
        
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_rank_ic(daily_ic: pd.DataFrame) -> dict[str, float]:
    if daily_ic.empty:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}
    mean_ic = float(daily_ic["rank_ic"].mean())
    std_ic = float(daily_ic["rank_ic"].std(ddof=0))
    ic_ir = float(mean_ic / std_ic) if std_ic > 0 else float("nan")
    return {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "n_days": int(len(daily_ic))}


def summarize_topn_returns(daily_ret: pd.DataFrame, *, holding_days: int = 1) -> dict[str, float]:
    hold_days = max(1, int(holding_days))
    if daily_ret.empty:
        return {
            "mean_ret": float("nan"),
            "std_ret": float("nan"),
            "ret_ir": float("nan"),
            "mean_net_ret": float("nan"),
            "std_net_ret": float("nan"),
            "net_ret_ir": float("nan"),
            "mean_ret_per_day": float("nan"),
            "mean_net_ret_per_day": float("nan"),
            "ret_ir_per_day": float("nan"),
            "net_ret_ir_per_day": float("nan"),
            "n_days": 0,
            "holding_days": hold_days,
        }
    mean_ret = float(daily_ret["mean_ret"].mean())
    std_ret = float(daily_ret["mean_ret"].std(ddof=0))
    ret_ir = float(mean_ret / std_ret) if std_ret > 0 else float("nan")
    mean_net = float(daily_ret["net_ret"].mean())
    std_net = float(daily_ret["net_ret"].std(ddof=0))
    net_ir = float(mean_net / std_net) if std_net > 0 else float("nan")
    mean_ret_per_day = float(mean_ret / hold_days)
    mean_net_per_day = float(mean_net / hold_days)
    ret_ir_per_day = float(ret_ir / hold_days) if np.isfinite(ret_ir) else float("nan")
    net_ir_per_day = float(net_ir / hold_days) if np.isfinite(net_ir) else float("nan")
    return {
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "ret_ir": ret_ir,
        "mean_net_ret": mean_net,
        "std_net_ret": std_net,
        "net_ret_ir": net_ir,
        "mean_ret_per_day": mean_ret_per_day,
        "mean_net_ret_per_day": mean_net_per_day,
        "ret_ir_per_day": ret_ir_per_day,
        "net_ret_ir_per_day": net_ir_per_day,
        "n_days": int(len(daily_ret)),
        "holding_days": hold_days,
    }


def evaluate_topn_returns(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
    top_n: int,
    cost_bps: float = 0.0,
    holding_days: int = 1,
    market_ret_col: str | None = None,
    beta_col: str | None = None,
) -> dict[str, object]:
    """Evaluate top-N realized returns across dates."""

    daily = compute_daily_topn_returns(
        df, date_col=date_col, label_col=label_col, pred_col=pred_col, top_n=top_n, cost_bps=cost_bps,
        market_ret_col=market_ret_col, beta_col=beta_col,
    )
    summary = summarize_topn_returns(daily, holding_days=holding_days)
    return {"summary": summary, "daily": daily}


def simulate_realistic_portfolio(
    df: pd.DataFrame,
    *,
    date_col: str,
    ticker_col: str,
    label_col: str,
    pred_col: str,
    top_n: int = 30,
    hold_days: int = 5,
    cost_bps: float = 20.0,
    market_ret_col: str | None = None,
) -> dict[str, object]:
    """Simulate realistic portfolio with periodic rebalancing.
    
    Unlike daily top-N which assumes 100% daily turnover, this:
    1. Rebalances every `hold_days` trading days
    2. Tracks actual positions held between rebalances
    3. Applies transaction costs only on trades (not daily)
    4. Computes realistic turnover metrics
    """
    if df.empty:
        return {
            "summary": {
                "total_return": float("nan"),
                "ann_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "avg_turnover": float("nan"),
                "total_cost_bps": float("nan"),
                "n_rebalances": 0,
            },
            "daily": pd.DataFrame(),
        }
    
    frame = df[[date_col, ticker_col, label_col, pred_col]].copy()
    if market_ret_col and market_ret_col in df.columns:
        frame[market_ret_col] = df[market_ret_col]
    
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame = frame.dropna(subset=[label_col, pred_col])
    
    dates = sorted(frame[date_col].unique())
    if len(dates) < hold_days + 1:
        return {
            "summary": {
                "total_return": float("nan"),
                "ann_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "avg_turnover": float("nan"),
                "total_cost_bps": float("nan"),
                "n_rebalances": 0,
            },
            "daily": pd.DataFrame(),
        }
    
    # Rebalance dates (every hold_days)
    rebalance_dates = dates[::hold_days]
    
    n = max(1, int(top_n))
    cost = float(cost_bps) * 1e-4
    
    portfolio_returns = []
    market_returns_list = []
    current_holdings = set()
    total_turnover = 0.0
    n_rebalances = 0
    total_cost_paid = 0.0
    
    for i, date in enumerate(dates):
        day_data = frame[frame[date_col] == date]
        if day_data.empty:
            continue
        
        # Check if rebalance day
        if date in rebalance_dates:
            # Select new top-N
            new_holdings = set(
                day_data.sort_values(pred_col, ascending=False)
                .head(n)[ticker_col]
                .tolist()
            )
            
            # Calculate turnover (% of portfolio changed)
            if current_holdings:
                overlap = len(current_holdings & new_holdings)
                turnover = 1.0 - (overlap / max(len(current_holdings), len(new_holdings)))
                total_turnover += turnover
                
                # Apply transaction costs on changed positions
                positions_changed = len(current_holdings - new_holdings) + len(new_holdings - current_holdings)
                day_cost = cost * (positions_changed / n) if n > 0 else 0
                total_cost_paid += day_cost
            else:
                day_cost = cost  # Initial entry
                total_cost_paid += day_cost
            
            current_holdings = new_holdings
            n_rebalances += 1
        
        # Calculate portfolio return for the day (equal-weighted current holdings)
        if current_holdings:
            held_data = day_data[day_data[ticker_col].isin(current_holdings)]
            if not held_data.empty:
                day_return = float(held_data[label_col].mean())
                
                # Apply cost on rebalance day
                if date in rebalance_dates and n_rebalances > 1:
                    day_return -= day_cost
                
                portfolio_returns.append({"date": date, "return": day_return})
                
                if market_ret_col and market_ret_col in held_data.columns:
                    market_returns_list.append(float(held_data[market_ret_col].mean()))
    
    if not portfolio_returns:
        return {
            "summary": {
                "total_return": float("nan"),
                "ann_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "avg_turnover": float("nan"),
                "total_cost_bps": float("nan"),
                "n_rebalances": 0,
            },
            "daily": pd.DataFrame(),
        }
    
    daily_df = pd.DataFrame(portfolio_returns)
    returns = daily_df["return"]
    
    # Compute metrics
    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)
    n_days = len(returns)
    ann_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else float("nan")
    
    # Sharpe
    if returns.std() > 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    else:
        sharpe = float("nan")
    
    # Max Drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    
    # Average turnover per rebalance
    avg_turnover = float(total_turnover / n_rebalances) if n_rebalances > 0 else 0.0
    
    # Alpha metrics if market data available
    alpha_ann = float("nan")
    if market_returns_list and len(market_returns_list) == len(returns):
        alpha_daily = returns.values - np.array(market_returns_list)
        alpha_ann = float(np.nanmean(alpha_daily) * 252)
    
    summary = {
        "total_return": total_return,
        "ann_return": ann_return,
        "return_per_day": float(total_return / n_days) if n_days > 0 else float("nan"),
        "sharpe_ratio": sharpe,
        "cost_adjusted_sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "turnover_efficiency": float(ann_return / avg_turnover) if avg_turnover > 0 else float("nan"),
        "total_cost_bps": float(total_cost_paid * 10000),
        "n_rebalances": n_rebalances,
        "n_days": n_days,
        "alpha_ann": alpha_ann,
    }
    
    return {"summary": summary, "daily": daily_df}


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
    group_col: str | None = None,
) -> dict[str, object]:
    """Evaluate rank IC across dates; optionally include group breakdowns."""

    daily = compute_daily_rank_ic(df, date_col=date_col, label_col=label_col, pred_col=pred_col)
    summary = summarize_rank_ic(daily)
    payload: dict[str, object] = {"summary": summary, "daily": daily}

    if group_col and group_col in df.columns:
        group_stats: dict[str, dict[str, float]] = {}
        for g, grp in df.groupby(group_col):
            g_daily = compute_daily_rank_ic(grp, date_col=date_col, label_col=label_col, pred_col=pred_col)
            group_stats[str(g)] = summarize_rank_ic(g_daily)
        payload["by_group"] = group_stats

    return payload


def compute_portfolio_metrics(
    daily_returns: pd.Series, 
    rf_annual: float = 0.05,
    daily_alpha: pd.Series | None = None,
    daily_market: pd.Series | None = None,
) -> dict[str, float]:
    """Compute comprehensive portfolio performance metrics including Sharpe, Sortino, max drawdown, and alpha.
    
    Args:
        daily_returns: Portfolio daily returns (absolute)
        rf_annual: Annual risk-free rate (default 5%)
        daily_alpha: Optional beta-adjusted alpha series
        daily_market: Optional market benchmark returns
    """
    if daily_returns.empty or len(daily_returns) < 2:
        return {
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "volatility_ann": float("nan"),
            "total_return": float("nan"),
            "alpha_ann": float("nan"),
            "alpha_sharpe": float("nan"),
            "n_days": 0,
        }
    
    rf_daily = rf_annual / 252  # Convert annual risk-free rate to daily
    excess = daily_returns - rf_daily
    
    # Sharpe Ratio
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else float("nan")
    
    # Sortino Ratio (only penalize downside volatility)
    downside = excess[excess < 0]
    sortino = np.sqrt(252) * excess.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else float("nan")
    
    # Max Drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Volatility (annualized)
    volatility_ann = daily_returns.std() * np.sqrt(252)
    
    # Total Return
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0
    
    # Alpha metrics (if available)
    alpha_ann = float("nan")
    alpha_sharpe = float("nan")
    
    if daily_alpha is not None and len(daily_alpha) > 1:
        alpha_ann = float(daily_alpha.mean() * 252)  # Annualized alpha
        alpha_std = daily_alpha.std()
        if alpha_std > 0:
            alpha_sharpe = float(np.sqrt(252) * daily_alpha.mean() / alpha_std)
    elif daily_market is not None and len(daily_market) == len(daily_returns):
        # Compute simple alpha if beta-adjusted not available
        simple_alpha = daily_returns.values - daily_market.values
        alpha_ann = float(np.nanmean(simple_alpha) * 252)
        alpha_std = float(np.nanstd(simple_alpha))
        if alpha_std > 0:
            alpha_sharpe = float(np.sqrt(252) * np.nanmean(simple_alpha) / alpha_std)
    
    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "volatility_ann": float(volatility_ann),
        "total_return": float(total_return),
        "alpha_ann": alpha_ann,
        "alpha_sharpe": alpha_sharpe,
        "n_days": int(len(daily_returns)),
    }


def compute_calibration(predictions: pd.Series, realized: pd.Series, n_bins: int = 10) -> dict[str, object]:
    """Measure prediction calibration across deciles."""
    if predictions.empty or realized.empty:
        return {
            "calibration_error": float("nan"),
            "expected_calibration_error": float("nan"),
            "directional_brier": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "by_decile": pd.DataFrame(),
        }
    
    df = pd.DataFrame({"pred": predictions, "real": realized}).dropna()
    
    if len(df) < n_bins:
        return {
            "calibration_error": float("nan"),
            "expected_calibration_error": float("nan"),
            "directional_brier": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "by_decile": pd.DataFrame(),
        }
    
    try:
        df["decile"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Not enough unique values for binning
        return {
            "calibration_error": float("nan"),
            "expected_calibration_error": float("nan"),
            "directional_brier": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "by_decile": pd.DataFrame(),
        }
    
    calibration = df.groupby("decile").agg({
        "pred": ["mean", "count"],
        "real": "mean",
    })
    calibration.columns = ["pred_mean", "count", "real_mean"]
    
    # Perfect calibration: pred_mean == real_mean for each decile
    # Use MSE as calibration error
    calibration_error = np.mean((calibration["pred_mean"] - calibration["real_mean"])**2)

    # Expected calibration error (weighted absolute calibration gap).
    count_total = float(calibration["count"].sum())
    if count_total > 0:
        weights = calibration["count"] / count_total
        expected_calibration_error = float(
            np.sum(weights * np.abs(calibration["pred_mean"] - calibration["real_mean"]))
        )
    else:
        expected_calibration_error = float("nan")

    # Reliability slope/intercept from linear fit: real ~= a*pred + b.
    calibration_slope = float("nan")
    calibration_intercept = float("nan")
    pred_std = float(df["pred"].std(ddof=0))
    if np.isfinite(pred_std) and pred_std > 0:
        try:
            slope, intercept = np.polyfit(df["pred"].values.astype(float), df["real"].values.astype(float), 1)
            calibration_slope = float(slope)
            calibration_intercept = float(intercept)
        except Exception:
            pass

    # Directional calibration quality via Brier score of up/down probability.
    directional_brier = float("nan")
    if np.isfinite(pred_std) and pred_std > 0:
        z = (df["pred"] - float(df["pred"].mean())) / pred_std
        p_up = 1.0 / (1.0 + np.exp(-z.clip(-20, 20)))
        y_up = (df["real"] > 0).astype(float)
        directional_brier = float(np.mean(np.square(p_up - y_up)))
    
    return {
        "calibration_error": float(calibration_error),
        "expected_calibration_error": expected_calibration_error,
        "directional_brier": directional_brier,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "by_decile": calibration.reset_index(),
    }
