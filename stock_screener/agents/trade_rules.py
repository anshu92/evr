"""Professional trading rules — gates and calculations inspired by Minervini, O'Neil, PTJ.

These rules act as pre-trade checks that must pass before the LLM's BUY decision
is executed. They enforce the discipline that separates profitable traders from
gamblers.

Rules:
  1. Market Direction Gate — don't buy into hostile markets
  2. Risk/Reward Gate — reject trades with R:R < 2.0
  3. Risk-Based Position Sizing — size from risk, not labels
  4. Trade Journal — log decisions for learning
  5. Scaling — buy 50%, add 50% on confirmation
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. Market Direction Gate ───────────────────────────────────────────────

def check_market_direction(
    vol_regime: float | None = None,
    market_trend: float | None = None,
    market_breadth: float | None = None,
    *,
    max_vol_regime: float = 1.8,
    min_trend: float = -0.05,
    min_breadth: float = 0.35,
) -> dict[str, Any]:
    """Check if market conditions are favorable for new BUY entries.

    Based on O'Neil: "3 out of 4 stocks follow the general market direction."
    When market is hostile (high vol + downtrend + poor breadth), block new buys.

    Returns dict with 'allow_buys', 'reasons', and individual checks.
    """
    checks: dict[str, Any] = {
        "vol_regime": vol_regime,
        "market_trend": market_trend,
        "market_breadth": market_breadth,
    }
    reasons: list[str] = []

    vol_ok = vol_regime is None or vol_regime <= max_vol_regime
    trend_ok = market_trend is None or market_trend >= min_trend
    breadth_ok = market_breadth is None or market_breadth >= min_breadth

    if not vol_ok:
        reasons.append(f"vol_regime={vol_regime:.2f} > {max_vol_regime} (high stress)")
    if not trend_ok:
        reasons.append(f"market_trend={market_trend:.2%} < {min_trend:.0%} (downtrend)")
    if not breadth_ok:
        reasons.append(f"breadth={market_breadth:.1%} < {min_breadth:.0%} (weak)")

    # Block buys only if MULTIPLE conditions are hostile (not just one)
    hostile_count = sum(1 for ok in [vol_ok, trend_ok, breadth_ok] if not ok)
    allow_buys = hostile_count < 2  # Need at least 2 hostile signals to block

    checks.update({
        "vol_ok": vol_ok,
        "trend_ok": trend_ok,
        "breadth_ok": breadth_ok,
        "hostile_count": hostile_count,
        "allow_buys": allow_buys,
        "reasons": reasons,
    })

    if not allow_buys:
        logger.warning(
            "MARKET GATE: blocking new BUYs — %d hostile signals: %s",
            hostile_count, "; ".join(reasons),
        )
    else:
        logger.info("Market gate: OK (hostile=%d/3)", hostile_count)

    return checks


# ── 2. Risk/Reward Gate ────────────────────────────────────────────────────

def calculate_risk_reward(
    entry_price: float,
    target_price: float | None,
    stop_price: float | None,
    pred_return: float = 0.0,
    stop_loss_pct: float = 0.08,
) -> dict[str, Any]:
    """Calculate risk/reward ratio for a potential trade.

    Based on PTJ: "I seek a 5:1 risk-reward ratio for every trade."
    Minimum acceptable: 2:1.

    If target_price or stop_price is not provided, estimates from
    pred_return and stop_loss_pct.
    """
    if entry_price <= 0:
        return {"rr_ratio": 0.0, "pass": False, "reason": "invalid entry price"}

    # Estimate target if not given
    if target_price is None or target_price <= entry_price:
        target_price = entry_price * (1 + max(pred_return, 0.01))

    # Estimate stop if not given
    if stop_price is None or stop_price <= 0 or stop_price >= entry_price:
        stop_price = entry_price * (1 - stop_loss_pct)

    upside = target_price - entry_price
    downside = entry_price - stop_price

    if downside <= 0:
        return {"rr_ratio": float("inf"), "pass": True, "reason": "no downside risk",
                "entry": entry_price, "target": target_price, "stop": stop_price}

    rr_ratio = upside / downside

    return {
        "rr_ratio": round(rr_ratio, 2),
        "pass": rr_ratio >= 2.0,
        "entry": round(entry_price, 2),
        "target": round(target_price, 2),
        "stop": round(stop_price, 2),
        "upside": round(upside, 2),
        "downside": round(downside, 2),
        "reason": f"R:R {rr_ratio:.1f}:1 {'≥' if rr_ratio >= 2.0 else '<'} 2:1 minimum",
    }


def filter_by_risk_reward(
    tickers: list[str],
    prices: dict[str, float],
    predictions: dict[str, float],
    stop_loss_pct: float = 0.08,
    min_rr: float = 2.0,
    log: logging.Logger | None = None,
) -> tuple[list[str], dict[str, dict]]:
    """Filter tickers by risk/reward ratio. Returns (passed_tickers, rr_details)."""
    _log = log or logger
    passed: list[str] = []
    details: dict[str, dict] = {}

    for t in tickers:
        px = prices.get(t, 0)
        pred = predictions.get(t, 0)
        rr = calculate_risk_reward(px, None, None, pred_return=pred, stop_loss_pct=stop_loss_pct)
        details[t] = rr

        if rr["pass"]:
            passed.append(t)
        else:
            _log.info("R:R GATE rejected %s: %s (pred=%.2f%%, R:R=%.1f:1)",
                       t, rr["reason"], pred * 100, rr["rr_ratio"])

    _log.info("R:R gate: %d/%d passed (min %.1f:1)", len(passed), len(tickers), min_rr)
    return passed, details


# ── 3. Risk-Based Position Sizing ──────────────────────────────────────────

def compute_risk_based_size(
    portfolio_value: float,
    entry_price: float,
    stop_price: float | None = None,
    stop_loss_pct: float = 0.08,
    risk_per_trade_pct: float = 0.02,
    max_position_pct: float = 0.20,
) -> dict[str, Any]:
    """Calculate position size from risk, not arbitrary labels.

    Based on universal rule: "Risk no more than 1-2% of capital per trade."
    Size = (Portfolio × Risk%) / (Entry - Stop)

    Returns dict with shares, weight, dollar_amount, and risk details.
    """
    if portfolio_value <= 0 or entry_price <= 0:
        return {"shares": 0, "weight": 0.0, "dollar_amount": 0.0, "reason": "invalid inputs"}

    if stop_price is None or stop_price <= 0 or stop_price >= entry_price:
        stop_price = entry_price * (1 - stop_loss_pct)

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return {"shares": 0, "weight": 0.0, "dollar_amount": 0.0, "reason": "no risk per share"}

    max_risk_dollars = portfolio_value * risk_per_trade_pct
    max_shares = max_risk_dollars / risk_per_share
    dollar_amount = max_shares * entry_price
    weight = dollar_amount / portfolio_value

    # Cap at max_position_pct
    if weight > max_position_pct:
        weight = max_position_pct
        dollar_amount = portfolio_value * weight
        max_shares = dollar_amount / entry_price

    return {
        "shares": round(max_shares, 4),
        "weight": round(weight, 4),
        "dollar_amount": round(dollar_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "max_risk_dollars": round(max_risk_dollars, 2),
        "stop_price": round(stop_price, 2),
        "risk_pct": risk_per_trade_pct,
        "reason": f"{risk_per_trade_pct:.0%} risk = ${max_risk_dollars:.0f}, {max_shares:.1f} shares @ ${entry_price:.2f}",
    }


# ── 4. Trade Journal ───────────────────────────────────────────────────────

def log_trade_entry(
    journal_path: Path,
    ticker: str,
    action: str,
    entry_price: float,
    shares: float,
    *,
    reason: str = "",
    llm_rating: str = "",
    llm_score: float = 0.0,
    pred_return: float = 0.0,
    stop_price: float | None = None,
    target_price: float | None = None,
    rr_ratio: float | None = None,
    market_regime: dict | None = None,
) -> None:
    """Append a trade entry to the journal (JSONL format).

    Based on: "Record reasoning, entry, exit, and lesson."
    """
    entry = {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "type": "ENTRY",
        "ticker": ticker,
        "action": action,
        "entry_price": entry_price,
        "shares": shares,
        "reason": reason,
        "llm_rating": llm_rating,
        "llm_score": llm_score,
        "pred_return": pred_return,
        "stop_price": stop_price,
        "target_price": target_price,
        "rr_ratio": rr_ratio,
        "market_regime": market_regime,
    }
    _append_journal(journal_path, entry)


def log_trade_exit(
    journal_path: Path,
    ticker: str,
    exit_price: float,
    shares: float,
    *,
    entry_price: float = 0.0,
    reason: str = "",
    days_held: int = 0,
    realized_pnl_pct: float | None = None,
    pred_return_at_entry: float = 0.0,
    actual_return: float | None = None,
) -> None:
    """Append a trade exit to the journal.

    Captures actual vs predicted for learning.
    """
    if actual_return is None and entry_price > 0:
        actual_return = (exit_price - entry_price) / entry_price

    entry = {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "type": "EXIT",
        "ticker": ticker,
        "exit_price": exit_price,
        "shares": shares,
        "entry_price": entry_price,
        "reason": reason,
        "days_held": days_held,
        "realized_pnl_pct": realized_pnl_pct,
        "pred_return_at_entry": pred_return_at_entry,
        "actual_return": actual_return,
        "prediction_error": (
            round(actual_return - pred_return_at_entry, 4)
            if actual_return is not None else None
        ),
    }
    _append_journal(journal_path, entry)


def _append_journal(path: Path, entry: dict) -> None:
    """Append a JSON line to the trade journal file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.warning("Trade journal write failed: %s", e)


# ── 5. Scaling ─────────────────────────────────────────────────────────────

def compute_scaled_entry(
    full_weight: float,
    initial_pct: float = 0.5,
) -> dict[str, float]:
    """Split a position into initial entry and add-on.

    Based on Druckenmiller: "Start small, add as thesis confirms."
    Returns initial_weight and addon_weight.
    """
    initial = full_weight * initial_pct
    addon = full_weight - initial
    return {
        "initial_weight": round(initial, 4),
        "addon_weight": round(addon, 4),
        "initial_pct": initial_pct,
        "full_weight": round(full_weight, 4),
    }
