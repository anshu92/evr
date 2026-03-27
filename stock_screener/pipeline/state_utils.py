from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_screener.portfolio.manager import TradeAction
from stock_screener.portfolio.state import append_portfolio_events, save_portfolio_state


def _compute_open_position_values(
    state,
    prices_cad: pd.Series,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for p in state.positions:
        if p.status != "OPEN" or not p.ticker or p.shares <= 0:
            continue
        px = float(prices_cad.get(p.ticker, float("nan")))
        if pd.isna(px) or px <= 0:
            px = float(getattr(p, "entry_price", 0.0) or 0.0)
        if px <= 0:
            continue
        values[p.ticker] = values.get(p.ticker, 0.0) + (float(px) * float(p.shares))
    return values


def _build_hold_only_target_weights(
    state,
    prices_cad: pd.Series,
    logger,
) -> pd.DataFrame:
    """Build target weights from currently open holdings only (no new entries)."""
    open_values = _compute_open_position_values(state, prices_cad)
    if not open_values:
        logger.warning(
            "Strategy mode HOLD_ONLY: no open positions available; no new buys will be created.",
        )
        return pd.DataFrame({"weight": pd.Series(dtype=float)})

    values = pd.Series(open_values, dtype=float).sort_values(ascending=False)
    total = float(values.sum())
    if total <= 0:
        logger.warning(
            "Strategy mode HOLD_ONLY: open position values are non-positive; no target weights generated.",
        )
        return pd.DataFrame({"weight": pd.Series(dtype=float)})

    out = pd.DataFrame({"weight": (values / total).astype(float)})
    logger.warning(
        "Strategy mode HOLD_ONLY: restricting targets to %d open holding(s); new buys disabled.",
        len(out),
    )
    return out


def _resolve_portfolio_state_path(
    raw_path: str,
    *,
    repo_root: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve portfolio state path consistently across run contexts.

    Relative paths are anchored to repo root by default. For backward compatibility,
    if only the current working directory candidate exists, we keep using it.
    """
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p

    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    cur = cwd if cwd is not None else Path.cwd()
    repo_candidate = (root / p).resolve()
    cwd_candidate = (cur / p).resolve()
    if cwd_candidate.exists() and not repo_candidate.exists():
        return cwd_candidate
    return repo_candidate


def _build_price_history_cad(prices: pd.DataFrame, fx_usdcad: pd.Series) -> pd.DataFrame:
    """Build ticker->CAD close history aligned to trading days for reward backfills."""
    if prices.empty or not isinstance(prices.columns, pd.MultiIndex):
        return pd.DataFrame()
    idx = pd.to_datetime(prices.index).sort_values()
    if idx.empty:
        return pd.DataFrame()

    fx = pd.to_numeric(fx_usdcad, errors="coerce").copy()
    fx.index = pd.to_datetime(fx.index)
    fx = fx.reindex(idx).ffill()

    out: dict[str, pd.Series] = {}
    tickers = list(dict.fromkeys(str(t) for t in prices.columns.get_level_values(0)))
    for t in tickers:
        col = (t, "Close")
        if col not in prices.columns:
            continue
        close = pd.to_numeric(prices[col], errors="coerce")
        is_tsx = str(t).upper().endswith(".TO") or str(t).upper().endswith(".V")
        if is_tsx:
            out[t] = close
        else:
            out[t] = close * fx
    if not out:
        return pd.DataFrame(index=idx)
    return pd.DataFrame(out, index=idx).sort_index()


def _sanitize_trade_actions(actions: list[TradeAction], logger) -> list[TradeAction]:
    """Enforce one-way action consistency before reporting.

    Rules:
    - If a ticker has any SELL/SELL_PARTIAL action, drop BUY/HOLD actions for that ticker.
    - Drop duplicate actions by (ticker, action), keeping the first occurrence.
    """
    if not actions:
        return []

    sell_tickers = {
        str(a.ticker).strip().upper()
        for a in actions
        if a.action in ("SELL", "SELL_PARTIAL") and str(getattr(a, "ticker", "")).strip()
    }

    kept: list[TradeAction] = []
    seen: set[tuple[str, str]] = set()
    dropped_conflicts = 0
    dropped_dupes = 0

    for action in actions:
        ticker = str(getattr(action, "ticker", "")).strip().upper()
        kind = str(getattr(action, "action", "")).strip().upper()
        if not ticker or not kind:
            continue
        if ticker in sell_tickers and kind not in {"SELL", "SELL_PARTIAL"}:
            dropped_conflicts += 1
            continue
        key = (ticker, kind)
        if key in seen:
            dropped_dupes += 1
            continue
        seen.add(key)
        kept.append(action)

    if dropped_conflicts > 0 or dropped_dupes > 0:
        logger.warning(
            "Sanitized trade actions: dropped %d conflicting and %d duplicate action(s).",
            dropped_conflicts,
            dropped_dupes,
        )
    return kept


def _position_changing_actions(actions: list[TradeAction]) -> list[TradeAction]:
    return [
        a for a in (actions or [])
        if str(getattr(a, "action", "")).strip().upper() in {"BUY", "SELL", "SELL_PARTIAL"}
    ]


def _trade_actions_to_event_payloads(
    actions: list[TradeAction],
    *,
    source: str,
    ts_utc: datetime,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for a in _position_changing_actions(actions):
        payloads.append(
            {
                "ts_utc": ts_utc.isoformat(),
                "source": source,
                "action": str(a.action).upper(),
                "ticker": str(a.ticker).upper(),
                "reason": str(a.reason),
                "shares": float(a.shares),
                "price_cad": float(a.price_cad),
                "days_held": a.days_held,
                "pred_return": a.pred_return,
                "entry_price": a.entry_price,
                "realized_gain_pct": a.realized_gain_pct,
                "replaces_ticker": a.replaces_ticker,
                "expected_sell_date": a.expected_sell_date,
            }
        )
    return payloads


def _append_pnl_snapshot(state, *, prices_cad=None, now=None) -> None:
    """Append a P&L snapshot to state.pnl_history.

    Called before each state save so P&L accumulates over time.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    open_positions = [p for p in state.positions if getattr(p, "status", "OPEN") == "OPEN"]
    n_open = len(open_positions)
    n_closed = len([p for p in state.positions if getattr(p, "status", "") != "OPEN"])

    open_mkt_value = 0.0
    realized_pl = 0.0
    unrealized_pl = 0.0
    for p in open_positions:
        # Use prices_cad if available, otherwise estimate from entry price
        if prices_cad is not None and hasattr(prices_cad, "get"):
            px = float(prices_cad.get(p.ticker, p.entry_price))
        else:
            px = p.entry_price
        mv = px * float(p.shares)
        open_mkt_value += mv
        unrealized_pl += (px - p.entry_price) * float(p.shares)

    equity = float(state.cash_cad) + open_mkt_value
    net_pl = equity - 500.0  # vs initial budget (config default)

    entry = {
        "asof_utc": now.isoformat(),
        "equity_cad": equity,
        "cash_cad": float(state.cash_cad),
        "open_market_value_cad": open_mkt_value,
        "realized_pl_cad": realized_pl,
        "unrealized_pl_cad": unrealized_pl,
        "net_pl_cad": net_pl,
        "n_open": n_open,
        "n_closed": n_closed,
    }
    state.pnl_history.append(entry)
    # Keep at most 365 entries
    if len(state.pnl_history) > 365:
        state.pnl_history = state.pnl_history[-365:]


def _persist_state_transition_or_fail(
    *,
    state_path: Path,
    state,
    event_log_path: Path,
    actions: list[TradeAction],
    source: str,
) -> int:
    """Persist action events + state; raise on any failure."""
    now = datetime.now(tz=timezone.utc)
    events = _trade_actions_to_event_payloads(actions, source=source, ts_utc=now)
    if events:
        append_portfolio_events(event_log_path, events)
    # Append P&L snapshot before saving (so history accumulates)
    _append_pnl_snapshot(state, prices_cad=None, now=now)
    save_portfolio_state(state_path, state)
    return len(events)


def _check_kill_switch(logger) -> bool:
    """Check if trading is halted via TRADING_HALT env var or marker file."""
    import os
    if os.getenv("TRADING_HALT", "").strip().lower() in ("1", "true", "yes"):
        logger.warning("KILL SWITCH: TRADING_HALT=1 — pipeline halted")
        return True
    halt_file = Path("HALT_TRADING")
    if halt_file.exists():
        logger.warning("KILL SWITCH: HALT_TRADING file exists — pipeline halted")
        return True
    return False
