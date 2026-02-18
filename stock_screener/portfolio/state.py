from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _dt_from_iso(x: str | None) -> datetime | None:
    if not x:
        return None
    # Keep all timestamps timezone-aware and normalized to UTC.
    try:
        dt = datetime.fromisoformat(str(x))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class Position:
    ticker: str
    entry_price: float
    entry_date: datetime
    shares: float  # Fractional shares supported for expensive stocks
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    status: str = "OPEN"  # OPEN | CLOSED:<reason>
    exit_price: float | None = None
    exit_date: datetime | None = None
    exit_reason: str | None = None
    highest_price: float | None = None  # Peak price for trailing stop
    entry_pred_peak_days: float | None = None  # Predicted peak day at entry
    entry_pred_return: float | None = None  # Predicted return at entry (for sell target)

    def days_held(self, now: datetime | None = None) -> int:
        n = now or _utcnow()
        return (n - self.entry_date).days
    
    def update_highest_price(self, current_price: float) -> None:
        """Update highest price for trailing stop calculation."""
        if current_price > 0:
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price


@dataclass
class PortfolioState:
    cash_cad: float
    positions: list[Position]
    last_updated: datetime
    pnl_history: list[dict[str, Any]] = field(default_factory=list)


def load_portfolio_state(path: str | Path, initial_cash_cad: float = 500.0) -> PortfolioState:
    """Load portfolio state or create a new one."""
    p = Path(path)
    if not p.exists():
        return PortfolioState(cash_cad=float(initial_cash_cad), positions=[], last_updated=_utcnow(), pnl_history=[])

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return PortfolioState(cash_cad=float(initial_cash_cad), positions=[], last_updated=_utcnow(), pnl_history=[])
    if not isinstance(data, dict):
        return PortfolioState(cash_cad=float(initial_cash_cad), positions=[], last_updated=_utcnow(), pnl_history=[])

    try:
        cash = float(data.get("cash_cad", initial_cash_cad))
    except (TypeError, ValueError):
        cash = float(initial_cash_cad)
    last_updated = _dt_from_iso(data.get("last_updated")) or _utcnow()

    positions: list[Position] = []
    for raw in data.get("positions", []) or []:
        if not isinstance(raw, dict):
            continue
        entry_date = raw.get("entry_date")
        pos = Position(
            ticker=str(raw.get("ticker", "")).upper(),
            entry_price=float(raw.get("entry_price", 0.0)),
            entry_date=_dt_from_iso(entry_date) or _utcnow(),
            shares=float(raw.get("shares", 0)),  # Fractional shares supported
            stop_loss_pct=raw.get("stop_loss_pct"),
            take_profit_pct=raw.get("take_profit_pct"),
            status=str(raw.get("status", "OPEN")),
            exit_price=raw.get("exit_price"),
            exit_date=_dt_from_iso(raw.get("exit_date")),
            exit_reason=raw.get("exit_reason"),
            highest_price=raw.get("highest_price"),
            entry_pred_peak_days=raw.get("entry_pred_peak_days"),
            entry_pred_return=raw.get("entry_pred_return"),
        )
        if pos.ticker and pos.shares > 0:
            positions.append(pos)

    pnl_history_raw = data.get("pnl_history", []) or []
    pnl_history: list[dict[str, Any]] = []
    if isinstance(pnl_history_raw, list):
        for item in pnl_history_raw:
            if not isinstance(item, dict):
                continue
            asof = item.get("asof_utc")
            if not isinstance(asof, str) or not asof.strip():
                continue
            # Keep the payload small and robust to schema tweaks.
            cleaned: dict[str, Any] = {"asof_utc": asof.strip()}
            for k in [
                "realized_pl_cad",
                "unrealized_pl_cad",
                "net_pl_cad",
                "open_market_value_cad",
                "open_cost_basis_cad",
                "cash_cad",
                "equity_cad",
                "n_open",
                "n_open_priced",
                "n_closed",
            ]:
                if k in item:
                    cleaned[k] = item[k]
            pnl_history.append(cleaned)

    return PortfolioState(cash_cad=cash, positions=positions, last_updated=last_updated, pnl_history=pnl_history)


def save_portfolio_state(path: str | Path, state: PortfolioState) -> None:
    """Save portfolio state to JSON."""
    p = Path(path)
    obj: dict[str, Any] = asdict(state)
    obj["last_updated"] = state.last_updated.isoformat()
    for pos, raw in zip(state.positions, obj.get("positions", [])):
        raw["entry_date"] = pos.entry_date.isoformat()
        raw["exit_date"] = pos.exit_date.isoformat() if pos.exit_date else None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def compute_portfolio_drawdown(state: PortfolioState) -> dict[str, float]:
    """Compute current drawdown from portfolio P&L history.
    
    Returns:
        Dict with 'current_drawdown', 'max_equity', 'current_equity', 'drawdown_scalar'
    """
    if not state.pnl_history:
        return {
            "current_drawdown": 0.0,
            "max_equity": 0.0,
            "current_equity": 0.0,
            "drawdown_scalar": 1.0,
            "days_in_drawdown": 0,
        }
    
    # Extract equity history
    equities = []
    for entry in state.pnl_history:
        equity = entry.get("equity_cad")
        if equity is not None and equity > 0:
            equities.append(float(equity))
    
    if not equities:
        return {
            "current_drawdown": 0.0,
            "max_equity": 0.0,
            "current_equity": 0.0,
            "drawdown_scalar": 1.0,
            "days_in_drawdown": 0,
        }
    
    current_equity = equities[-1]
    max_equity = max(equities)
    
    if max_equity <= 0:
        return {
            "current_drawdown": 0.0,
            "max_equity": 0.0,
            "current_equity": current_equity,
            "drawdown_scalar": 1.0,
            "days_in_drawdown": 0,
        }
    
    # Calculate drawdown (negative value)
    current_drawdown = (current_equity - max_equity) / max_equity
    
    # Count days in drawdown
    days_in_dd = 0
    for eq in reversed(equities):
        if eq < max_equity * 0.99:  # 1% tolerance
            days_in_dd += 1
        else:
            break
    
    return {
        "current_drawdown": current_drawdown,
        "max_equity": max_equity,
        "current_equity": current_equity,
        "drawdown_scalar": 1.0,  # Will be computed by caller
        "days_in_drawdown": days_in_dd,
    }


def compute_drawdown_scalar(
    state: PortfolioState,
    max_drawdown_threshold: float = -0.10,
    min_scalar: float = 0.25,
    recovery_threshold: float = -0.02,
) -> tuple[float, dict]:
    """Compute position sizing scalar based on portfolio drawdown.
    
    When in drawdown, reduce position sizes to limit further losses.
    
    Args:
        state: Portfolio state with P&L history
        max_drawdown_threshold: Drawdown at which to apply minimum scalar (e.g., -0.10 = -10%)
        min_scalar: Minimum position sizing multiplier (e.g., 0.25 = 25% normal size)
        recovery_threshold: Drawdown threshold to return to normal sizing (e.g., -0.02 = -2%)
        
    Returns:
        Tuple of (scalar, drawdown_info dict)
    """
    dd_info = compute_portfolio_drawdown(state)
    current_dd = dd_info["current_drawdown"]
    
    # No drawdown or recovered
    if current_dd >= recovery_threshold:
        dd_info["drawdown_scalar"] = 1.0
        return 1.0, dd_info
    
    # In significant drawdown - scale down linearly
    # At max_drawdown_threshold, use min_scalar
    # Between recovery_threshold and max_drawdown_threshold, interpolate
    if current_dd <= max_drawdown_threshold:
        scalar = min_scalar
    else:
        # Linear interpolation
        dd_range = recovery_threshold - max_drawdown_threshold
        dd_position = (current_dd - max_drawdown_threshold) / dd_range
        scalar = min_scalar + (1.0 - min_scalar) * dd_position
    
    dd_info["drawdown_scalar"] = scalar
    return scalar, dd_info
