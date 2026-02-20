from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None


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


def _opt_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def _backup_path(path: Path) -> Path:
    return Path(str(path) + ".bak")


def resolve_portfolio_event_log_path(state_path: str | Path) -> Path:
    p = Path(state_path)
    return Path(str(p) + ".events.jsonl")


@contextmanager
def _state_lock(path: Path, *, exclusive: bool):
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fh.fileno(), mode)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        fh.close()


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd: int | None = None
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.fsync(fd)
    except OSError:
        return
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = str(path) + ".tmp"
    try:
        with open(tmp_name, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
        _fsync_directory(path.parent)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def append_portfolio_events(
    event_log_path: str | Path,
    events: list[dict[str, Any]],
) -> int:
    """Append portfolio action events to a JSONL log with fsync durability."""
    p = Path(event_log_path)
    normalized_lines: list[str] = []
    for raw in events or []:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("action", "")).strip().upper()
        ticker = str(raw.get("ticker", "")).strip().upper()
        if action not in {"BUY", "SELL", "SELL_PARTIAL"}:
            continue
        if not ticker:
            continue

        shares = _opt_float(raw.get("shares"))
        price_cad = _opt_float(raw.get("price_cad"))
        if shares is None or shares <= 0:
            continue
        if price_cad is None or price_cad <= 0:
            continue

        event = dict(raw)
        event["action"] = action
        event["ticker"] = ticker
        event["shares"] = float(shares)
        event["price_cad"] = float(price_cad)

        ts = _dt_from_iso(str(event.get("ts_utc") or "")) or _utcnow()
        event["ts_utc"] = ts.isoformat()
        normalized_lines.append(json.dumps(event, sort_keys=True))

    if not normalized_lines:
        return 0

    with _state_lock(p, exclusive=True):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for line in normalized_lines:
                fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_directory(p.parent)
    return len(normalized_lines)


def rebuild_portfolio_state_from_events(
    state_path: str | Path,
    initial_cash_cad: float = 500.0,
) -> PortfolioState | None:
    """Best-effort state reconstruction from append-only action events."""
    event_path = resolve_portfolio_event_log_path(state_path)
    if not event_path.exists():
        return None

    open_positions: dict[str, Position] = {}
    closed_positions: list[Position] = []
    cash_cad = float(initial_cash_cad)
    last_updated = _utcnow()
    n_events = 0

    with _state_lock(event_path, exclusive=False):
        try:
            lines = event_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue

        action = str(event.get("action", "")).strip().upper()
        ticker = str(event.get("ticker", "")).strip().upper()
        if action not in {"BUY", "SELL", "SELL_PARTIAL"} or not ticker:
            continue

        shares = _opt_float(event.get("shares"))
        price_cad = _opt_float(event.get("price_cad"))
        if shares is None or shares <= 0:
            continue
        if price_cad is None or price_cad <= 0:
            continue

        ts = _dt_from_iso(str(event.get("ts_utc") or "")) or _utcnow()
        reason = str(event.get("reason", action)).strip() or action
        n_events += 1
        last_updated = ts

        if action == "BUY":
            cost = float(price_cad) * float(shares)
            cash_cad -= cost
            pos = open_positions.get(ticker)
            if pos is None:
                open_positions[ticker] = Position(
                    ticker=ticker,
                    entry_price=float(price_cad),
                    entry_date=ts,
                    shares=float(shares),
                )
                continue

            prev_shares = float(pos.shares)
            total_shares = prev_shares + float(shares)
            if total_shares <= 0:
                continue
            pos.entry_price = (
                (float(pos.entry_price) * prev_shares) + (float(price_cad) * float(shares))
            ) / total_shares
            pos.shares = total_shares
            if ts < pos.entry_date:
                pos.entry_date = ts
            continue

        pos = open_positions.get(ticker)
        if pos is None:
            continue

        sell_shares = min(float(shares), float(pos.shares))
        if sell_shares <= 0:
            continue
        cash_cad += float(price_cad) * sell_shares

        remaining = float(pos.shares) - sell_shares
        if remaining <= 1e-9:
            pos.status = f"CLOSED:{reason}"
            pos.exit_price = float(price_cad)
            pos.exit_date = ts
            pos.exit_reason = reason
            closed_positions.append(pos)
            del open_positions[ticker]
            continue

        pos.shares = round(remaining, 8)
        if action == "SELL_PARTIAL":
            pos.last_partial_sell_at = ts

    if n_events == 0:
        return None
    return PortfolioState(
        cash_cad=float(cash_cad),
        positions=list(open_positions.values()) + closed_positions,
        last_updated=last_updated,
        pnl_history=[],
    )


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
    last_partial_sell_at: datetime | None = None  # Cooldown marker for repeated partial exits

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
    bkp = _backup_path(p)
    if not p.exists() and not bkp.exists():
        rebuilt = rebuild_portfolio_state_from_events(p, initial_cash_cad=initial_cash_cad)
        if rebuilt is not None:
            return rebuilt
        return PortfolioState(cash_cad=float(initial_cash_cad), positions=[], last_updated=_utcnow(), pnl_history=[])

    with _state_lock(p, exclusive=False):
        data = _read_json_dict(p)
        if data is None:
            data = _read_json_dict(bkp)
    if data is None:
        rebuilt = rebuild_portfolio_state_from_events(p, initial_cash_cad=initial_cash_cad)
        if rebuilt is not None:
            return rebuilt
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
        entry_price = _opt_float(raw.get("entry_price", 0.0))
        shares = _opt_float(raw.get("shares", 0))
        if entry_price is None or shares is None:
            continue
        pos = Position(
            ticker=str(raw.get("ticker", "")).upper(),
            entry_price=float(entry_price),
            entry_date=_dt_from_iso(entry_date) or _utcnow(),
            shares=float(shares),  # Fractional shares supported
            stop_loss_pct=_opt_float(raw.get("stop_loss_pct")),
            take_profit_pct=_opt_float(raw.get("take_profit_pct")),
            status=str(raw.get("status", "OPEN")),
            exit_price=_opt_float(raw.get("exit_price")),
            exit_date=_dt_from_iso(raw.get("exit_date")),
            exit_reason=raw.get("exit_reason"),
            highest_price=_opt_float(raw.get("highest_price")),
            entry_pred_peak_days=_opt_float(raw.get("entry_pred_peak_days")),
            entry_pred_return=_opt_float(raw.get("entry_pred_return")),
            last_partial_sell_at=_dt_from_iso(raw.get("last_partial_sell_at")),
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
    bkp = _backup_path(p)
    obj: dict[str, Any] = asdict(state)
    obj["last_updated"] = state.last_updated.isoformat()
    for pos, raw in zip(state.positions, obj.get("positions", [])):
        raw["entry_date"] = pos.entry_date.isoformat()
        raw["exit_date"] = pos.exit_date.isoformat() if pos.exit_date else None
        raw["last_partial_sell_at"] = (
            pos.last_partial_sell_at.isoformat() if pos.last_partial_sell_at else None
        )
    payload = json.dumps(obj, indent=2, sort_keys=True)
    with _state_lock(p, exclusive=True):
        _atomic_write_text(p, payload)
        _atomic_write_text(bkp, payload)


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
