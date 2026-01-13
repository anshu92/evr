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
    # datetime.fromisoformat preserves embedded timezone offsets if present.
    return datetime.fromisoformat(x)


@dataclass
class Position:
    ticker: str
    entry_price: float
    entry_date: datetime
    shares: int
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    status: str = "OPEN"  # OPEN | CLOSED:<reason>
    exit_price: float | None = None
    exit_date: datetime | None = None
    exit_reason: str | None = None

    def days_held(self, now: datetime | None = None) -> int:
        n = now or _utcnow()
        return (n - self.entry_date).days


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

    data = json.loads(p.read_text(encoding="utf-8"))
    cash = float(data.get("cash_cad", initial_cash_cad))
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
            shares=int(raw.get("shares", 0)),
            stop_loss_pct=raw.get("stop_loss_pct"),
            take_profit_pct=raw.get("take_profit_pct"),
            status=str(raw.get("status", "OPEN")),
            exit_price=raw.get("exit_price"),
            exit_date=_dt_from_iso(raw.get("exit_date")),
            exit_reason=raw.get("exit_reason"),
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
    for pos, raw in zip(state.positions, obj.get("positions", []), strict=False):
        raw["entry_date"] = pos.entry_date.isoformat()
        raw["exit_date"] = pos.exit_date.isoformat() if pos.exit_date else None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


