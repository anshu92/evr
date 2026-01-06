from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from stock_screener.portfolio.state import PortfolioState, Position, save_portfolio_state


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class TradeAction:
    ticker: str
    action: str  # BUY | SELL | HOLD
    reason: str
    shares: int
    price_cad: float
    days_held: int | None = None


@dataclass(frozen=True)
class TradePlan:
    actions: list[TradeAction]
    holdings: pd.DataFrame  # index=ticker, includes weight/score/metrics and days_held/entry_price


class PortfolioManager:
    def __init__(
        self,
        *,
        state_path: str,
        max_holding_days: int,
        max_positions: int,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        logger,
    ) -> None:
        self.state_path = state_path
        self.max_holding_days = max(1, int(max_holding_days))
        self.max_positions = max(1, int(max_positions))
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.logger = logger

    def _positions_by_ticker(self, state: PortfolioState) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in state.positions:
            if p.status == "OPEN" and p.ticker:
                out[p.ticker] = p
        return out

    def _close_position(self, p: Position, *, price_cad: float, reason: str) -> Position:
        p.status = f"CLOSED:{reason}"
        p.exit_price = float(price_cad)
        p.exit_date = _utcnow()
        p.exit_reason = reason
        return p

    def apply_exits(self, state: PortfolioState, prices_cad: pd.Series) -> list[TradeAction]:
        actions: list[TradeAction] = []
        now = _utcnow()
        open_positions: list[Position] = [p for p in state.positions if p.status == "OPEN"]
        keep: list[Position] = []

        for p in open_positions:
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.isna(px) or px <= 0:
                keep.append(p)
                continue

            days = p.days_held(now)
            # Priority: time exit first (enforces <= max_holding_days).
            if days >= self.max_holding_days:
                self._close_position(p, price_cad=px, reason="TIME_EXIT")
                actions.append(
                    TradeAction(ticker=p.ticker, action="SELL", reason="TIME_EXIT", shares=p.shares, price_cad=px, days_held=days)
                )
                continue

            # Optional stop/target exits.
            if p.entry_price > 0:
                ret = px / float(p.entry_price) - 1.0
            else:
                ret = 0.0

            if self.stop_loss_pct is not None and ret <= -abs(float(self.stop_loss_pct)):
                self._close_position(p, price_cad=px, reason="STOP_LOSS")
                actions.append(
                    TradeAction(ticker=p.ticker, action="SELL", reason="STOP_LOSS", shares=p.shares, price_cad=px, days_held=days)
                )
                continue

            if self.take_profit_pct is not None and ret >= abs(float(self.take_profit_pct)):
                self._close_position(p, price_cad=px, reason="TAKE_PROFIT")
                actions.append(
                    TradeAction(ticker=p.ticker, action="SELL", reason="TAKE_PROFIT", shares=p.shares, price_cad=px, days_held=days)
                )
                continue

            keep.append(p)

        # Replace state positions: keep open positions + closed positions from this run.
        closed = [p for p in state.positions if p.status != "OPEN"]
        state.positions = keep + closed
        return actions

    def build_trade_plan(
        self,
        *,
        state: PortfolioState,
        screened: pd.DataFrame,
        weights: pd.DataFrame,
        prices_cad: pd.Series,
    ) -> TradePlan:
        # weights is expected to represent the *target* holdings universe, but we may rotate.
        now = _utcnow()
        actions: list[TradeAction] = []

        open_by_ticker = self._positions_by_ticker(state)
        open_tickers = set(open_by_ticker.keys())

        target_tickers = list(weights.index.astype(str))
        target_set = set(target_tickers)

        # SELL anything not in target (rotation), after exits have already been applied.
        for t, p in list(open_by_ticker.items()):
            if t not in target_set:
                px = float(prices_cad.get(t, float("nan")))
                if pd.isna(px) or px <= 0:
                    continue
                days = p.days_held(now)
                self._close_position(p, price_cad=px, reason="ROTATION")
                actions.append(TradeAction(ticker=t, action="SELL", reason="ROTATION", shares=p.shares, price_cad=px, days_held=days))
                open_tickers.discard(t)

        # Update state positions after rotation sells.
        keep_open = [p for p in state.positions if p.status == "OPEN"]
        closed = [p for p in state.positions if p.status != "OPEN"]
        state.positions = keep_open + closed
        open_by_ticker = self._positions_by_ticker(state)
        open_tickers = set(open_by_ticker.keys())

        # BUY missing target tickers up to max_positions
        slots = max(self.max_positions - len(open_tickers), 0)
        if slots > 0:
            for t in target_tickers:
                if slots <= 0:
                    break
                if t in open_tickers:
                    continue
                px = float(prices_cad.get(t, float("nan")))
                if pd.isna(px) or px <= 0:
                    continue
                # Shares are a placeholder for now; we treat weights as the executable sizing instruction.
                # Keep shares=1 to avoid implying a specific capital sizing scheme.
                pos = Position(
                    ticker=t,
                    entry_price=px,
                    entry_date=now,
                    shares=1,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                )
                state.positions.append(pos)
                actions.append(TradeAction(ticker=t, action="BUY", reason="TOP_RANKED", shares=1, price_cad=px, days_held=0))
                open_tickers.add(t)
                slots -= 1

        # HOLD: open tickers in target.
        for t in sorted(open_tickers):
            if t not in target_set:
                continue
            p = open_by_ticker.get(t)
            px = float(prices_cad.get(t, float("nan")))
            days = p.days_held(now) if p else None
            actions.append(TradeAction(ticker=t, action="HOLD", reason="IN_TARGET", shares=p.shares if p else 0, price_cad=px, days_held=days))

        # Build holdings view: join weights with state-derived fields.
        holdings = weights.copy()
        holdings["days_held"] = pd.NA
        holdings["entry_price_cad"] = pd.NA
        for t, p in self._positions_by_ticker(state).items():
            if t in holdings.index:
                holdings.loc[t, "days_held"] = int(p.days_held(now))
                holdings.loc[t, "entry_price_cad"] = float(p.entry_price)

        state.last_updated = now
        save_portfolio_state(self.state_path, state)
        return TradePlan(actions=actions, holdings=holdings)


