from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from stock_screener.portfolio.state import PortfolioState, Position, save_portfolio_state
from typing import Any


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
        max_holding_days_hard: int,
        extend_hold_min_pred_return: float | None,
        extend_hold_min_score: float | None,
        max_positions: int,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        logger,
    ) -> None:
        self.state_path = state_path
        self.max_holding_days = max(1, int(max_holding_days))
        self.max_holding_days_hard = max(self.max_holding_days, int(max_holding_days_hard))
        self.extend_hold_min_pred_return = extend_hold_min_pred_return
        self.extend_hold_min_score = extend_hold_min_score
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

    def _sell_position(self, state: PortfolioState, p: Position, *, price_cad: float, reason: str, days_held: int | None) -> TradeAction:
        self._close_position(p, price_cad=price_cad, reason=reason)
        proceeds = float(price_cad) * float(p.shares)
        state.cash_cad = float(state.cash_cad) + float(proceeds)
        return TradeAction(
            ticker=p.ticker,
            action="SELL",
            reason=reason,
            shares=p.shares,
            price_cad=float(price_cad),
            days_held=days_held,
        )

    def _compute_pnl_snapshot(self, state: PortfolioState, *, prices_cad: pd.Series) -> dict[str, Any]:
        realized = 0.0
        unrealized = 0.0
        open_mkt_value = 0.0
        open_cost_basis = 0.0
        n_open = 0
        n_open_priced = 0
        n_closed = 0

        for p in state.positions:
            if not p.ticker or p.shares <= 0:
                continue
            sh = float(p.shares)
            entry = float(p.entry_price or 0.0)

            if p.status == "OPEN":
                n_open += 1
                px = float(prices_cad.get(p.ticker, float("nan")))
                if pd.isna(px) or px <= 0:
                    continue
                n_open_priced += 1
                open_mkt_value += px * sh
                open_cost_basis += entry * sh
                unrealized += (px - entry) * sh
                continue

            n_closed += 1
            if p.exit_price is None:
                continue
            exit_px = float(p.exit_price)
            if exit_px <= 0:
                continue
            realized += (exit_px - entry) * sh

        return {
            "realized_pl_cad": float(realized),
            "unrealized_pl_cad": float(unrealized),
            "net_pl_cad": float(realized + unrealized),
            "open_market_value_cad": float(open_mkt_value),
            "open_cost_basis_cad": float(open_cost_basis),
            "cash_cad": float(state.cash_cad),
            "equity_cad": float(state.cash_cad) + float(open_mkt_value),
            "n_open": int(n_open),
            "n_open_priced": int(n_open_priced),
            "n_closed": int(n_closed),
        }

    def _append_pnl_history(
        self,
        state: PortfolioState,
        *,
        asof_utc: datetime,
        snapshot: dict[str, Any],
        max_points: int = 365,
    ) -> None:
        item = {"asof_utc": asof_utc.isoformat(), **snapshot}
        state.pnl_history.append(item)
        if len(state.pnl_history) > int(max_points):
            state.pnl_history = state.pnl_history[-int(max_points) :]

    def apply_exits(
        self,
        state: PortfolioState,
        prices_cad: pd.Series,
        *,
        pred_return: pd.Series | None = None,
        score: pd.Series | None = None,
    ) -> list[TradeAction]:
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
            # Time exits:
            # - Default behavior: exit at max_holding_days.
            # - Extension behavior: if model signal is strong, allow holding up to max_holding_days_hard.
            if days >= self.max_holding_days:
                if days >= self.max_holding_days_hard:
                    actions.append(self._sell_position(state, p, price_cad=px, reason="TIME_EXIT_HARD", days_held=days))
                    continue

                strong = False
                strong_reason = None
                if pred_return is not None and self.extend_hold_min_pred_return is not None:
                    pr = float(pred_return.get(p.ticker, float("nan")))
                    if not pd.isna(pr) and pr >= float(self.extend_hold_min_pred_return):
                        strong = True
                        strong_reason = f"EXTEND_PRED_RETURN>={self.extend_hold_min_pred_return}"
                if (not strong) and score is not None and self.extend_hold_min_score is not None:
                    sc = float(score.get(p.ticker, float("nan")))
                    if not pd.isna(sc) and sc >= float(self.extend_hold_min_score):
                        strong = True
                        strong_reason = f"EXTEND_SCORE>={self.extend_hold_min_score}"

                if strong:
                    keep.append(p)
                    actions.append(
                        TradeAction(
                            ticker=p.ticker,
                            action="HOLD",
                            reason=strong_reason or "EXTENDED",
                            shares=p.shares,
                            price_cad=px,
                            days_held=days,
                        )
                    )
                    continue

                actions.append(self._sell_position(state, p, price_cad=px, reason="TIME_EXIT", days_held=days))
                continue

            # Optional stop/target exits.
            if p.entry_price > 0:
                ret = px / float(p.entry_price) - 1.0
            else:
                ret = 0.0

            if self.stop_loss_pct is not None and ret <= -abs(float(self.stop_loss_pct)):
                actions.append(self._sell_position(state, p, price_cad=px, reason="STOP_LOSS", days_held=days))
                continue

            if self.take_profit_pct is not None and ret >= abs(float(self.take_profit_pct)):
                actions.append(self._sell_position(state, p, price_cad=px, reason="TAKE_PROFIT", days_held=days))
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
                actions.append(self._sell_position(state, p, price_cad=px, reason="ROTATION", days_held=days))
                open_tickers.discard(t)

        # Update state positions after rotation sells.
        keep_open = [p for p in state.positions if p.status == "OPEN"]
        closed = [p for p in state.positions if p.status != "OPEN"]
        state.positions = keep_open + closed
        open_by_ticker = self._positions_by_ticker(state)
        open_tickers = set(open_by_ticker.keys())

        # BUY missing target tickers up to max_positions
        # Size positions using current portfolio equity (cash + market value of open positions) and target weights.
        open_mkt_value = 0.0
        for p in state.positions:
            if p.status != "OPEN" or not p.ticker or p.shares <= 0:
                continue
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.isna(px) or px <= 0:
                continue
            open_mkt_value += float(px) * float(p.shares)
        equity_cad = float(state.cash_cad) + float(open_mkt_value)

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
                if float(state.cash_cad) < float(px):
                    continue

                w = 0.0
                try:
                    w = float(weights.loc[t, "weight"]) if "weight" in weights.columns else 0.0
                except Exception:
                    w = 0.0

                target_value = float(equity_cad) * float(w) if w > 0 else float(state.cash_cad) / max(slots, 1)
                target_shares = int(max(0.0, target_value) // float(px))
                affordable_shares = int(float(state.cash_cad) // float(px))
                shares = min(affordable_shares, max(1, target_shares))
                if shares <= 0:
                    continue
                cost = float(px) * float(shares)
                state.cash_cad = float(state.cash_cad) - float(cost)
                pos = Position(
                    ticker=t,
                    entry_price=px,
                    entry_date=now,
                    shares=int(shares),
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                )
                state.positions.append(pos)
                actions.append(
                    TradeAction(ticker=t, action="BUY", reason="TOP_RANKED", shares=int(shares), price_cad=px, days_held=0)
                )
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
        try:
            pnl = self._compute_pnl_snapshot(state, prices_cad=prices_cad)
            self._append_pnl_history(state, asof_utc=now, snapshot=pnl)
        except Exception as e:
            # P&L tracking should never break the daily run; fall back gracefully.
            self.logger.warning("Could not compute/append portfolio P&L snapshot: %s", e)
        save_portfolio_state(self.state_path, state)
        return TradePlan(actions=actions, holdings=holdings)


