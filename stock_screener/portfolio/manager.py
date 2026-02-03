from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pandas as pd

from stock_screener.portfolio.state import PortfolioState, Position, save_portfolio_state
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class TradeAction:
    ticker: str
    action: str  # BUY | SELL | SELL_PARTIAL | HOLD
    reason: str
    shares: int
    price_cad: float
    days_held: int | None = None
    pred_return: float | None = None  # Predicted return for this position
    expected_sell_date: str | None = None  # Expected sell date (based on max hold days)


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
        trailing_stop_enabled: bool = True,
        trailing_stop_activation_pct: float = 0.05,
        trailing_stop_distance_pct: float = 0.08,
        peak_based_exit: bool = True,
        twr_optimization: bool = True,
        quick_profit_pct: float = 0.05,
        quick_profit_days: int = 3,
        min_daily_return: float = 0.005,
        momentum_decay_exit: bool = True,
        signal_decay_exit_enabled: bool = True,
        signal_decay_threshold: float = -0.02,
        dynamic_holding_enabled: bool = True,
        dynamic_holding_vol_scale: float = 0.5,
        vol_adjusted_stop_enabled: bool = True,
        vol_adjusted_stop_base: float = 0.08,
        vol_adjusted_stop_min: float = 0.04,
        vol_adjusted_stop_max: float = 0.15,
        age_urgency_enabled: bool = True,
        age_urgency_start_day: int = 2,
        age_urgency_min_return: float = 0.01,
        peak_detection_enabled: bool,
        peak_sell_portion_pct: float,
        peak_min_gain_pct: float | None,
        peak_min_holding_days: int,
        peak_pred_return_threshold: float | None,
        peak_score_percentile_drop: float | None,
        peak_rsi_overbought: float | None,
        peak_above_ma_ratio: float | None,
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
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.peak_based_exit = peak_based_exit
        self.twr_optimization = twr_optimization
        self.quick_profit_pct = quick_profit_pct
        self.quick_profit_days = max(1, int(quick_profit_days))
        self.min_daily_return = min_daily_return
        self.momentum_decay_exit = momentum_decay_exit
        self.signal_decay_exit_enabled = signal_decay_exit_enabled
        self.signal_decay_threshold = signal_decay_threshold
        self.dynamic_holding_enabled = dynamic_holding_enabled
        self.dynamic_holding_vol_scale = dynamic_holding_vol_scale
        self.vol_adjusted_stop_enabled = vol_adjusted_stop_enabled
        self.vol_adjusted_stop_base = vol_adjusted_stop_base
        self.vol_adjusted_stop_min = vol_adjusted_stop_min
        self.vol_adjusted_stop_max = vol_adjusted_stop_max
        self.age_urgency_enabled = age_urgency_enabled
        self.age_urgency_start_day = max(1, int(age_urgency_start_day))
        self.age_urgency_min_return = age_urgency_min_return
        self.peak_detection_enabled = peak_detection_enabled
        self.peak_sell_portion_pct = peak_sell_portion_pct
        self.peak_min_gain_pct = peak_min_gain_pct
        self.peak_min_holding_days = max(0, int(peak_min_holding_days))
        self.peak_pred_return_threshold = peak_pred_return_threshold
        self.peak_score_percentile_drop = peak_score_percentile_drop
        self.peak_rsi_overbought = peak_rsi_overbought
        self.peak_above_ma_ratio = peak_above_ma_ratio
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

    def _detect_peak(
        self,
        p: Position,
        *,
        current_price: float,
        pred_return: float | None,
        score: float | None,
        score_percentile: float | None,
        rsi_14: float | None,
        ma20_ratio: float | None,
        days_held: int,
    ) -> tuple[bool, str]:
        """Detect if position has peaked using combination of signals."""
        if not self.peak_detection_enabled:
            return False, ""
        
        # Must meet minimum requirements
        if days_held < self.peak_min_holding_days:
            return False, ""
        
        if self.peak_min_gain_pct is not None and p.entry_price > 0:
            gain = (current_price - p.entry_price) / p.entry_price
            if gain < self.peak_min_gain_pct:
                return False, ""
        
        # Check multiple peak signals (combination approach)
        signals = []
        
        # Signal 1: Negative ML prediction (reversal expected)
        if (pred_return is not None and 
            self.peak_pred_return_threshold is not None and
            not pd.isna(pred_return) and
            pred_return < self.peak_pred_return_threshold):
            signals.append("NEG_PRED")
        
        # Signal 2: Score dropped significantly vs universe
        if (score_percentile is not None and 
            self.peak_score_percentile_drop is not None and
            not pd.isna(score_percentile) and
            score_percentile < self.peak_score_percentile_drop):
            signals.append("SCORE_DROP")
        
        # Signal 3: Technical overbought (RSI)
        if (rsi_14 is not None and 
            self.peak_rsi_overbought is not None and
            not pd.isna(rsi_14) and
            rsi_14 > self.peak_rsi_overbought):
            signals.append("RSI_OB")
        
        # Signal 4: Price extended above moving average
        if (ma20_ratio is not None and 
            self.peak_above_ma_ratio is not None and
            not pd.isna(ma20_ratio) and
            ma20_ratio > self.peak_above_ma_ratio):
            signals.append("MA_EXT")
        
        # Require at least 2 signals for peak confirmation
        if len(signals) >= 2:
            return True, "+".join(signals)
        
        return False, ""

    def _sell_partial_position(
        self,
        state: PortfolioState,
        p: Position,
        *,
        price_cad: float,
        reason: str,
        days_held: int,
        sell_portion: float,
    ) -> tuple[TradeAction, Position]:
        """Sell a portion of position, keeping the rest open."""
        shares_to_sell = max(1, int(p.shares * sell_portion))
        shares_remaining = p.shares - shares_to_sell
        
        # Don't do partial sell if it would leave us with 0 shares
        if shares_remaining <= 0:
            shares_to_sell = p.shares
            shares_remaining = 0
        
        # Execute partial sell
        proceeds = float(price_cad) * float(shares_to_sell)
        state.cash_cad = float(state.cash_cad) + float(proceeds)
        
        # Update position to reduced shares
        p.shares = shares_remaining
        
        action = TradeAction(
            ticker=p.ticker,
            action="SELL_PARTIAL",
            reason=f"PEAK_{reason}",
            shares=shares_to_sell,
            price_cad=float(price_cad),
            days_held=days_held,
        )
        
        return action, p

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

    def compute_dynamic_holding_days(
        self,
        market_vol_regime: float | None,
    ) -> tuple[int, int]:
        """Compute adjusted holding days based on market volatility.
        
        In high volatility (regime > 1), reduce holding period.
        In low volatility (regime < 1), extend holding period.
        
        Returns:
            Tuple of (adjusted_max_holding_days, adjusted_hard_limit)
        """
        if not self.dynamic_holding_enabled or market_vol_regime is None:
            return self.max_holding_days, self.max_holding_days_hard
        
        # Clamp regime to reasonable bounds [0.5, 2.0]
        regime = max(0.5, min(2.0, float(market_vol_regime)))
        
        # Scale factor: high vol (regime=2) -> 0.5x days, low vol (regime=0.5) -> 1.5x days
        # Formula: scale = 1 + (1 - regime) * vol_scale
        # regime=1.0 -> scale=1.0 (normal)
        # regime=2.0 -> scale=0.5 (shorter holds in high vol)
        # regime=0.5 -> scale=1.25 (longer holds in low vol)
        scale = 1.0 + (1.0 - regime) * self.dynamic_holding_vol_scale
        scale = max(0.5, min(1.5, scale))  # Clamp to [0.5, 1.5]
        
        adjusted_max = max(2, int(round(self.max_holding_days * scale)))
        adjusted_hard = max(adjusted_max + 2, int(round(self.max_holding_days_hard * scale)))
        
        return adjusted_max, adjusted_hard

    def compute_vol_adjusted_stop(
        self,
        ticker: str,
        features: pd.DataFrame | None,
    ) -> float | None:
        """Compute volatility-adjusted stop-loss for a ticker.
        
        High volatility stocks get wider stops to avoid noise-triggered exits.
        Low volatility stocks get tighter stops to protect capital.
        
        Returns:
            Stop-loss percentage (e.g., 0.08 for 8%) or None if disabled
        """
        if not self.vol_adjusted_stop_enabled:
            return self.stop_loss_pct
        
        if features is None or "vol_60d_ann" not in features.columns:
            return self.vol_adjusted_stop_base
        
        # Get volatility for this ticker
        if ticker in features.index:
            vol = features.loc[ticker, "vol_60d_ann"]
        elif "ticker" in features.columns:
            row = features[features["ticker"] == ticker]
            vol = row["vol_60d_ann"].iloc[0] if len(row) > 0 else None
        else:
            vol = None
        
        if vol is None or pd.isna(vol):
            return self.vol_adjusted_stop_base
        
        # Typical stock volatility ~30% annualized, scale stop-loss proportionally
        # vol_ratio = stock_vol / typical_vol
        typical_vol = 0.30
        vol_ratio = float(vol) / typical_vol
        
        # Scale stop-loss: higher vol -> wider stop
        # At vol_ratio=1.0, use base stop
        # At vol_ratio=2.0, use wider stop (approaching max)
        # At vol_ratio=0.5, use tighter stop (approaching min)
        adjusted_stop = self.vol_adjusted_stop_base * vol_ratio
        adjusted_stop = max(self.vol_adjusted_stop_min, min(self.vol_adjusted_stop_max, adjusted_stop))
        
        return adjusted_stop

    def apply_exits(
        self,
        state: PortfolioState,
        prices_cad: pd.Series,
        *,
        pred_return: pd.Series | None = None,
        score: pd.Series | None = None,
        features: pd.DataFrame | None = None,
        market_vol_regime: float | None = None,
    ) -> list[TradeAction]:
        actions: list[TradeAction] = []
        now = _utcnow()
        open_positions: list[Position] = [p for p in state.positions if p.status == "OPEN"]
        keep: list[Position] = []
        
        # Compute dynamic holding period based on market volatility
        max_hold, max_hold_hard = self.compute_dynamic_holding_days(market_vol_regime)
        if self.dynamic_holding_enabled and market_vol_regime is not None:
            if max_hold != self.max_holding_days:
                self.logger.info(
                    "Dynamic holding: vol_regime=%.2f, max_days=%d->%d, hard=%d->%d",
                    market_vol_regime, self.max_holding_days, max_hold,
                    self.max_holding_days_hard, max_hold_hard
                )

        for p in open_positions:
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.isna(px) or px <= 0:
                keep.append(p)
                continue

            days = p.days_held(now)
            # Time exits (skipped if peak_based_exit is enabled):
            # - Default behavior: exit at max_holding_days (dynamic if enabled).
            # - Extension behavior: if model signal is strong, allow holding up to max_holding_days_hard.
            # When peak_based_exit is True, we rely on trailing stops instead of fixed time.
            if not self.peak_based_exit and days >= max_hold:
                if days >= max_hold_hard:
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

            # Volatility-adjusted or fixed stop-loss
            effective_stop = self.compute_vol_adjusted_stop(p.ticker, features)
            if effective_stop is not None and ret <= -abs(float(effective_stop)):
                if self.vol_adjusted_stop_enabled:
                    self.logger.info(
                        "%s vol-adjusted stop: ret=%.1f%%, stop=%.1f%%",
                        p.ticker, ret * 100, effective_stop * 100
                    )
                actions.append(self._sell_position(state, p, price_cad=px, reason="STOP_LOSS", days_held=days))
                continue

            if self.take_profit_pct is not None and ret >= abs(float(self.take_profit_pct)):
                actions.append(self._sell_position(state, p, price_cad=px, reason="TAKE_PROFIT", days_held=days))
                continue

            # Age urgency: exit underperformers earlier as they age
            # As positions age, we require progressively higher returns to justify holding
            if self.age_urgency_enabled and days >= self.age_urgency_start_day:
                # Calculate required return that increases with age
                # At start_day: require min_return
                # At max_hold: require 2x min_return
                age_ratio = min(1.0, (days - self.age_urgency_start_day) / max(1, max_hold - self.age_urgency_start_day))
                required_return = self.age_urgency_min_return * (1.0 + age_ratio)
                
                if ret < required_return:
                    # Position is underperforming for its age
                    self.logger.info(
                        "%s age urgency: day %d, ret=%.1f%% < required=%.1f%%",
                        p.ticker, days, ret * 100, required_return * 100
                    )
                    actions.append(self._sell_position(state, p, price_cad=px, reason="AGE_URGENCY", days_held=days))
                    continue

            # Time-weighted return optimization: exit to maximize capital efficiency
            if self.twr_optimization and p.entry_price > 0:
                gain = px / float(p.entry_price) - 1.0
                
                # Quick profit exit: take gains if we hit target quickly
                if days <= self.quick_profit_days and gain >= self.quick_profit_pct:
                    annualized = ((1 + gain) ** (252 / max(1, days))) - 1
                    self.logger.info(
                        "%s quick profit: %.1f%% gain in %d days (annualized: %.0f%%)",
                        p.ticker, gain * 100, days, annualized * 100
                    )
                    actions.append(self._sell_position(state, p, price_cad=px, reason="QUICK_PROFIT", days_held=days))
                    continue
                
                # Daily return check: exit if return per day is too low
                if days >= 2 and gain > 0:
                    daily_return = gain / days
                    if daily_return < self.min_daily_return:
                        self.logger.info(
                            "%s low daily return: %.2f%% total / %d days = %.2f%%/day (min: %.2f%%)",
                            p.ticker, gain * 100, days, daily_return * 100, self.min_daily_return * 100
                        )
                        actions.append(self._sell_position(state, p, price_cad=px, reason="LOW_DAILY_RETURN", days_held=days))
                        continue
                
                # Momentum decay: exit if we're past peak and velocity is slowing
                if self.momentum_decay_exit and p.highest_price and p.highest_price > p.entry_price:
                    current_gain = gain
                    peak_gain = (p.highest_price / float(p.entry_price)) - 1.0
                    
                    # If we've given back more than 40% of our peak gain, exit
                    if peak_gain > 0.02 and current_gain < peak_gain * 0.6:
                        self.logger.info(
                            "%s momentum decay: current %.1f%% vs peak %.1f%% (gave back %.0f%%)",
                            p.ticker, current_gain * 100, peak_gain * 100, (1 - current_gain / peak_gain) * 100
                        )
                        actions.append(self._sell_position(state, p, price_cad=px, reason="MOMENTUM_DECAY", days_held=days))
                        continue

            # Trailing stop: update highest price and check trailing stop trigger
            if self.trailing_stop_enabled and p.entry_price > 0:
                # Update highest price for this position
                p.update_highest_price(px)
                
                # Check if trailing stop is activated (position has gained enough)
                gain_from_entry = px / float(p.entry_price) - 1.0
                if gain_from_entry >= self.trailing_stop_activation_pct and p.highest_price:
                    # Trailing stop is active - check if we should exit
                    trailing_stop_level = p.highest_price * (1.0 - self.trailing_stop_distance_pct)
                    if px <= trailing_stop_level:
                        # Price has dropped below trailing stop
                        pct_from_peak = (px / p.highest_price - 1.0) * 100
                        self.logger.info(
                            "%s trailing stop: price $%.2f dropped %.1f%% from peak $%.2f",
                            p.ticker, px, pct_from_peak, p.highest_price
                        )
                        actions.append(self._sell_position(state, p, price_cad=px, reason="TRAILING_STOP", days_held=days))
                        continue

            # Signal decay exit: exit if prediction has turned significantly negative
            if self.signal_decay_exit_enabled and pred_return is not None:
                pr_val = float(pred_return.get(p.ticker, float("nan")))
                if not pd.isna(pr_val) and pr_val <= self.signal_decay_threshold:
                    # Model now predicts negative return - exit early
                    self.logger.info(
                        "%s signal decay: pred_return=%.2f%% (threshold=%.2f%%)",
                        p.ticker, pr_val * 100, self.signal_decay_threshold * 100
                    )
                    actions.append(self._sell_position(state, p, price_cad=px, reason="SIGNAL_DECAY", days_held=days))
                    continue

            # Peak detection with partial exit (only for winning positions)
            if self.peak_detection_enabled and p.entry_price > 0:
                gain = px / float(p.entry_price) - 1.0
                if gain > 0:  # Only check peaks for winning positions
                    # Get prediction for this ticker
                    pr = None
                    if pred_return is not None:
                        pr_val = float(pred_return.get(p.ticker, float("nan")))
                        if not pd.isna(pr_val):
                            pr = pr_val
                    
                    # Get score and compute percentile
                    sc = None
                    sc_pct = None
                    if score is not None:
                        sc_val = float(score.get(p.ticker, float("nan")))
                        if not pd.isna(sc_val):
                            sc = sc_val
                            # Compute percentile: rank score against all available scores
                            all_scores = score.dropna()
                            if len(all_scores) > 0:
                                sc_pct = (all_scores < sc_val).sum() / len(all_scores)
                    
                    # Get technical indicators from features
                    rsi = None
                    ma20_r = None
                    if features is not None and p.ticker in features.index:
                        if "rsi_14" in features.columns:
                            rsi_val = float(features.loc[p.ticker, "rsi_14"])
                            if not pd.isna(rsi_val):
                                rsi = rsi_val
                        if "ma20_ratio" in features.columns:
                            ma_val = float(features.loc[p.ticker, "ma20_ratio"])
                            if not pd.isna(ma_val):
                                ma20_r = ma_val
                    
                    is_peak, peak_reason = self._detect_peak(
                        p,
                        current_price=px,
                        pred_return=pr,
                        score=sc,
                        score_percentile=sc_pct,
                        rsi_14=rsi,
                        ma20_ratio=ma20_r,
                        days_held=days,
                    )
                    
                    if is_peak and p.shares >= 2:  # Only do partial sell if we have at least 2 shares
                        action, remaining_pos = self._sell_partial_position(
                            state, p, price_cad=px, reason=peak_reason,
                            days_held=days, sell_portion=self.peak_sell_portion_pct
                        )
                        actions.append(action)
                        if remaining_pos.shares > 0:
                            keep.append(remaining_pos)
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
                # Get pred_return from weights if available
                pred_ret = None
                try:
                    if "pred_return" in weights.columns:
                        pred_ret = float(weights.loc[t, "pred_return"])
                except Exception:
                    pass
                # Compute expected sell date/strategy
                expected_sell = None
                if self.peak_based_exit:
                    expected_sell = "Peak Detection"
                elif self.max_holding_days:
                    sell_dt = now + timedelta(days=self.max_holding_days)
                    expected_sell = sell_dt.strftime("%Y-%m-%d")
                actions.append(
                    TradeAction(
                        ticker=t, action="BUY", reason="TOP_RANKED", shares=int(shares), 
                        price_cad=px, days_held=0, pred_return=pred_ret, expected_sell_date=expected_sell
                    )
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
            # Get pred_return from weights if available
            pred_ret = None
            try:
                if "pred_return" in weights.columns:
                    pred_ret = float(weights.loc[t, "pred_return"])
            except Exception:
                pass
            # Compute expected sell date/strategy from entry
            expected_sell = None
            if self.peak_based_exit:
                expected_sell = "Peak Detection"
            elif p and self.max_holding_days:
                sell_dt = p.entry_date + timedelta(days=self.max_holding_days)
                expected_sell = sell_dt.strftime("%Y-%m-%d")
            actions.append(
                TradeAction(
                    ticker=t, action="HOLD", reason="IN_TARGET", shares=p.shares if p else 0, 
                    price_cad=px, days_held=days, pred_return=pred_ret, expected_sell_date=expected_sell
                )
            )

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


