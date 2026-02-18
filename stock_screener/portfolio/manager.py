from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pandas as pd

from stock_screener.portfolio.state import PortfolioState, Position, save_portfolio_state
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Trading-day calendar: weekdays minus market holidays (NYSE / TSX).
# Self-contained — no external dependency required.
# ---------------------------------------------------------------------------
from datetime import date as _date

_HOLIDAY_CACHE: dict[tuple[str, int], set[_date]] = {}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _date:
    """Return the Nth occurrence of a weekday in a month (1-indexed)."""
    first = _date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _date:
    """Return the last occurrence of a weekday in a month."""
    if month == 12:
        last_day = _date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = _date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed(d: _date) -> _date:
    """Shift a fixed holiday to the observed weekday (Fri if Sat, Mon if Sun)."""
    if d.weekday() == 5:  # Saturday → Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday → Monday
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> _date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return _date(year, month, day + 1)


def _nyse_holidays(year: int) -> set[_date]:
    """Compute NYSE market holidays for a given year."""
    key = ("US", year)
    if key in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[key]

    holidays = set()

    # New Year's Day (Jan 1)
    holidays.add(_observed(_date(year, 1, 1)))

    # Martin Luther King Jr. Day (3rd Monday in January)
    holidays.add(_nth_weekday(year, 1, 0, 3))  # 0 = Monday

    # Presidents' Day (3rd Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday (2 days before Easter Sunday)
    holidays.add(_easter(year) - timedelta(days=2))

    # Memorial Day (last Monday in May)
    holidays.add(_last_weekday(year, 5, 0))

    # Juneteenth (June 19) — observed since 2022
    if year >= 2022:
        holidays.add(_observed(_date(year, 6, 19)))

    # Independence Day (July 4)
    holidays.add(_observed(_date(year, 7, 4)))

    # Labor Day (1st Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving (4th Thursday in November)
    holidays.add(_nth_weekday(year, 11, 3, 4))  # 3 = Thursday

    # Christmas (December 25)
    holidays.add(_observed(_date(year, 12, 25)))

    _HOLIDAY_CACHE[key] = holidays
    return holidays


def _canada_observed(d: _date) -> _date:
    """Canada-style observed holiday (Sat/Sun -> Monday)."""
    if d.weekday() == 5:  # Saturday -> Monday
        return d + timedelta(days=2)
    if d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


def _tsx_holidays(year: int) -> set[_date]:
    """Compute core TSX holidays for a given year."""
    key = ("CA", year)
    if key in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[key]

    holidays: set[_date] = set()

    # New Year's Day
    holidays.add(_canada_observed(_date(year, 1, 1)))

    # Family Day (3rd Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday
    holidays.add(_easter(year) - timedelta(days=2))

    # Victoria Day (Monday preceding May 25)
    victoria = _date(year, 5, 24)
    while victoria.weekday() != 0:
        victoria -= timedelta(days=1)
    holidays.add(victoria)

    # Canada Day
    holidays.add(_canada_observed(_date(year, 7, 1)))

    # Civic Holiday (1st Monday in August)
    holidays.add(_nth_weekday(year, 8, 0, 1))

    # Labour Day (1st Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving (2nd Monday in October)
    holidays.add(_nth_weekday(year, 10, 0, 2))

    # Christmas + Boxing Day (ensure distinct observed dates)
    christmas_obs = _canada_observed(_date(year, 12, 25))
    holidays.add(christmas_obs)
    boxing_obs = _canada_observed(_date(year, 12, 26))
    while boxing_obs in holidays or boxing_obs.weekday() >= 5:
        boxing_obs += timedelta(days=1)
    holidays.add(boxing_obs)

    _HOLIDAY_CACHE[key] = holidays
    return holidays


def _market_for_ticker(ticker: str | None) -> str:
    t = str(ticker or "").strip().upper()
    if t.endswith(".TO") or t.endswith(".V"):
        return "CA"
    return "US"


def _is_trading_day(d, market: str = "US") -> bool:
    """Check if a date is a trading day for the requested market."""
    if hasattr(d, "date"):
        d = d.date()
    if d.weekday() >= 5:
        return False
    m = str(market or "US").upper()
    holidays = _tsx_holidays(d.year) if m == "CA" else _nyse_holidays(d.year)
    return d not in holidays


def _trading_days_between(start: datetime, end: datetime, market: str = "US") -> int:
    """Count trading days between two dates, excluding start, including end."""
    if end <= start:
        return 0
    count = 0
    d = start.date() + timedelta(days=1)
    end_d = end.date()
    while d <= end_d:
        if _is_trading_day(d, market=market):
            count += 1
        d += timedelta(days=1)
    return count


def _add_trading_days(start: datetime, trading_days: int, market: str = "US") -> datetime:
    """Return the datetime that is N trading days after start."""
    if trading_days <= 0:
        return start
    d = start
    added = 0
    while added < trading_days:
        d += timedelta(days=1)
        if _is_trading_day(d, market=market):
            added += 1
    return d


@dataclass(frozen=True)
class TradeAction:
    ticker: str
    action: str  # BUY | SELL | SELL_PARTIAL | HOLD
    reason: str
    shares: float  # Fractional shares supported for expensive stocks
    price_cad: float
    days_held: int | None = None
    pred_return: float | None = None  # Predicted return for this position
    expected_sell_date: str | None = None  # Expected sell date from peak model (if available)
    # For SELL actions: realized gain/loss info
    entry_price: float | None = None  # Entry price (for sells)
    realized_gain_pct: float | None = None  # Realized % gain/loss (for sells)
    # For BUY actions: explicit linkage to the rotated-out ticker (if any)
    replaces_ticker: str | None = None


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
        min_trade_notional_cad: float = 10.0,
        min_rebalance_weight_delta: float = 0.01,
        rotate_on_missing_data: bool = False,
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
        self.min_trade_notional_cad = max(1.0, float(min_trade_notional_cad))
        self.min_rebalance_weight_delta = max(0.0, float(min_rebalance_weight_delta))
        self.rotate_on_missing_data = bool(rotate_on_missing_data)
        self.logger = logger

    def _positions_by_ticker(self, state: PortfolioState) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in state.positions:
            if p.status == "OPEN" and p.ticker:
                out[p.ticker] = p
        return out

    def _normalize_open_positions(self, state: PortfolioState) -> None:
        """Consolidate duplicate OPEN lots per ticker into one lot.

        The runtime expects one open position per ticker. Legacy/manual states can
        violate this and cause silent lot drops in planning paths.
        """
        grouped: dict[str, list[Position]] = {}
        closed: list[Position] = []
        for p in state.positions:
            if p.status == "OPEN" and p.ticker and p.shares > 0:
                ticker = str(p.ticker).upper()
                p.ticker = ticker
                grouped.setdefault(ticker, []).append(p)
            else:
                closed.append(p)

        if not grouped:
            state.positions = closed
            return

        normalized_open: list[Position] = []
        duplicate_tickers = 0
        duplicate_lots = 0
        for ticker, lots in grouped.items():
            if len(lots) == 1:
                normalized_open.append(lots[0])
                continue

            duplicate_tickers += 1
            duplicate_lots += len(lots) - 1

            total_shares = float(sum(float(l.shares) for l in lots if l.shares > 0))
            if total_shares <= 0:
                continue

            px_num = 0.0
            px_den = 0.0
            pred_ret_num = 0.0
            pred_ret_den = 0.0
            peak_days_num = 0.0
            peak_days_den = 0.0
            for lot in lots:
                sh = float(lot.shares)
                if sh <= 0:
                    continue
                ep = float(lot.entry_price) if lot.entry_price is not None else 0.0
                if ep > 0:
                    px_num += ep * sh
                    px_den += sh
                if lot.entry_pred_return is not None and not pd.isna(lot.entry_pred_return):
                    pred_ret_num += float(lot.entry_pred_return) * sh
                    pred_ret_den += sh
                if lot.entry_pred_peak_days is not None and not pd.isna(lot.entry_pred_peak_days):
                    peak_days_num += float(lot.entry_pred_peak_days) * sh
                    peak_days_den += sh

            entry_price = (px_num / px_den) if px_den > 0 else float(lots[0].entry_price)
            entry_date = min(l.entry_date for l in lots)
            highest_vals = [float(l.highest_price) for l in lots if l.highest_price is not None and float(l.highest_price) > 0]
            highest_price = max(highest_vals) if highest_vals else None
            stop_loss = next((l.stop_loss_pct for l in lots if l.stop_loss_pct is not None), lots[0].stop_loss_pct)
            take_profit = next((l.take_profit_pct for l in lots if l.take_profit_pct is not None), lots[0].take_profit_pct)
            pred_return = (pred_ret_num / pred_ret_den) if pred_ret_den > 0 else lots[0].entry_pred_return
            pred_peak_days = (peak_days_num / peak_days_den) if peak_days_den > 0 else lots[0].entry_pred_peak_days

            normalized_open.append(
                Position(
                    ticker=ticker,
                    entry_price=float(entry_price),
                    entry_date=entry_date,
                    shares=float(total_shares),
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                    highest_price=highest_price,
                    entry_pred_peak_days=pred_peak_days,
                    entry_pred_return=pred_return,
                )
            )

        if duplicate_lots > 0:
            self.logger.warning(
                "Consolidated %d duplicate OPEN lot(s) across %d ticker(s).",
                duplicate_lots,
                duplicate_tickers,
            )

        state.positions = normalized_open + closed

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
        # Calculate realized gain/loss
        entry_px = float(p.entry_price) if p.entry_price else None
        realized_gain = ((price_cad / entry_px) - 1.0) if entry_px and entry_px > 0 else None
        return TradeAction(
            ticker=p.ticker,
            action="SELL",
            reason=reason,
            shares=p.shares,
            price_cad=float(price_cad),
            days_held=days_held,
            entry_price=entry_px,
            realized_gain_pct=realized_gain,
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
    ) -> tuple[TradeAction, Position | None]:
        """Sell a portion of position, keeping the rest open."""
        shares_to_sell = max(0.01, round(p.shares * sell_portion, 4))  # Fractional shares
        shares_remaining = round(p.shares - shares_to_sell, 4)
        
        # Don't do partial sell if it would leave us with 0 shares
        if shares_remaining <= 0:
            full_action = self._sell_position(
                state,
                p,
                price_cad=price_cad,
                reason=f"PEAK_{reason}",
                days_held=days_held,
            )
            return full_action, None
        
        # Execute partial sell
        proceeds = float(price_cad) * float(shares_to_sell)
        state.cash_cad = float(state.cash_cad) + float(proceeds)
        
        # Update position to reduced shares
        p.shares = shares_remaining
        entry_px = float(p.entry_price) if p.entry_price else None
        realized_gain = ((price_cad / entry_px) - 1.0) if entry_px and entry_px > 0 else None
        
        action = TradeAction(
            ticker=p.ticker,
            action="SELL_PARTIAL",
            reason=f"PEAK_{reason}",
            shares=shares_to_sell,
            price_cad=float(price_cad),
            days_held=days_held,
            entry_price=entry_px,
            realized_gain_pct=realized_gain,
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
        self._normalize_open_positions(state)
        actions: list[TradeAction] = []
        now = _utcnow()
        open_positions: list[Position] = [p for p in state.positions if p.status == "OPEN"]
        keep: list[Position] = []
        adjusted_max_hold, _ = self.compute_dynamic_holding_days(market_vol_regime)
        age_urgency_start = self.age_urgency_start_day
        if self.dynamic_holding_enabled and self.max_holding_days > 0:
            ratio = float(adjusted_max_hold) / float(self.max_holding_days)
            age_urgency_start = max(1, int(round(float(self.age_urgency_start_day) * ratio)))

        for p in open_positions:
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.isna(px) or px <= 0:
                keep.append(p)
                continue

            market = _market_for_ticker(p.ticker)
            days = _trading_days_between(p.entry_date, now, market=market)  # trading days (matches model)

            # Adaptive peak-target exit: combine entry-time and today's predictions.
            # pred_peak_days is in trading days (model trained on trading-day rows).
            if self.peak_based_exit:
                entry_target = None
                if p.entry_pred_peak_days is not None and not pd.isna(p.entry_pred_peak_days):
                    entry_target = max(1, int(p.entry_pred_peak_days))

                # Today's prediction: how many trading days from NOW until peak
                today_target = None
                if features is not None and p.ticker in features.index and "pred_peak_days" in features.columns:
                    v = features.loc[p.ticker, "pred_peak_days"]
                    if not pd.isna(v) and float(v) > 0:
                        # Convert "trading days from today" to "trading days from entry"
                        today_target = days + max(1, int(float(v)))

                # Choose the tighter (earlier) target
                effective_target = None
                exit_source = None
                if entry_target is not None and today_target is not None:
                    if today_target <= entry_target:
                        effective_target = today_target
                        exit_source = "updated"
                    else:
                        effective_target = entry_target
                        exit_source = "entry"
                elif entry_target is not None:
                    effective_target = entry_target
                    exit_source = "entry"
                elif today_target is not None:
                    effective_target = today_target
                    exit_source = "updated"

                if effective_target is not None and days >= effective_target:
                    actions.append(self._sell_position(
                        state, p, price_cad=px,
                        reason=f"PEAK_TARGET(day{effective_target},{exit_source})",
                        days_held=days,
                    ))
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
                            p.ticker,
                            gain * 100,
                            days,
                            daily_return * 100,
                            self.min_daily_return * 100,
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

            # Age urgency: once a position is old enough, require a minimum
            # absolute gain floor to keep capital efficient.
            if self.age_urgency_enabled and days >= age_urgency_start and ret < self.age_urgency_min_return:
                self.logger.info(
                    "%s age urgency: %.2f%% gain after %d trading days (< %.2f%% minimum)",
                    p.ticker,
                    ret * 100,
                    days,
                    self.age_urgency_min_return * 100,
                )
                actions.append(self._sell_position(state, p, price_cad=px, reason="AGE_URGENCY", days_held=days))
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
                    
                    if is_peak:
                        potential_shares = max(0.01, round(p.shares * self.peak_sell_portion_pct, 4))
                        if float(potential_shares) * float(px) < self.min_trade_notional_cad:
                            keep.append(p)
                            continue
                        action, remaining_pos = self._sell_partial_position(
                            state, p, price_cad=px, reason=peak_reason,
                            days_held=days, sell_portion=self.peak_sell_portion_pct
                        )
                        actions.append(action)
                        if remaining_pos is not None and remaining_pos.shares > 0:
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
        scored: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
        blocked_buys: set[str] | None = None,
    ) -> TradePlan:
        self._normalize_open_positions(state)
        # weights is expected to represent the *target* holdings universe, but we may rotate.
        now = _utcnow()
        actions: list[TradeAction] = []

        open_by_ticker = self._positions_by_ticker(state)
        open_tickers = set(open_by_ticker.keys())

        target_tickers = list(weights.index.astype(str))
        target_set = set(target_tickers)
        target_set_upper = {t.upper() for t in target_set}
        blocked_buy_tickers = {
            str(t).strip().upper()
            for t in (blocked_buys or set())
            if str(t).strip()
        }

        # SELL positions not in target - but only if they meet rotation criteria.
        # Avoid excessive turnover by requiring deteriorating fundamentals to rotate out.
        # Positions that survive rotation will get a HOLD action in the loop below
        # so they are always visible in the report.
        open_mkt_value_for_rotation = 0.0
        for p in state.positions:
            if p.status != "OPEN" or not p.ticker or p.shares <= 0:
                continue
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.isna(px) or px <= 0:
                continue
            open_mkt_value_for_rotation += float(px) * float(p.shares)
        equity_for_rotation = float(state.cash_cad) + float(open_mkt_value_for_rotation)

        for t, p in list(open_by_ticker.items()):
            if t.upper() not in target_set_upper:
                px = float(prices_cad.get(t, float("nan")))
                if pd.isna(px) or px <= 0:
                    # No price data – keep position; HOLD action emitted later.
                    continue
                # Use trading days consistently for tenure-aware rotation checks.
                market = _market_for_ticker(t)
                days = _trading_days_between(p.entry_date, now, market=market)
                
                # Get current predictions for this ticker (if available in screened)
                ticker_data = screened[screened.index == t] if t in screened.index else None
                pred_ret = None
                score_pct = None
                if ticker_data is not None and len(ticker_data) > 0:
                    pred_ret = ticker_data.get("pred_return", pd.Series([None])).iloc[0]
                    if "score" in ticker_data.columns:
                        ticker_score = ticker_data["score"].iloc[0]
                        score_pct = (screened["score"] < ticker_score).mean() if "score" in screened.columns else None
                
                # Decide whether to rotate: only sell if one of these conditions is met
                should_rotate = False
                rotation_reason = "ROTATION"
                
                # 1. Predicted return is now negative (bearish signal)
                if pred_ret is not None and pred_ret < 0:
                    should_rotate = True
                    rotation_reason = "ROTATION:NEG_PRED"
                # 2. Stock dropped to bottom 30% of screened universe
                elif score_pct is not None and score_pct < 0.30:
                    should_rotate = True
                    rotation_reason = "ROTATION:LOW_RANK"
                # 3. Optional policy: rotate stale no-data holdings (disabled by default).
                elif pred_ret is None and self.rotate_on_missing_data and days >= 2:
                    should_rotate = True
                    rotation_reason = "ROTATION:NO_DATA"

                # Churn controls: avoid tiny trades and tiny reallocations.
                trade_notional = float(px) * float(p.shares)
                cur_weight = (
                    trade_notional / equity_for_rotation
                    if equity_for_rotation > 0
                    else 0.0
                )
                if trade_notional < self.min_trade_notional_cad:
                    should_rotate = False
                elif (
                    should_rotate
                    and cur_weight < self.min_rebalance_weight_delta
                    and rotation_reason in {"ROTATION:LOW_RANK", "ROTATION:NO_DATA"}
                ):
                    should_rotate = False
                
                if should_rotate:
                    actions.append(self._sell_position(state, p, price_cad=px, reason=rotation_reason, days_held=days))
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
        rotation_sell_queue: list[str] = []
        for a in actions:
            if a.action == "SELL" and str(a.reason).startswith("ROTATION"):
                rotation_sell_queue.append(str(a.ticker).upper())
        if slots > 0:
            for t in target_tickers:
                t_norm = t.upper()
                if slots <= 0:
                    break
                if t_norm in open_tickers:
                    continue
                if t_norm in blocked_buy_tickers:
                    continue
                px = float(prices_cad.get(t_norm, prices_cad.get(t, float("nan"))))
                if pd.isna(px) or px <= 0:
                    continue

                w = 0.0
                try:
                    w = float(weights.loc[t, "weight"]) if "weight" in weights.columns else 0.0
                except Exception:
                    w = 0.0

                target_value = float(equity_cad) * float(w) if w > 0 else float(state.cash_cad) / max(slots, 1)
                # FRACTIONAL SHARES: Allow buying partial shares for expensive stocks
                # This ensures we can invest in high-priced stocks within budget
                target_shares = max(0.0, target_value) / float(px)
                affordable_shares = float(state.cash_cad) / float(px)
                if affordable_shares < 0.01:
                    continue
                # Minimum 0.01 shares (most brokers support fractional to 0.001)
                shares = min(affordable_shares, max(0.01, target_shares))
                if shares < 0.01 or (shares * px) < 1.0:  # Skip if less than $1 investment
                    continue
                # Round to 4 decimal places for practical fractional share trading
                shares = round(shares, 4)

                # Turnover controls: skip small target allocations/trades.
                if w > 0 and w < self.min_rebalance_weight_delta:
                    continue
                
                # Get pred_return and pred_peak_days from weights BEFORE creating position
                pred_ret = None
                pred_peak_days = None
                try:
                    if "pred_return" in weights.columns:
                        v = float(weights.loc[t, "pred_return"])
                        if not pd.isna(v):
                            pred_ret = v
                    if "pred_peak_days" in weights.columns:
                        v = float(weights.loc[t, "pred_peak_days"])
                        if not pd.isna(v):
                            pred_peak_days = v
                except Exception:
                    pass
                
                cost = float(px) * float(shares)
                seed_notional_entry = False
                if cost < self.min_trade_notional_cad:
                    # Cold start exception: allow exactly one small seed entry
                    # (>= 1 CAD) so the portfolio can recover from zero-exposure lockout.
                    if len(open_tickers) == 0 and cost >= 1.0:
                        seed_notional_entry = True
                        self.logger.info(
                            "Seed entry for %s below min_trade_notional (cost=%.2f < %.2f) to avoid empty portfolio.",
                            t,
                            cost,
                            self.min_trade_notional_cad,
                        )
                    else:
                        continue
                state.cash_cad = float(state.cash_cad) - float(cost)
                pos = Position(
                    ticker=t_norm,
                    entry_price=px,
                    entry_date=now,
                    shares=float(shares),  # Fractional shares
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    entry_pred_peak_days=pred_peak_days,
                    entry_pred_return=pred_ret,
                )
                state.positions.append(pos)
                
                # Compute expected sell date from predicted peak day.
                # pred_peak_days is in trading days (model trained on trading-day data).
                if pred_peak_days is not None and pred_peak_days > 0:
                    peak_day = max(1, int(pred_peak_days))
                    sell_dt = _add_trading_days(now, peak_day, market=_market_for_ticker(t_norm))
                    expected_sell = f"{sell_dt.strftime('%Y-%m-%d')} (peak day {peak_day})"
                else:
                    expected_sell = "N/A (peak signal pending)"
                
                actions.append(
                    TradeAction(
                        ticker=t_norm,
                        action="BUY",
                        reason="TOP_RANKED:SEED_NOTIONAL" if seed_notional_entry else "TOP_RANKED",
                        shares=float(shares),
                        price_cad=px, days_held=0, pred_return=pred_ret, expected_sell_date=expected_sell,
                        replaces_ticker=rotation_sell_queue.pop(0) if rotation_sell_queue else None,
                    )
                )
                open_tickers.add(t_norm)
                slots -= 1

        # Track tickers that already have an action (BUY or SELL) so we don't double-report.
        actioned_tickers = {a.ticker for a in actions if a.action in ("BUY", "SELL", "SELL_PARTIAL")}
        
        # HOLD: emit an action for every open position so the report always
        # accounts for each held ticker.  Positions still in the target set
        # are labelled IN_TARGET; those kept despite leaving the target are
        # labelled HOLDING (pending rotation).
        for t in sorted(open_tickers):
            if t in actioned_tickers:
                continue  # Already have a BUY/SELL action for this ticker
            p = open_by_ticker.get(t)
            px = float(prices_cad.get(t, float("nan")))
            market = _market_for_ticker(t)
            days = _trading_days_between(p.entry_date, now, market=market) if p else None

            in_target = t.upper() in target_set_upper
            hold_reason = "IN_TARGET" if in_target else "HOLDING"

            # Get pred_return and pred_peak_days -- cascade through progressively
            # broader DataFrames: weights → screened → scored → features (unfiltered).
            # The features fallback ensures held positions that were filtered out by
            # quality gates (vol cap, price, liquidity) still show predictions.
            pred_ret = None
            pred_peak_days = None
            try:
                sources = [weights if in_target else None, screened, scored, features]
                for source in sources:
                    if source is None or source.empty or t not in source.index:
                        continue
                    if pred_ret is None and "pred_return" in source.columns:
                        val = source.loc[t, "pred_return"]
                        if not pd.isna(val):
                            pred_ret = float(val)
                    if pred_peak_days is None and "pred_peak_days" in source.columns:
                        val = source.loc[t, "pred_peak_days"]
                        if not pd.isna(val):
                            pred_peak_days = float(val)
                    if pred_ret is not None and pred_peak_days is not None:
                        break
            except Exception:
                pass
            
            # Adaptive sell date: mirrors the exit logic — use the earlier of
            # entry prediction and today's prediction (whichever triggers first).
            entry_peak = None
            today_peak = None
            if p and p.entry_pred_peak_days is not None and not pd.isna(p.entry_pred_peak_days) and p.entry_pred_peak_days > 0:
                entry_peak = max(1, int(p.entry_pred_peak_days))
            if pred_peak_days is not None and pred_peak_days > 0:
                today_peak = max(1, int(pred_peak_days))

            if entry_peak is not None or today_peak is not None:
                # Convert trading-day offsets to actual calendar sell dates.
                entry_sell_dt = _add_trading_days(p.entry_date, entry_peak, market=market) if entry_peak and p else None
                today_sell_dt = _add_trading_days(now, today_peak, market=market) if today_peak else None

                # Pick the earlier date (adaptive: react to new info)
                if entry_sell_dt and today_sell_dt:
                    if today_sell_dt <= entry_sell_dt:
                        sell_dt = today_sell_dt
                        source = "updated"
                    else:
                        sell_dt = entry_sell_dt
                        source = "entry"
                elif entry_sell_dt:
                    sell_dt = entry_sell_dt
                    source = "entry"
                else:
                    sell_dt = today_sell_dt
                    source = "updated"

                days_left = _trading_days_between(now, sell_dt, market=market)
                expected_sell = f"{sell_dt.strftime('%Y-%m-%d')} ({source}, {days_left}td left)"
            else:
                expected_sell = "N/A (peak signal pending)"
            
            # For sell price: prefer today's prediction (latest info),
            # fall back to entry prediction for positions filtered from scoring.
            if pred_ret is None and p and p.entry_pred_return is not None and not pd.isna(p.entry_pred_return):
                pred_ret = float(p.entry_pred_return)
            
            actions.append(
                TradeAction(
                    ticker=t, action="HOLD", reason=hold_reason, shares=p.shares if p else 0, 
                    price_cad=px, days_held=days, pred_return=pred_ret, expected_sell_date=expected_sell
                )
            )

        # Build holdings view: join weights with state-derived fields.
        holdings = weights.copy()
        holdings["days_held"] = pd.NA
        holdings["entry_price_cad"] = pd.NA
        for t, p in self._positions_by_ticker(state).items():
            if t in holdings.index:
                holdings.loc[t, "days_held"] = int(_trading_days_between(p.entry_date, now, market=_market_for_ticker(t)))
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
