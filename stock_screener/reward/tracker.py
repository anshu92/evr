"""Reward tracker -- logs per-ticker predictions, per-action outcomes, and counterfactuals."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stock_screener.portfolio.manager import _add_trading_days, _market_for_ticker, _trading_days_between


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _date_to_utc_datetime(x: str) -> datetime:
    ts = pd.Timestamp(x)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


# ---------------------------------------------------------------------------
# Per-action reward tracking
# ---------------------------------------------------------------------------

@dataclass
class ActionRewardEntry:
    """Tracks a single trade action and its subsequent outcome."""

    date: str  # Date the action was taken (YYYY-MM-DD)
    ticker: str
    action: str  # BUY | SELL | SELL_PARTIAL | HOLD
    reason: str  # Why the action was taken
    price_at_action: float  # Price when the action was executed
    shares: float = 0.0
    predicted_return: float = 0.0  # What the model predicted at action time
    confidence: float = 0.0

    # Post-action tracking (filled in on subsequent days)
    price_1d_after: float | None = None
    price_3d_after: float | None = None
    price_5d_after: float | None = None
    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None

    # For SELL: what the stock did after we sold
    post_sell_return_5d: float | None = None
    was_premature_sell: bool | None = None

    # For BUY: did the buy work out?
    entry_price: float | None = None
    buy_outcome_return: float | None = None
    was_bad_buy: bool | None = None

    # For HOLD: did holding cost us?
    hold_period_return: float | None = None
    was_wrong_hold: bool | None = None

    # ---- Rotation tracking (recorded at action time) ----
    # For SELL: which stock replaced this one in the portfolio
    replaced_by_ticker: str | None = None
    replaced_by_price: float | None = None  # Replacement's price at action time
    # For BUY: which stock was sold to make room
    replaced_ticker: str | None = None
    replaced_price: float | None = None  # Replaced stock's price at action time
    # Backfilled: return of the rotation counterpart over 5d
    replaced_by_return_5d: float | None = None
    replaced_return_5d: float | None = None
    rotation_alpha: float | None = None  # replacement_return - sold_stock_return

    # ---- Selection quality (recorded at action time) ----
    screened_avg_pred_return: float | None = None  # Avg predicted return of all screened
    screened_top_pred_return: float | None = None  # Best available predicted return
    selection_alpha_5d: float | None = None  # Our return vs screened avg return (backfilled)

    # ---- Volatility context (recorded at action time) ----
    stock_volatility: float | None = None  # Annualized vol at action time
    vol_adjusted_return_5d: float | None = None  # return_5d / (stock_vol / sqrt(252) * 5)

    # ---- Confidence calibration ----
    confidence_calibration: float | None = None  # How aligned was confidence with outcome

    # ---- Composite score breakdown ----
    action_reward: float | None = None  # Final composite reward
    reward_components: dict | None = None  # Per-component breakdown

    def key(self) -> tuple[str, str, str]:
        """Dedup key: (date, ticker, action)."""
        return (self.date, self.ticker, self.action)


@dataclass
class ActionRewardLog:
    """Persistent log of per-action reward entries."""

    entries: list[ActionRewardEntry] = field(default_factory=list)
    max_days: int = 365

    @classmethod
    def load(cls, path: str | Path, max_days: int = 365) -> "ActionRewardLog":
        """Load from JSON or return empty log."""
        p = Path(path)
        if not p.exists():
            return cls(max_days=max_days)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(max_days=max_days)

        raw = data if isinstance(data, list) else data.get("entries", [])
        entries: list[ActionRewardEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(ActionRewardEntry(
                    date=str(item["date"]),
                    ticker=str(item["ticker"]),
                    action=str(item["action"]),
                    reason=str(item.get("reason", "")),
                    price_at_action=float(item["price_at_action"]),
                    shares=float(item.get("shares", 0)),
                    predicted_return=float(item.get("predicted_return", 0)),
                    confidence=float(item.get("confidence", 0)),
                    price_1d_after=item.get("price_1d_after"),
                    price_3d_after=item.get("price_3d_after"),
                    price_5d_after=item.get("price_5d_after"),
                    return_1d=item.get("return_1d"),
                    return_3d=item.get("return_3d"),
                    return_5d=item.get("return_5d"),
                    post_sell_return_5d=item.get("post_sell_return_5d"),
                    was_premature_sell=item.get("was_premature_sell"),
                    entry_price=item.get("entry_price"),
                    buy_outcome_return=item.get("buy_outcome_return"),
                    was_bad_buy=item.get("was_bad_buy"),
                    hold_period_return=item.get("hold_period_return"),
                    was_wrong_hold=item.get("was_wrong_hold"),
                    replaced_by_ticker=item.get("replaced_by_ticker"),
                    replaced_by_price=item.get("replaced_by_price"),
                    replaced_ticker=item.get("replaced_ticker"),
                    replaced_price=item.get("replaced_price"),
                    replaced_by_return_5d=item.get("replaced_by_return_5d"),
                    replaced_return_5d=item.get("replaced_return_5d"),
                    rotation_alpha=item.get("rotation_alpha"),
                    screened_avg_pred_return=item.get("screened_avg_pred_return"),
                    screened_top_pred_return=item.get("screened_top_pred_return"),
                    selection_alpha_5d=item.get("selection_alpha_5d"),
                    stock_volatility=item.get("stock_volatility"),
                    vol_adjusted_return_5d=item.get("vol_adjusted_return_5d"),
                    confidence_calibration=item.get("confidence_calibration"),
                    action_reward=item.get("action_reward"),
                    reward_components=item.get("reward_components"),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        log = cls(entries=entries, max_days=max_days)
        log._trim()
        return log

    def save(self, path: str | Path) -> None:
        """Persist to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(e) for e in self.entries]
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def append_batch(self, entries: list[ActionRewardEntry]) -> None:
        """Append entries, deduplicating by (date, ticker, action)."""
        new_keys = {e.key() for e in entries}
        self.entries = [e for e in self.entries if e.key() not in new_keys]
        self.entries.extend(entries)
        self._trim()

    def backfill_prices(
        self,
        prices_cad: pd.Series,
        current_date: str,
        *,
        price_history_cad: pd.DataFrame | None = None,
    ) -> int:
        """Back-fill post-action prices, rotation returns, and derived metrics.

        Fills 1d/3d/5d prices, rotation counterpart returns, vol-adjusted
        returns, selection alpha, and confidence calibration once enough
        data is available.  Returns count of fields updated.
        """
        history = None
        history_col_map: dict[str, str] = {}
        current_day = pd.Timestamp(current_date)
        if current_day.tzinfo is not None:
            current_day = current_day.tz_convert("UTC").tz_localize(None)
        current_day = current_day.normalize()
        if price_history_cad is not None and not price_history_cad.empty:
            history = price_history_cad.copy()
            idx = pd.to_datetime(history.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            history.index = idx.normalize()
            history = history[~history.index.duplicated(keep="last")].sort_index()
            history_col_map = {str(c).upper(): str(c) for c in history.columns}

        def _history_horizon_price(ticker: str, action_date: str, horizon_td: int) -> float | None:
            if history is None or horizon_td <= 0:
                return None
            ticker_norm = str(ticker).strip().upper()
            if not ticker_norm:
                return None
            col = history_col_map.get(ticker_norm)
            if col is None:
                return None
            try:
                start_dt = _date_to_utc_datetime(action_date)
                target_dt = _add_trading_days(start_dt, int(horizon_td), market=_market_for_ticker(ticker_norm))
            except Exception:
                return None
            target_day = pd.Timestamp(target_dt)
            if target_day.tzinfo is not None:
                target_day = target_day.tz_convert("UTC").tz_localize(None)
            target_day = target_day.normalize()
            series = pd.to_numeric(history[col], errors="coerce")
            val = series.get(target_day)
            if val is not None and not pd.isna(val) and float(val) > 0:
                return float(val)
            future = series.loc[series.index >= target_day].dropna()
            if future.empty:
                return None
            candidate_day = future.index[0]
            if candidate_day > current_day:
                return None
            candidate_px = float(future.iloc[0])
            if pd.isna(candidate_px) or candidate_px <= 0:
                return None
            return candidate_px

        updated = 0
        for e in self.entries:
            if e.price_at_action is None or e.price_at_action <= 0:
                continue
            px = prices_cad.get(e.ticker)
            if px is None or pd.isna(px) or float(px) <= 0:
                continue
            px_f = float(px)
            try:
                days_elapsed = _trading_days_between(
                    _date_to_utc_datetime(e.date),
                    _date_to_utc_datetime(current_date),
                    market=_market_for_ticker(e.ticker),
                )
            except Exception:
                # Fallback for malformed dates: keep prior calendar-day behavior.
                days_elapsed = (pd.Timestamp(current_date) - pd.Timestamp(e.date)).days

            if days_elapsed >= 1 and e.price_1d_after is None:
                horizon_px = _history_horizon_price(e.ticker, e.date, 1)
                px_1d = horizon_px if horizon_px is not None else px_f
                e.price_1d_after = px_1d
                e.return_1d = (px_1d - e.price_at_action) / e.price_at_action
                updated += 1
            if days_elapsed >= 3 and e.price_3d_after is None:
                horizon_px = _history_horizon_price(e.ticker, e.date, 3)
                px_3d = horizon_px if horizon_px is not None else px_f
                e.price_3d_after = px_3d
                e.return_3d = (px_3d - e.price_at_action) / e.price_at_action
                updated += 1
            if days_elapsed >= 5 and e.price_5d_after is None:
                horizon_px = _history_horizon_price(e.ticker, e.date, 5)
                px_5d = horizon_px if horizon_px is not None else px_f
                e.price_5d_after = px_5d
                e.return_5d = (px_5d - e.price_at_action) / e.price_at_action
                updated += 1

                # ---- Classify action quality ----
                if e.action in ("SELL", "SELL_PARTIAL"):
                    e.post_sell_return_5d = e.return_5d
                    e.was_premature_sell = e.return_5d > 0.03
                elif e.action == "BUY":
                    e.buy_outcome_return = e.return_5d
                    e.was_bad_buy = e.return_5d < -0.03
                elif e.action == "HOLD":
                    e.hold_period_return = e.return_5d
                    e.was_wrong_hold = e.return_5d < -0.05

                # ---- Rotation counterpart return ----
                if e.replaced_by_ticker and e.replaced_by_price and e.replaced_by_price > 0:
                    rpl_px = _history_horizon_price(e.replaced_by_ticker, e.date, 5)
                    if rpl_px is None:
                        rpl_px_raw = prices_cad.get(e.replaced_by_ticker)
                        if rpl_px_raw is not None and not pd.isna(rpl_px_raw) and float(rpl_px_raw) > 0:
                            rpl_px = float(rpl_px_raw)
                    if rpl_px is not None and float(rpl_px) > 0:
                        e.replaced_by_return_5d = (float(rpl_px) - e.replaced_by_price) / e.replaced_by_price
                        e.rotation_alpha = e.replaced_by_return_5d - e.return_5d
                        updated += 1
                if e.replaced_ticker and e.replaced_price and e.replaced_price > 0:
                    rpl_px = _history_horizon_price(e.replaced_ticker, e.date, 5)
                    if rpl_px is None:
                        rpl_px_raw = prices_cad.get(e.replaced_ticker)
                        if rpl_px_raw is not None and not pd.isna(rpl_px_raw) and float(rpl_px_raw) > 0:
                            rpl_px = float(rpl_px_raw)
                    if rpl_px is not None and float(rpl_px) > 0:
                        e.replaced_return_5d = (float(rpl_px) - e.replaced_price) / e.replaced_price
                        # For BUY: rotation_alpha = our return - what the sold stock did
                        e.rotation_alpha = e.return_5d - e.replaced_return_5d
                        updated += 1

                # ---- Vol-adjusted return ----
                if e.stock_volatility and e.stock_volatility > 0:
                    # Expected 5-day vol = annual_vol * sqrt(5/252)
                    expected_5d_vol = e.stock_volatility * np.sqrt(5.0 / 252.0)
                    if expected_5d_vol > 1e-6:
                        e.vol_adjusted_return_5d = e.return_5d / expected_5d_vol

                # ---- Selection alpha (vs screened average) ----
                # This uses the *predicted* average as proxy for what
                # alternatives would have returned (imperfect but useful signal).
                if e.screened_avg_pred_return is not None:
                    e.selection_alpha_5d = e.return_5d - e.screened_avg_pred_return

                # ---- Confidence calibration ----
                # Measures alignment between confidence and outcome direction.
                # +1 = high-conf correct, -1 = high-conf wrong, ~0 = low-conf
                if e.confidence and e.confidence > 0:
                    outcome_sign = 1.0 if e.return_5d >= 0 else -1.0
                    pred_sign = 1.0 if e.predicted_return >= 0 else -1.0
                    # Was the directional prediction correct?
                    correct = outcome_sign == pred_sign
                    e.confidence_calibration = e.confidence if correct else -e.confidence

                updated += 1
        return updated

    def entries_for_window(self, last_n_days: int | None = None) -> list[ActionRewardEntry]:
        """Return entries within the last N calendar days."""
        if last_n_days is None:
            return list(self.entries)
        if not self.entries:
            return []
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=last_n_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        return [e for e in self.entries if e.date >= cutoff_str]

    def scored_entries(self) -> list[ActionRewardEntry]:
        """Return entries that have been scored."""
        return [e for e in self.entries if e.action_reward is not None]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entries to a DataFrame."""
        if not self.entries:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.entries])

    def _trim(self) -> None:
        if not self.entries:
            return
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.max_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        self.entries = [e for e in self.entries if e.date >= cutoff_str]
        self.entries.sort(key=lambda e: (e.date, e.ticker, e.action))


# ---------------------------------------------------------------------------
# Original per-ticker prediction tracking
# ---------------------------------------------------------------------------

@dataclass
class RewardEntry:
    """Single observation: prediction vs. realized outcome for one ticker on one date."""

    date: str  # ISO date (YYYY-MM-DD)
    ticker: str
    predicted_return: float  # Calibrated predicted return
    predicted_alpha: float | None = None
    model_score: float | None = None
    confidence: float | None = None
    weight_assigned: float | None = None
    price_at_prediction: float | None = None  # Close price when prediction was made
    price_next_day: float | None = None  # Close price the following day
    realized_1d_return: float | None = None
    realized_cumulative_return: float | None = None
    days_held: int | None = None
    exit_reason: str | None = None
    # Per-model raw predictions (for per-model IC tracking)
    per_model_preds: list[float] | None = None
    # Event type allows prediction rows and close-label rows to coexist safely.
    event_type: str = "PREDICTION"  # PREDICTION | CLOSE

    def key(self) -> tuple[str, ...]:
        """Dedup key.

        For CLOSE rows, include extra fields so multiple same-day closes for one
        ticker are not collapsed into a single record.
        """
        event = self.event_type.upper()
        if event == "CLOSE":
            return (
                self.date,
                self.ticker,
                event,
                str(self.exit_reason or ""),
                str(self.days_held) if self.days_held is not None else "",
            )
        return (self.date, self.ticker, event)


@dataclass
class RewardLog:
    """Persistent log of reward entries backed by a JSON file."""

    entries: list[RewardEntry] = field(default_factory=list)
    max_days: int = 365

    # ---- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, max_days: int = 365) -> "RewardLog":
        """Load from JSON or return empty log."""
        p = Path(path)
        if not p.exists():
            return cls(max_days=max_days)

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(max_days=max_days)

        raw_entries = data if isinstance(data, list) else data.get("entries", [])
        entries: list[RewardEntry] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            try:
                event_type = (
                    str(item.get("event_type")).upper()
                    if item.get("event_type") is not None
                    else ("CLOSE" if item.get("exit_reason") is not None else "PREDICTION")
                )
                if event_type not in {"PREDICTION", "CLOSE"}:
                    event_type = "PREDICTION"
                entries.append(RewardEntry(
                    date=str(item["date"]),
                    ticker=str(item["ticker"]),
                    predicted_return=float(item.get("predicted_return", 0.0)),
                    predicted_alpha=item.get("predicted_alpha"),
                    model_score=item.get("model_score"),
                    confidence=item.get("confidence"),
                    weight_assigned=item.get("weight_assigned"),
                    price_at_prediction=item.get("price_at_prediction"),
                    price_next_day=item.get("price_next_day"),
                    realized_1d_return=item.get("realized_1d_return"),
                    realized_cumulative_return=item.get("realized_cumulative_return"),
                    days_held=item.get("days_held"),
                    exit_reason=item.get("exit_reason"),
                    per_model_preds=item.get("per_model_preds"),
                    event_type=event_type,
                ))
            except (KeyError, TypeError, ValueError):
                continue

        log = cls(entries=entries, max_days=max_days)
        log._trim()
        return log

    def save(self, path: str | Path) -> None:
        """Persist to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(e) for e in self.entries]
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # ---- mutators -----------------------------------------------------------

    def append(self, entry: RewardEntry) -> None:
        """Append entry, replacing any existing entry with the same dedup key."""
        key = entry.key()
        self.entries = [e for e in self.entries if e.key() != key]
        self.entries.append(entry)
        self._trim()

    def append_batch(self, entries: list[RewardEntry]) -> None:
        """Append multiple entries, deduplicating by key (including within batch)."""
        deduped: dict[tuple[str, ...], RewardEntry] = {}
        for e in entries:
            deduped[e.key()] = e
        new_entries = list(deduped.values())
        new_keys = set(deduped.keys())
        self.entries = [e for e in self.entries if e.key() not in new_keys]
        self.entries.extend(new_entries)
        self._trim()

    def update_realized_returns(
        self,
        prices_cad: pd.Series,
        date_str: str,
    ) -> int:
        """Back-fill realized_1d_return for the most recent unresolved prediction day.

        We intentionally update only one cohort (latest prior prediction date)
        per run to avoid labeling stale multi-day moves as "1d" returns.
        Returns count updated.
        """
        pending_dates = sorted({
            str(e.date)
            for e in self.entries
            if str(getattr(e, "event_type", "PREDICTION")).upper() == "PREDICTION"
            and e.realized_1d_return is None
            and str(e.date) < str(date_str)
        })
        if not pending_dates:
            return 0
        target_date = pending_dates[-1]

        updated = 0
        for e in self.entries:
            if str(e.event_type).upper() != "PREDICTION":
                continue
            if str(e.date) != target_date:
                continue
            if e.realized_1d_return is not None:
                continue  # Already filled
            if e.price_at_prediction is None or e.price_at_prediction <= 0:
                continue
            px = prices_cad.get(e.ticker)
            if px is None or pd.isna(px) or float(px) <= 0:
                continue
            e.price_next_day = float(px)
            e.realized_1d_return = (float(px) - e.price_at_prediction) / e.price_at_prediction
            updated += 1
        return updated

    # ---- queries ------------------------------------------------------------

    def entries_for_window(self, last_n_days: int | None = None) -> list[RewardEntry]:
        """Return entries within the last N calendar days (or all if None)."""
        if last_n_days is None:
            return list(self.entries)
        if not self.entries:
            return []
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=last_n_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        return [e for e in self.entries if e.date >= cutoff_str]

    def closed_entries(self) -> list[RewardEntry]:
        """Return entries that have an exit_reason (completed trades)."""
        return [e for e in self.entries if e.exit_reason is not None]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entries to a DataFrame for analysis."""
        if not self.entries:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.entries])

    # ---- internals ----------------------------------------------------------

    def _trim(self) -> None:
        """Keep only the most recent max_days of data."""
        if not self.entries:
            return
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.max_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        self.entries = [e for e in self.entries if e.date >= cutoff_str]
        # Secondary sort to keep chronological order
        self.entries.sort(key=lambda e: (e.date, e.ticker, e.event_type))
