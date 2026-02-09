"""Reward tracker -- logs per-ticker predictions and realized outcomes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


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

    def key(self) -> tuple[str, str]:
        """Dedup key: (date, ticker)."""
        return (self.date, self.ticker)


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
        """Append entry, replacing any existing entry with the same (date, ticker)."""
        key = entry.key()
        self.entries = [e for e in self.entries if e.key() != key]
        self.entries.append(entry)
        self._trim()

    def append_batch(self, entries: list[RewardEntry]) -> None:
        """Append multiple entries, deduplicating by (date, ticker)."""
        new_keys = {e.key() for e in entries}
        self.entries = [e for e in self.entries if e.key() not in new_keys]
        self.entries.extend(entries)
        self._trim()

    def update_realized_returns(
        self,
        prices_cad: pd.Series,
        date_str: str,
    ) -> int:
        """Back-fill realized_1d_return for entries from the previous day.

        For entries whose price_next_day is still None, fills it with the
        current price and computes the 1-day return.  Returns count updated.
        """
        updated = 0
        for e in self.entries:
            if e.date == date_str:
                continue  # Same day; skip -- return not yet realized
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
        self.entries.sort(key=lambda e: (e.date, e.ticker))
