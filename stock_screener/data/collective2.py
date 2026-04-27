from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import requests

from stock_screener.utils import sanitize_ticker


_BASE_URL = "https://api.collective2.com/world/apiv3"


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class C2System:
    system_id: str
    system_name: str
    owner_screenname: str
    trades_stocks: bool
    trades_stocks_short: bool
    trades_options: bool
    trades_futures: bool
    trades_forex: bool
    minimum_portfolio_size_required: float
    monthly_fee: float
    free_trial_days: int
    created_when: str
    is_alive: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class C2Signal:
    system_id: str
    signal_id: str
    symbol: str
    action: str
    quantity: float
    status: str
    instrument: str
    posted_time: str
    posted_time_unix: int
    traded_time_unix: int
    is_market_order: bool
    is_limit_order: bool
    is_stop_order: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class C2Trade:
    system_id: str
    trade_id: str
    symbol: str
    instrument: str
    long_or_short: str
    open_or_closed: str
    opened_when: str
    closed_when: str
    opening_price: float
    closing_price: float
    pl: float
    raw: dict[str, Any]


class Collective2Client:
    """Minimal Collective2 World API client for paper-copy signal ingestion."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _BASE_URL,
        timeout_seconds: int = 20,
        throttle_seconds: float = 1.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Collective2 API key is required")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.throttle_seconds = max(0.0, float(throttle_seconds))
        self._last_request_ts = 0.0

    def post(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.time() - self._last_request_ts
        if self._last_request_ts > 0 and elapsed < self.throttle_seconds:
            time.sleep(self.throttle_seconds - elapsed)
        self._last_request_ts = time.time()

        body = dict(payload or {})
        body["apikey"] = self.api_key
        resp = requests.post(
            f"{self.base_url}/{command}",
            json=body,
            timeout=self.timeout_seconds,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Collective2 response for {command}: {type(data)!r}")
        return data

    def get_system_roster(self, *, filter_value: str = "active") -> list[C2System]:
        data = self.post("getSystemRoster", {"filter": filter_value})
        return [parse_system(x) for x in _response_list(data)]

    def list_all_systems(self) -> list[dict[str, Any]]:
        data = self.post("listAllSystems", {})
        return [x for x in _response_list(data) if isinstance(x, dict)]

    def get_system_details(self, system_id: str) -> dict[str, Any]:
        data = self.post("getSystemDetails", {"systemid": str(system_id)})
        response = data.get("response")
        return response if isinstance(response, dict) else {}

    def retrieve_signals_all(
        self,
        system_id: str,
        *,
        filter_type: str,
        start_ny: str,
        end_ny: str,
    ) -> list[C2Signal]:
        data = self.post(
            "retrieveSignalsAll",
            {
                "systemid": str(system_id),
                "filter_type": filter_type,
                "filter_date_time_start": start_ny,
                "filter_date_time_end": end_ny,
            },
        )
        return [
            s for s in (parse_signal(str(system_id), x) for x in _response_list(data))
            if s is not None
        ]

    def retrieve_signals_working(self, system_id: str) -> list[C2Signal]:
        data = self.post("retrieveSignalsWorking", {"systemid": str(system_id)})
        return [
            s for s in (parse_signal(str(system_id), x) for x in _response_list(data))
            if s is not None
        ]

    def request_trades(self, system_id: str, *, open_only: bool = False) -> list[C2Trade]:
        cmd = "requestTradesOpen" if open_only else "requestTrades"
        data = self.post(cmd, {"systemid": str(system_id)})
        return [
            t for t in (parse_trade(str(system_id), x) for x in _response_list(data))
            if t is not None
        ]


def _response_list(data: dict[str, Any]) -> list[Any]:
    response = data.get("response", [])
    return response if isinstance(response, list) else []


def parse_system(raw: dict[str, Any]) -> C2System:
    sid = str(raw.get("system_id") or raw.get("systemid") or "").strip()
    return C2System(
        system_id=sid,
        system_name=str(raw.get("system_name") or raw.get("systemName") or "").strip(),
        owner_screenname=str(raw.get("owner_screenname") or raw.get("creatorScreenName") or "").strip(),
        trades_stocks=_to_bool(raw.get("trades_stocks")),
        trades_stocks_short=_to_bool(raw.get("trades_stocks_short") or raw.get("shorts_stocks")),
        trades_options=_to_bool(raw.get("trades_options") or raw.get("trades_options_short")),
        trades_futures=_to_bool(raw.get("trades_futures")),
        trades_forex=_to_bool(raw.get("trades_forex")),
        minimum_portfolio_size_required=_to_float(raw.get("minimum_portfolio_size_required")),
        monthly_fee=_to_float(raw.get("monthlyFee")),
        free_trial_days=_to_int(raw.get("freeTrialPeriodDays")),
        created_when=str(raw.get("created_when") or raw.get("createdWhen") or "").strip(),
        is_alive=_to_bool(raw.get("isAlive")),
        raw=dict(raw),
    )


def is_long_stock_system(system: C2System, *, max_minimum_portfolio_usd: float) -> bool:
    if not system.system_id:
        return False
    if not system.is_alive:
        return False
    if not system.trades_stocks:
        return False
    if system.trades_stocks_short or system.trades_options or system.trades_futures or system.trades_forex:
        return False
    min_required = float(system.minimum_portfolio_size_required or 0.0)
    return min_required <= float(max_minimum_portfolio_usd)


def parse_signal(system_id: str, raw: Any) -> C2Signal | None:
    if not isinstance(raw, dict):
        return None
    symbol = sanitize_ticker(str(raw.get("symbol") or ""))
    action = str(raw.get("action") or "").strip().upper()
    signal_id = str(raw.get("signal_id") or raw.get("signalid") or raw.get("guid") or "").strip()
    if not symbol or not action or not signal_id:
        return None
    instrument = str(raw.get("instrument") or raw.get("typeofsymbol") or "").strip().lower()
    return C2Signal(
        system_id=str(system_id),
        signal_id=signal_id,
        symbol=symbol,
        action=action,
        quantity=abs(_to_float(raw.get("quant"))),
        status=str(raw.get("status") or "").strip().lower(),
        instrument=instrument,
        posted_time=str(raw.get("posted_time") or "").strip(),
        posted_time_unix=_to_int(raw.get("posted_time_unix") or raw.get("postedwhen")),
        traded_time_unix=_to_int(raw.get("traded_time_unix") or raw.get("tradedwhen")),
        is_market_order=_to_bool(raw.get("isMarketOrder") or raw.get("market")),
        is_limit_order=raw.get("isLimitOrder") not in {None, "", "0", 0},
        is_stop_order=raw.get("isStopOrder") not in {None, "", "0", 0},
        raw=dict(raw),
    )


def parse_trade(system_id: str, raw: Any) -> C2Trade | None:
    if not isinstance(raw, dict):
        return None
    symbol = sanitize_ticker(str(raw.get("symbol") or ""))
    trade_id = str(raw.get("trade_id") or "").strip()
    if not symbol or not trade_id:
        return None
    return C2Trade(
        system_id=str(system_id),
        trade_id=trade_id,
        symbol=symbol,
        instrument=str(raw.get("instrument") or "").strip().lower(),
        long_or_short=str(raw.get("long_or_short") or "").strip().lower(),
        open_or_closed=str(raw.get("open_or_closed") or "").strip().lower(),
        opened_when=str(raw.get("openedWhen") or "").strip(),
        closed_when=str(raw.get("closedWhen") or "").strip(),
        opening_price=_to_float(raw.get("opening_price_VWAP")),
        closing_price=_to_float(raw.get("closing_price_VWAP")),
        pl=_to_float(raw.get("PL")),
        raw=dict(raw),
    )


def signal_posted_at(signal: C2Signal) -> datetime | None:
    if signal.posted_time_unix > 0:
        return datetime.fromtimestamp(signal.posted_time_unix, tz=timezone.utc)
    return None


def is_supported_long_stock_signal(signal: C2Signal) -> bool:
    if signal.instrument not in {"stock", ""}:
        return False
    if signal.action not in {"BTO", "STC"}:
        return False
    status = signal.status.lower()
    if status in {"canceled", "cancelled", "expired"}:
        return False
    return signal.quantity > 0
