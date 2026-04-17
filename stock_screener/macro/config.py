from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return default


def _s(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class MacroConfig:
    """Environment-driven macro pipeline settings."""

    portfolio_state_path: str
    portfolio_budget_cad: float
    max_positions: int
    hard_max_positions: int
    target_gross_exposure: float
    min_cash_weight: float
    min_etf_weight: float
    single_stock_max_positions: int
    etf_weight_cap: float
    stock_weight_cap: float
    min_hold_days: int
    soft_max_hold_days: int
    max_hold_days: int
    max_new_positions_per_run: int
    max_full_exits_per_run: int
    memory_sqlite_path: str
    reports_dir: str
    cache_dir: str
    max_runtime_minutes: float
    dry_run: bool
    min_adv_cad: float
    llm_agent_enabled: bool
    llm_decision_primary: bool
    llm_ml_weight: float
    llm_llm_weight: float
    llm_max_tickers: int
    llm_portfolio_reasoning: bool
    llm_portfolio_enforce: bool

    @staticmethod
    def from_env() -> "MacroConfig":
        return MacroConfig(
            portfolio_state_path=_s("MACRO_PORTFOLIO_STATE_PATH", "macro_portfolio_state.json"),
            portfolio_budget_cad=_f("MACRO_PORTFOLIO_BUDGET_CAD", 500.0),
            max_positions=_i("MACRO_PORTFOLIO_MAX_POSITIONS", 6),
            hard_max_positions=_i("MACRO_PORTFOLIO_HARD_MAX_POSITIONS", 8),
            target_gross_exposure=_f("MACRO_TARGET_GROSS_EXPOSURE", 0.85),
            min_cash_weight=_f("MACRO_MIN_CASH_WEIGHT", 0.15),
            min_etf_weight=_f("MACRO_MIN_ETF_WEIGHT", 0.50),
            single_stock_max_positions=_i("MACRO_SINGLE_STOCK_MAX_POSITIONS", 2),
            etf_weight_cap=_f("MACRO_ETF_WEIGHT_CAP", 0.30),
            stock_weight_cap=_f("MACRO_STOCK_WEIGHT_CAP", 0.18),
            min_hold_days=_i("MACRO_MIN_HOLD_DAYS", 2),
            soft_max_hold_days=_i("MACRO_SOFT_MAX_HOLD_DAYS", 10),
            max_hold_days=_i("MACRO_MAX_HOLD_DAYS", 20),
            max_new_positions_per_run=_i("MACRO_MAX_NEW_POSITIONS_PER_RUN", 2),
            max_full_exits_per_run=_i("MACRO_MAX_FULL_EXITS_PER_RUN", 2),
            memory_sqlite_path=_s("MACRO_MEMORY_SQLITE_PATH", "data_runtime/macro_memory.sqlite"),
            reports_dir=_s("MACRO_REPORTS_DIR", "reports"),
            cache_dir=_s("MACRO_CACHE_DIR", "cache"),
            max_runtime_minutes=_f("MAX_MACRO_RUNTIME_MINUTES", 12.0),
            dry_run=_b("MACRO_INSIGHTS_DRY_RUN", False),
            min_adv_cad=_f("MACRO_MIN_ADV_CAD", 5_000_000.0),
            llm_agent_enabled=_b("MACRO_LLM_AGENT_ENABLED", _b("LLM_AGENT_ENABLED", False)),
            llm_decision_primary=_b("MACRO_LLM_PRIMARY", _b("LLM_DECISION_PRIMARY", False)),
            llm_ml_weight=_f("LLM_AGENT_ML_WEIGHT", 0.7),
            llm_llm_weight=_f("LLM_AGENT_LLM_WEIGHT", 0.3),
            llm_max_tickers=_i("MACRO_LLM_MAX_TICKERS", 6),
            llm_portfolio_reasoning=_b(
                "MACRO_LLM_PORTFOLIO_REASONING",
                _b("AGENT_PORTFOLIO_REASONING", True),
            ),
            llm_portfolio_enforce=_b("MACRO_LLM_PORTFOLIO_ENFORCE", True),
        )
