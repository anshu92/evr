from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    """Runtime config for the daily pipeline."""

    # Universe
    max_us_tickers: int | None = None
    max_tsx_tickers: int | None = None
    max_total_tickers: int | None = 4000

    # Data windows
    liquidity_lookback_days: int = 30
    feature_lookback_days: int = 180

    # Filters
    min_price_cad: float = 2.0
    min_avg_dollar_volume_cad: float = 250_000.0

    # Screening + portfolio
    top_n: int = 50
    portfolio_size: int = 5
    weight_cap: float = 0.10

    # ML model (optional)
    use_ml: bool = False
    model_path: str = "models/ensemble/manifest.json"
    label_horizon_days: int = 5

    # Portfolio/trading (stateful)
    portfolio_budget_cad: float = 500.0
    max_holding_days: int = 5
    max_holding_days_hard: int = 10
    extend_hold_min_pred_return: float | None = 0.03
    extend_hold_min_score: float | None = None
    portfolio_state_path: str = "screener_portfolio_state.json"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    # FX
    fx_ticker: str = "USDCAD=X"  # USD->CAD
    base_currency: str = "CAD"

    # TSX directory endpoint (can change; allow override)
    tsx_directory_url: str = "https://www.tsx.com/json/company-directory/search"

    # IO locations
    cache_dir: str = "cache"
    data_cache_dir: str = "data_cache"
    reports_dir: str = "reports"

    # Operational
    batch_size: int = 200
    yfinance_threads: bool = True

    @staticmethod
    def from_env() -> "Config":
        """Create config from environment variables."""

        def _get_int(name: str, default: int | None) -> int | None:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            return int(raw)

        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            return float(raw)

        def _get_str(name: str, default: str) -> str:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            return raw

        # Hard cap: keep the portfolio to <= 5 tickers.
        # This enforces the strategy constraint even if an env var attempts to override it.
        ps = _get_int("PORTFOLIO_SIZE", 5) or 5
        ps = max(1, min(int(ps), 5))

        return Config(
            max_us_tickers=_get_int("MAX_US_TICKERS", None),
            max_tsx_tickers=_get_int("MAX_TSX_TICKERS", None),
            max_total_tickers=_get_int("MAX_TICKERS", 4000),
            liquidity_lookback_days=_get_int("LIQUIDITY_LOOKBACK_DAYS", 30) or 30,
            feature_lookback_days=_get_int("FEATURE_LOOKBACK_DAYS", 180) or 180,
            min_price_cad=_get_float("MIN_PRICE_CAD", 2.0),
            min_avg_dollar_volume_cad=_get_float("MIN_AVG_DOLLAR_VOLUME_CAD", 250_000.0),
            top_n=_get_int("TOP_N", 50) or 50,
            portfolio_size=ps,
            weight_cap=_get_float("WEIGHT_CAP", 0.10),
            use_ml=os.getenv("USE_ML", "0").strip() in {"1", "true", "True"},
            model_path=_get_str("MODEL_PATH", "models/ensemble/manifest.json"),
            label_horizon_days=_get_int("LABEL_HORIZON_DAYS", 5) or 5,
            portfolio_budget_cad=_get_float("PORTFOLIO_BUDGET_CAD", 500.0),
            max_holding_days=_get_int("MAX_HOLDING_DAYS", 5) or 5,
            max_holding_days_hard=_get_int("MAX_HOLDING_DAYS_HARD", 10) or 10,
            extend_hold_min_pred_return=(
                _get_float("EXTEND_HOLD_MIN_PRED_RETURN", 0.0)
                if os.getenv("EXTEND_HOLD_MIN_PRED_RETURN") not in {None, ""}
                else None
            ),
            extend_hold_min_score=(
                _get_float("EXTEND_HOLD_MIN_SCORE", 0.0) if os.getenv("EXTEND_HOLD_MIN_SCORE") not in {None, ""} else None
            ),
            portfolio_state_path=_get_str("PORTFOLIO_STATE_PATH", "screener_portfolio_state.json"),
            stop_loss_pct=(
                _get_float("STOP_LOSS_PCT", 0.0) if os.getenv("STOP_LOSS_PCT") not in {None, ""} else None
            ),
            take_profit_pct=(
                _get_float("TAKE_PROFIT_PCT", 0.0) if os.getenv("TAKE_PROFIT_PCT") not in {None, ""} else None
            ),
            fx_ticker=_get_str("FX_TICKER", "USDCAD=X"),
            base_currency=_get_str("BASE_CURRENCY", "CAD"),
            tsx_directory_url=_get_str(
                "TSX_DIRECTORY_URL", "https://www.tsx.com/json/company-directory/search"
            ),
            cache_dir=_get_str("CACHE_DIR", "cache"),
            data_cache_dir=_get_str("DATA_CACHE_DIR", "data_cache"),
            reports_dir=_get_str("REPORTS_DIR", "reports"),
            batch_size=_get_int("BATCH_SIZE", 200) or 200,
            yfinance_threads=os.getenv("YFINANCE_THREADS", "1").strip() not in {"0", "false", "False"},
        )


