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
    trade_cost_bps: float = 0.0
    
    # Training configuration
    use_fundamentals_in_training: bool = False  # Set to False to avoid lookahead bias
    train_ensemble_seeds: list[int] | None = None  # Random seeds for ensemble (default: [7, 13, 21, 42, 73, 99, 123])
    train_cv_splits: int = 3
    train_val_window_days: int = 60
    train_embargo_days: int = 10  # Should be >= 2x label_horizon_days to prevent leakage
    fundamentals_cache_ttl_days: int = 7
    
    # Advanced modeling
    use_lightgbm: bool = True
    use_optuna: bool = True
    optuna_n_trials: int = 12
    optuna_timeout_seconds: int = 180
    ensemble_xgb_count: int = 3
    ensemble_lgbm_count: int = 3
    
    # Portfolio optimization
    use_correlation_weights: bool = False  # Requires scipy; set True to enable
    confidence_weight_floor: float = 0.3
    
    # Training target and objective
    use_market_relative_returns: bool = True  # Train on alpha (stock return - market return) instead of absolute returns
    use_ranking_objective: bool = True  # Use LTR (learning-to-rank) objective instead of regression
    
    # Portfolio construction
    sector_neutral_selection: bool = True  # Diversify picks across sectors instead of pure top-N
    volatility_targeting: bool = True  # Scale exposure based on market volatility
    target_volatility: float = 0.15  # Target annualized portfolio volatility (15%)
    
    # Drawdown risk management
    drawdown_management: bool = True  # Reduce exposure when in drawdown
    max_drawdown_threshold: float = -0.10  # Drawdown level to minimize exposure (-10%)
    drawdown_min_scalar: float = 0.25  # Minimum position size multiplier (25%)
    
    # Conviction-based position sizing
    conviction_sizing: bool = True  # Scale positions by prediction strength and confidence
    conviction_min_scalar: float = 0.5  # Minimum position scaling (0.5x normal)
    conviction_max_scalar: float = 2.0  # Maximum position scaling (2x normal)
    
    # Liquidity-adjusted sizing
    liquidity_adjustment: bool = True  # Scale positions by stock liquidity
    min_liquidity_cad: float = 100_000  # Min daily volume to include stock
    target_liquidity_cad: float = 1_000_000  # Full-size threshold
    max_position_pct_of_volume: float = 0.05  # Max position as % of daily volume
    
    # Correlation-based position limits
    correlation_limits: bool = True  # Limit combined weight of correlated stocks
    max_corr_weight: float = 0.25  # Max combined weight for correlated pair (25%)
    corr_threshold: float = 0.70  # Correlation level to trigger limit (70%)
    
    # Beta-adjusted position sizing
    beta_adjustment: bool = True  # Scale positions by inverse beta
    target_portfolio_beta: float = 1.0  # Target overall portfolio beta
    beta_min_scalar: float = 0.5  # Min adjustment for high-beta stocks
    beta_max_scalar: float = 1.5  # Max adjustment for low-beta stocks
    
    # Maximum position cap
    max_position_pct: float = 0.20  # Max 20% in any single position
    
    # Minimum position size
    min_position_pct: float = 0.02  # Min 2% - remove smaller "dust" positions
    
    # Regime-aware exposure scaling
    regime_exposure_enabled: bool = True  # Scale exposure based on market regime
    regime_trend_weight: float = 0.4  # Weight for market trend signal
    regime_breadth_weight: float = 0.3  # Weight for market breadth signal
    regime_vol_weight: float = 0.3  # Weight for inverse volatility signal
    regime_min_scalar: float = 0.5  # Minimum exposure in bearish regime
    regime_max_scalar: float = 1.2  # Maximum exposure in bullish regime

    # Portfolio/trading (stateful)
    portfolio_budget_cad: float = 500.0
    max_holding_days: int = 5
    max_holding_days_hard: int = 10
    peak_based_exit: bool = True  # Exit at peak (trailing stop) instead of fixed days
    
    # Time-weighted return optimization
    twr_optimization: bool = True  # Optimize for time-weighted returns
    quick_profit_pct: float = 0.05  # Take profit if hit 5% gain quickly
    quick_profit_days: int = 3  # "Quickly" means within 3 days
    min_daily_return: float = 0.005  # Exit if daily return drops below 0.5%/day
    momentum_decay_exit: bool = True  # Exit when return momentum decelerates
    extend_hold_min_pred_return: float | None = 0.03
    extend_hold_min_score: float | None = None
    portfolio_state_path: str = "screener_portfolio_state.json"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    
    # Trailing stop settings
    trailing_stop_enabled: bool = True  # Enable trailing stop-loss
    trailing_stop_activation_pct: float = 0.05  # Activate after 5% gain
    trailing_stop_distance_pct: float = 0.08  # Trail 8% below peak
    
    # Signal decay exit settings
    signal_decay_exit_enabled: bool = True  # Exit when prediction turns negative
    signal_decay_threshold: float = -0.02  # Exit if predicted return drops below -2%
    
    # Dynamic holding period settings
    dynamic_holding_enabled: bool = True  # Adjust holding period based on volatility
    dynamic_holding_vol_scale: float = 0.5  # How much to adjust (0.5 = +/-50% at extremes)
    
    # Volatility-adjusted stop-loss settings
    vol_adjusted_stop_enabled: bool = True  # Adjust stop-loss by stock volatility
    vol_adjusted_stop_base: float = 0.08  # Base stop-loss for average volatility stock
    vol_adjusted_stop_min: float = 0.04  # Minimum stop-loss (low vol stocks)
    vol_adjusted_stop_max: float = 0.15  # Maximum stop-loss (high vol stocks)
    
    # Position age urgency settings
    age_urgency_enabled: bool = True  # Exit underperformers earlier as they age
    age_urgency_start_day: int = 2  # Start applying urgency after N days
    age_urgency_min_return: float = 0.01  # Min return to avoid urgency exit (1%)

    # Peak detection and partial exit settings
    peak_detection_enabled: bool = True
    peak_sell_portion_pct: float = 0.50
    peak_min_gain_pct: float | None = 0.10
    peak_min_holding_days: int = 2
    peak_pred_return_threshold: float | None = -0.02
    peak_score_percentile_drop: float | None = 0.30
    peak_rsi_overbought: float | None = 70.0
    peak_above_ma_ratio: float | None = 0.15
    
    # Entry confirmation filters (reduce false positives)
    entry_min_confidence: float | None = 0.5  # Minimum model confidence to enter
    entry_min_pred_return: float | None = 0.01  # Minimum predicted return (calibrated)
    entry_max_volatility: float | None = 0.60  # Max annualized volatility to enter
    entry_min_momentum_5d: float | None = -0.05  # Reject stocks with recent sharp drops
    entry_momentum_alignment: bool = True  # Reject bullish signals with bearish price action

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

        def _get_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            return raw.lower() in ("true", "1", "yes", "on")

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
            trade_cost_bps=_get_float("TRADE_COST_BPS", 0.0),
            use_fundamentals_in_training=os.getenv("USE_FUNDAMENTALS_IN_TRAINING", "0").strip() in {"1", "true", "True"},
            train_ensemble_seeds=None,  # Use default in train.py
            train_cv_splits=_get_int("TRAIN_CV_SPLITS", 3) or 3,
            train_val_window_days=_get_int("TRAIN_VAL_WINDOW_DAYS", 60) or 60,
            train_embargo_days=_get_int("TRAIN_EMBARGO_DAYS", 5) or 5,
            fundamentals_cache_ttl_days=_get_int("FUNDAMENTALS_CACHE_TTL_DAYS", 7) or 7,
            use_lightgbm=os.getenv("USE_LIGHTGBM", "1").strip() in {"1", "true", "True"},
            use_optuna=os.getenv("USE_OPTUNA", "1").strip() in {"1", "true", "True"},
            optuna_n_trials=_get_int("OPTUNA_N_TRIALS", 12) or 12,
            optuna_timeout_seconds=_get_int("OPTUNA_TIMEOUT_SECONDS", 180) or 180,
            ensemble_xgb_count=_get_int("ENSEMBLE_XGB_COUNT", 3) or 3,
            ensemble_lgbm_count=_get_int("ENSEMBLE_LGBM_COUNT", 3) or 3,
            use_correlation_weights=os.getenv("USE_CORRELATION_WEIGHTS", "0").strip() in {"1", "true", "True"},
            confidence_weight_floor=_get_float("CONFIDENCE_WEIGHT_FLOOR", 0.3),
            use_market_relative_returns=os.getenv("USE_MARKET_RELATIVE_RETURNS", "1").strip() in {"1", "true", "True"},
            use_ranking_objective=os.getenv("USE_RANKING_OBJECTIVE", "1").strip() in {"1", "true", "True"},
            sector_neutral_selection=os.getenv("SECTOR_NEUTRAL_SELECTION", "1").strip() in {"1", "true", "True"},
            volatility_targeting=os.getenv("VOLATILITY_TARGETING", "1").strip() in {"1", "true", "True"},
            target_volatility=_get_float("TARGET_VOLATILITY", 0.15),
            drawdown_management=os.getenv("DRAWDOWN_MANAGEMENT", "1").strip() in {"1", "true", "True"},
            max_drawdown_threshold=_get_float("MAX_DRAWDOWN_THRESHOLD", -0.10),
            drawdown_min_scalar=_get_float("DRAWDOWN_MIN_SCALAR", 0.25),
            conviction_sizing=os.getenv("CONVICTION_SIZING", "1").strip() in {"1", "true", "True"},
            conviction_min_scalar=_get_float("CONVICTION_MIN_SCALAR", 0.5),
            conviction_max_scalar=_get_float("CONVICTION_MAX_SCALAR", 2.0),
            liquidity_adjustment=os.getenv("LIQUIDITY_ADJUSTMENT", "1").strip() in {"1", "true", "True"},
            min_liquidity_cad=_get_float("MIN_LIQUIDITY_CAD", 100_000),
            target_liquidity_cad=_get_float("TARGET_LIQUIDITY_CAD", 1_000_000),
            max_position_pct_of_volume=_get_float("MAX_POSITION_PCT_OF_VOLUME", 0.05),
            correlation_limits=os.getenv("CORRELATION_LIMITS", "1").strip() in {"1", "true", "True"},
            max_corr_weight=_get_float("MAX_CORR_WEIGHT", 0.25),
            corr_threshold=_get_float("CORR_THRESHOLD", 0.70),
            beta_adjustment=os.getenv("BETA_ADJUSTMENT", "1").strip() in {"1", "true", "True"},
            target_portfolio_beta=_get_float("TARGET_PORTFOLIO_BETA", 1.0),
            beta_min_scalar=_get_float("BETA_MIN_SCALAR", 0.5),
            beta_max_scalar=_get_float("BETA_MAX_SCALAR", 1.5),
            max_position_pct=_get_float("MAX_POSITION_PCT", 0.20),
            min_position_pct=_get_float("MIN_POSITION_PCT", 0.02),
            regime_exposure_enabled=_get_bool("REGIME_EXPOSURE_ENABLED", True),
            regime_trend_weight=_get_float("REGIME_TREND_WEIGHT", 0.4),
            regime_breadth_weight=_get_float("REGIME_BREADTH_WEIGHT", 0.3),
            regime_vol_weight=_get_float("REGIME_VOL_WEIGHT", 0.3),
            regime_min_scalar=_get_float("REGIME_MIN_SCALAR", 0.5),
            regime_max_scalar=_get_float("REGIME_MAX_SCALAR", 1.2),
            portfolio_budget_cad=_get_float("PORTFOLIO_BUDGET_CAD", 500.0),
            max_holding_days=_get_int("MAX_HOLDING_DAYS", 5) or 5,
            max_holding_days_hard=_get_int("MAX_HOLDING_DAYS_HARD", 10) or 10,
            peak_based_exit=_get_bool("PEAK_BASED_EXIT", True),
            twr_optimization=_get_bool("TWR_OPTIMIZATION", True),
            quick_profit_pct=_get_float("QUICK_PROFIT_PCT", 0.05),
            quick_profit_days=_get_int("QUICK_PROFIT_DAYS", 3) or 3,
            min_daily_return=_get_float("MIN_DAILY_RETURN", 0.005),
            momentum_decay_exit=_get_bool("MOMENTUM_DECAY_EXIT", True),
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
            trailing_stop_enabled=os.getenv("TRAILING_STOP_ENABLED", "1").strip() in {"1", "true", "True"},
            trailing_stop_activation_pct=_get_float("TRAILING_STOP_ACTIVATION_PCT", 0.05),
            trailing_stop_distance_pct=_get_float("TRAILING_STOP_DISTANCE_PCT", 0.08),
            signal_decay_exit_enabled=os.getenv("SIGNAL_DECAY_EXIT_ENABLED", "1").strip() in {"1", "true", "True"},
            signal_decay_threshold=_get_float("SIGNAL_DECAY_THRESHOLD", -0.02),
            dynamic_holding_enabled=os.getenv("DYNAMIC_HOLDING_ENABLED", "1").strip() in {"1", "true", "True"},
            dynamic_holding_vol_scale=_get_float("DYNAMIC_HOLDING_VOL_SCALE", 0.5),
            vol_adjusted_stop_enabled=os.getenv("VOL_ADJUSTED_STOP_ENABLED", "1").strip() in {"1", "true", "True"},
            vol_adjusted_stop_base=_get_float("VOL_ADJUSTED_STOP_BASE", 0.08),
            vol_adjusted_stop_min=_get_float("VOL_ADJUSTED_STOP_MIN", 0.04),
            vol_adjusted_stop_max=_get_float("VOL_ADJUSTED_STOP_MAX", 0.15),
            age_urgency_enabled=os.getenv("AGE_URGENCY_ENABLED", "1").strip() in {"1", "true", "True"},
            age_urgency_start_day=_get_int("AGE_URGENCY_START_DAY", 2) or 2,
            age_urgency_min_return=_get_float("AGE_URGENCY_MIN_RETURN", 0.01),
            peak_detection_enabled=os.getenv("PEAK_DETECTION_ENABLED", "1").strip() in {"1", "true", "True"},
            peak_sell_portion_pct=_get_float("PEAK_SELL_PORTION_PCT", 0.50),
            peak_min_gain_pct=(
                _get_float("PEAK_MIN_GAIN_PCT", 0.0)
                if os.getenv("PEAK_MIN_GAIN_PCT") not in {None, ""}
                else None
            ),
            peak_min_holding_days=_get_int("PEAK_MIN_HOLDING_DAYS", 2) or 2,
            peak_pred_return_threshold=(
                _get_float("PEAK_PRED_RETURN_THRESHOLD", 0.0)
                if os.getenv("PEAK_PRED_RETURN_THRESHOLD") not in {None, ""}
                else None
            ),
            peak_score_percentile_drop=(
                _get_float("PEAK_SCORE_PERCENTILE_DROP", 0.0)
                if os.getenv("PEAK_SCORE_PERCENTILE_DROP") not in {None, ""}
                else None
            ),
            peak_rsi_overbought=(
                _get_float("PEAK_RSI_OVERBOUGHT", 0.0)
                if os.getenv("PEAK_RSI_OVERBOUGHT") not in {None, ""}
                else None
            ),
            peak_above_ma_ratio=(
                _get_float("PEAK_ABOVE_MA_RATIO", 0.0)
                if os.getenv("PEAK_ABOVE_MA_RATIO") not in {None, ""}
                else None
            ),
            entry_min_confidence=(
                _get_float("ENTRY_MIN_CONFIDENCE", 0.5)
                if os.getenv("ENTRY_MIN_CONFIDENCE") not in {None, ""}
                else None
            ),
            entry_min_pred_return=(
                _get_float("ENTRY_MIN_PRED_RETURN", 0.01)
                if os.getenv("ENTRY_MIN_PRED_RETURN") not in {None, ""}
                else None
            ),
            entry_max_volatility=(
                _get_float("ENTRY_MAX_VOLATILITY", 0.60)
                if os.getenv("ENTRY_MAX_VOLATILITY") not in {None, ""}
                else None
            ),
            entry_min_momentum_5d=(
                _get_float("ENTRY_MIN_MOMENTUM_5D", -0.05)
                if os.getenv("ENTRY_MIN_MOMENTUM_5D") not in {None, ""}
                else None
            ),
            entry_momentum_alignment=os.getenv("ENTRY_MOMENTUM_ALIGNMENT", "1").strip() in {"1", "true", "True"},
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

