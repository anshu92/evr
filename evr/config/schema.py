"""Configuration schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DataConfig:
    """Data source configuration."""
    
    # Data sources
    primary_source: str = "yfinance"  # yfinance, stooq, fred
    fallback_sources: List[str] = field(default_factory=lambda: ["stooq"])
    
    # Cache settings
    cache_dir: str = "~/.evr/data"
    cache_format: str = "parquet"  # parquet, feather
    cache_ttl_days: int = 1
    
    # Universe settings
    universe_file: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    
    # Data settings
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "1d"  # 1d, 1h, 5m, etc.
    adjust_splits: bool = True
    adjust_dividends: bool = True
    timezone: str = "America/Toronto"
    
    # API settings
    api_keys: Dict[str, str] = field(default_factory=dict)
    rate_limit_delay: float = 0.1  # seconds between requests


@dataclass
class ProbConfig:
    """Probability model configuration."""
    
    # Rolling window settings
    window_size: int = 252  # trading days
    min_samples: int = 50
    
    # Smoothing parameters
    alpha: float = 0.1  # EWMA smoothing factor
    beta_prior_alpha: float = 1.0  # Beta distribution alpha
    beta_prior_beta: float = 1.0   # Beta distribution beta
    
    # Regime detection
    use_regimes: bool = False
    regime_window: int = 63  # ~3 months
    regime_threshold: float = 0.1
    
    # Update frequency
    update_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Portfolio settings
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_sector_exposure: float = 0.3  # 30% per sector
    max_correlation: float = 0.7  # Max correlation between positions
    
    # Risk per trade
    max_risk_per_trade: float = 0.02  # 2% per trade
    min_risk_per_trade: float = 0.005  # 0.5% per trade
    
    # Kelly sizing
    use_kelly: bool = True
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    max_kelly_fraction: float = 0.5  # Cap on Kelly fraction
    
    # Circuit breakers
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    weekly_loss_limit: float = 0.15  # 15% weekly loss limit
    max_drawdown_limit: float = 0.25  # 25% max drawdown limit
    
    # Position limits
    max_positions: int = 20
    min_positions: int = 1
    
    # Slippage and costs
    commission_per_trade: float = 1.0  # $ per trade
    slippage_bps: float = 5.0  # basis points
    slippage_atr_multiplier: float = 0.5  # ATR multiplier for slippage


@dataclass
class ReportConfig:
    """Reporting configuration."""
    
    # Output settings
    output_dir: str = "./runs"
    format: str = "html"  # html, markdown, json
    include_charts: bool = True
    
    # Chart settings
    chart_style: str = "seaborn"
    chart_size: tuple = (12, 8)
    dpi: int = 300
    
    # Report sections
    include_equity_curve: bool = True
    include_drawdown: bool = True
    include_returns_histogram: bool = True
    include_trade_analysis: bool = True
    include_setup_analysis: bool = True
    include_calibration: bool = True
    
    # Template settings
    template_dir: Optional[str] = None
    custom_css: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    
    data: DataConfig = field(default_factory=DataConfig)
    prob: ProbConfig = field(default_factory=ProbConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # Global settings
    name: str = "EVR Strategy"
    description: str = "Empirical Volatility Regime Trading Strategy"
    version: str = "1.0.0"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance
    n_jobs: int = -1  # -1 for all cores
    memory_limit: str = "4GB"
    
    # Validation
    validate_config: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.validate_config:
            # Validate data config
            if self.data.start_date and self.data.end_date:
                start = pd.to_datetime(self.data.start_date)
                end = pd.to_datetime(self.data.end_date)
                if start >= end:
                    raise ValueError("start_date must be before end_date")
            
            # Validate risk config
            if self.risk.max_risk_per_trade <= self.risk.min_risk_per_trade:
                raise ValueError("max_risk_per_trade must be greater than min_risk_per_trade")
            
            if self.risk.kelly_fraction <= 0 or self.risk.kelly_fraction > 1:
                raise ValueError("kelly_fraction must be between 0 and 1")
            
            if self.risk.max_kelly_fraction <= self.risk.kelly_fraction:
                raise ValueError("max_kelly_fraction must be greater than kelly_fraction")
            
            # Validate prob config
            if self.prob.window_size <= 0:
                raise ValueError("window_size must be positive")
            
            if self.prob.min_samples >= self.prob.window_size:
                raise ValueError("min_samples must be less than window_size")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": self.data.__dict__,
            "prob": self.prob.__dict__,
            "risk": self.risk.__dict__,
            "report": self.report.__dict__,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "n_jobs": self.n_jobs,
            "memory_limit": self.memory_limit,
            "validate_config": self.validate_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """Create config from dictionary."""
        config = cls()
        
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "prob" in data:
            config.prob = ProbConfig(**data["prob"])
        if "risk" in data:
            config.risk = RiskConfig(**data["risk"])
        if "report" in data:
            config.report = ReportConfig(**data["report"])
        
        # Set global attributes
        for key in ["name", "description", "version", "log_level", "log_file", 
                   "n_jobs", "memory_limit", "validate_config"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
