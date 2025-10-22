"""EVR - Empirical Volatility Regime Trading Framework."""

__version__ = "1.0.0"
__author__ = "EVR Contributors"
__email__ = "evr@example.com"

from .types import Bar, Signal, TradePlan, TradeResult, Metrics
from .config import load_config

__all__ = [
    "Bar",
    "Signal", 
    "TradePlan",
    "TradeResult",
    "Metrics",
    "load_config",
]
