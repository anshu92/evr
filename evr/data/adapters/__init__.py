"""Data adapters for different sources."""

from .base import DataAdapter
from .yfinance import YFinanceAdapter
from .stooq import StooqAdapter

__all__ = [
    "DataAdapter",
    "YFinanceAdapter",
    "StooqAdapter",
]
