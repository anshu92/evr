"""Built-in trading setups."""

from .squeeze import SqueezeBreakout
from .pullback import TrendPullback
from .meanrevert import MeanReversion

__all__ = [
    "SqueezeBreakout",
    "TrendPullback", 
    "MeanReversion",
]
