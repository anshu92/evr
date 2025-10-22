"""Base setup class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from ...types import Bars, Features, Signal


class BaseSetup(ABC):
    """Abstract base class for trading setups."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize setup.
        
        Args:
            name: Setup name
            **kwargs: Setup-specific parameters
        """
        self.name = name
        self.parameters = kwargs
    
    @abstractmethod
    def signals(
        self,
        bars: Bars,
        features: Features,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> List[Signal]:
        """Generate trading signals.
        
        Args:
            bars: OHLCV data
            features: Computed features
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate setup parameters.
        
        Returns:
            True if parameters are valid
        """
        pass
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for this setup.
        
        Returns:
            List of required feature names
        """
        return []
    
    def get_parameter(self, key: str, default=None):
        """Get a setup parameter.
        
        Args:
            key: Parameter key
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value) -> None:
        """Set a setup parameter.
        
        Args:
            key: Parameter key
            value: Parameter value
        """
        self.parameters[key] = value
    
    def get_description(self) -> str:
        """Get setup description.
        
        Returns:
            Setup description
        """
        return f"{self.name} setup"
    
    def get_metadata(self) -> dict:
        """Get setup metadata.
        
        Returns:
            Dictionary with setup metadata
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'required_features': self.get_required_features(),
        }
    
    def __repr__(self) -> str:
        """String representation of setup."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
