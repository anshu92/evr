"""Feature pipeline for lazy computation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from .library import FeatureLibrary


class FeatureGraph:
    """Feature computation graph with lazy evaluation."""
    
    def __init__(self):
        """Initialize feature graph."""
        self._features: Dict[str, Any] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._computed: Set[str] = set()
        self._data: Optional[pd.DataFrame] = None
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for feature computation.
        
        Args:
            data: OHLCV data
        """
        self._data = data.copy()
        self._computed.clear()
    
    def add_feature(
        self,
        name: str,
        func: callable,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Add a feature to the graph.
        
        Args:
            name: Feature name
            func: Function to compute the feature
            dependencies: List of feature dependencies
            **kwargs: Additional arguments for the function
        """
        self._features[name] = {
            'func': func,
            'kwargs': kwargs,
            'dependencies': set(dependencies or [])
        }
        
        # Update dependency graph
        self._dependencies[name] = set(dependencies or [])
    
    def compute_feature(self, name: str) -> pd.Series:
        """Compute a specific feature.
        
        Args:
            name: Feature name
            
        Returns:
            Computed feature series
            
        Raises:
            KeyError: If feature is not defined
            ValueError: If data is not set
        """
        if self._data is None:
            raise ValueError("Data must be set before computing features")
        
        if name in self._computed:
            return self._features[name]['result']
        
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not defined")
        
        feature = self._features[name]
        
        # Compute dependencies first
        for dep in feature['dependencies']:
            if dep not in self._computed:
                self.compute_feature(dep)
        
        # Compute the feature
        try:
            result = feature['func'](self._data, **feature['kwargs'])
            feature['result'] = result
            self._computed.add(name)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute feature '{name}': {e}")
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all features.
        
        Returns:
            DataFrame with all computed features
        """
        if self._data is None:
            raise ValueError("Data must be set before computing features")
        
        results = {}
        
        for name in self._features:
            results[name] = self.compute_feature(name)
        
        return pd.DataFrame(results, index=self._data.index)
    
    def get_feature(self, name: str) -> pd.Series:
        """Get a computed feature.
        
        Args:
            name: Feature name
            
        Returns:
            Feature series
        """
        if name not in self._computed:
            self.compute_feature(name)
        
        return self._features[name]['result']
    
    def has_feature(self, name: str) -> bool:
        """Check if feature is defined.
        
        Args:
            name: Feature name
            
        Returns:
            True if feature is defined
        """
        return name in self._features
    
    def list_features(self) -> List[str]:
        """List all defined features.
        
        Returns:
            List of feature names
        """
        return list(self._features.keys())
    
    def clear_computed(self) -> None:
        """Clear all computed features."""
        self._computed.clear()
        for feature in self._features.values():
            feature.pop('result', None)
    
    def add_technical_indicators(self) -> None:
        """Add common technical indicators to the graph."""
        # RSI
        self.add_feature('rsi_14', FeatureLibrary.rsi, period=14)
        self.add_feature('rsi_21', FeatureLibrary.rsi, period=21)
        
        # ATR
        self.add_feature('atr_14', FeatureLibrary.atr, period=14)
        self.add_feature('atr_21', FeatureLibrary.atr, period=21)
        
        # Bollinger Bands
        self.add_feature('bb_width_20', FeatureLibrary.bb_width, period=20, std=2.0)
        self.add_feature('bb_width_21', FeatureLibrary.bb_width, period=21, std=2.0)
        
        # Z-score
        self.add_feature('zscore_20', FeatureLibrary.zscore, period=20)
        self.add_feature('zscore_50', FeatureLibrary.zscore, period=50)
        
        # ADX
        self.add_feature('adx_14', FeatureLibrary.adx, period=14)
        
        # VWAP
        self.add_feature('vwap', FeatureLibrary.vwap)
        
        # KAMA
        self.add_feature('kama_14', FeatureLibrary.kama, period=14)
        self.add_feature('kama_21', FeatureLibrary.kama, period=21)
        
        # MACD
        self.add_feature('macd_12_26_9', FeatureLibrary.macd, fast=12, slow=26, signal=9)
        
        # Moving Averages
        self.add_feature('sma_20', FeatureLibrary.sma, period=20)
        self.add_feature('sma_50', FeatureLibrary.sma, period=50)
        self.add_feature('ema_12', FeatureLibrary.ema, period=12)
        self.add_feature('ema_26', FeatureLibrary.ema, period=26)
        
        # Stochastic
        self.add_feature('stoch_14_3', FeatureLibrary.stoch, k_period=14, d_period=3)
        
        # Williams %R
        self.add_feature('williams_r_14', FeatureLibrary.williams_r, period=14)
        
        # CCI
        self.add_feature('cci_20', FeatureLibrary.cci, period=20)
        
        # ROC
        self.add_feature('roc_10', FeatureLibrary.roc, period=10)
        
        # Momentum
        self.add_feature('momentum_10', FeatureLibrary.momentum, period=10)
        
        # Volatility
        self.add_feature('volatility_20', FeatureLibrary.volatility, period=20)
        
        # Price Position
        self.add_feature('price_position_20', FeatureLibrary.price_position, period=20)
        
        # Volume Ratio
        self.add_feature('volume_ratio_20', FeatureLibrary.volume_ratio, period=20)
    
    def add_custom_feature(self, name: str, func: callable, **kwargs) -> None:
        """Add a custom feature.
        
        Args:
            name: Feature name
            func: Function to compute the feature
            **kwargs: Additional arguments for the function
        """
        self.add_feature(name, func, **kwargs)
    
    def get_feature_info(self, name: str) -> Dict[str, Any]:
        """Get information about a feature.
        
        Args:
            name: Feature name
            
        Returns:
            Dictionary with feature information
        """
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not defined")
        
        feature = self._features[name]
        return {
            'name': name,
            'func': feature['func'].__name__,
            'dependencies': list(feature['dependencies']),
            'kwargs': feature['kwargs'],
            'computed': name in self._computed,
            'has_result': 'result' in feature
        }
