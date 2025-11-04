#!/usr/bin/env python3
"""
Parameter Integration Module

This module loads trained parameters from the historical backtesting
and integrates them into the official scanner's probability model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class TrainedParameterLoader:
    """Loads and manages trained parameters from historical backtesting."""
    
    def __init__(self, parameters_path: str = "trained_parameters/scanner_parameters.json"):
        """Initialize parameter loader.
        
        Args:
            parameters_path: Path to trained parameters JSON file
        """
        self.parameters_path = Path(parameters_path)
        self.parameters = {}
        self.is_loaded = False
        
        # Load parameters if file exists
        if self.parameters_path.exists():
            self.load_parameters()
    
    def load_parameters(self) -> None:
        """Load parameters from JSON file."""
        try:
            with open(self.parameters_path, 'r') as f:
                self.parameters = json.load(f)
            self.is_loaded = True
            logger.info(f"Loaded trained parameters from {self.parameters_path}")
            
            # Log summary
            metadata = self.parameters.get('metadata', {})
            logger.info(f"Parameters trained on {metadata.get('num_trades', 0)} trades "
                       f"from {metadata.get('num_tickers', 0)} tickers")
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.is_loaded = False
    
    def get_global_prior(self) -> Tuple[float, float, float]:
        """Get global prior parameters.
        
        Returns:
            Tuple of (p_win, avg_win, avg_loss)
        """
        if not self.is_loaded:
            # Return default values
            return 0.5, 0.05, -0.03
        
        priors = self.parameters.get('priors', {}).get('global', {})
        return (
            priors.get('p_win', 0.5),
            priors.get('avg_win', 0.05),
            priors.get('avg_loss', -0.03)
        )
    
    def get_beta_prior(self) -> Tuple[float, float]:
        """Get beta distribution prior parameters.
        
        Returns:
            Tuple of (alpha, beta)
        """
        if not self.is_loaded:
            return 1.0, 1.0
        
        priors = self.parameters.get('priors', {}).get('global', {})
        return (
            priors.get('alpha', 1.0),
            priors.get('beta', 1.0)
        )
    
    def _normalize_setup_name(self, setup: str) -> List[str]:
        """Normalize setup names to match between scanner and trainer.
        
        Args:
            setup: Original setup name from scanner
            
        Returns:
            List of possible matching setup names to try
        """
        setup_lower = setup.lower().replace('_', '').replace(' ', '')
        
        # Mapping from scanner names to trainer names
        name_mappings = {
            'rsioversold': ['RSI_Oversold_Long'],
            'rsioverbought': ['RSI_Overbought_Short', 'Mean_Reversion_Short'],
            'macdbullish': ['MACD_Cross_Long'],
            'macdcross': ['MACD_Cross_Long'],
            'macdcrosslong': ['MACD_Cross_Long'],
            'bbbreakout': ['BB_Bounce_Long'],
            'bbbounce': ['BB_Bounce_Long'],
            'bbbouncelong': ['BB_Bounce_Long'],
            'stronguptrend': ['Trend_Following_Long'],
            'trendfollowing': ['Trend_Following_Long'],
            'trendfollowinglong': ['Trend_Following_Long'],
            'macrossover': ['Trend_Following_Long'],
            'meanreversion': ['Mean_Reversion_Short'],
            'meanreversionshort': ['Mean_Reversion_Short'],
            'volumemomentum': ['Trend_Following_Long'],
            'volatilitybreakout': ['Trend_Following_Long'],
            'stochoversold': ['RSI_Oversold_Long'],
            'stochoverbought': ['Mean_Reversion_Short'],
            'williamsoversold': ['RSI_Oversold_Long'],
            'williamsoverbought': ['Mean_Reversion_Short'],
            'ccioversold': ['RSI_Oversold_Long'],
            'ccioverbought': ['Mean_Reversion_Short'],
            'supportbounce': ['BB_Bounce_Long', 'RSI_Oversold_Long'],
            'resistancerejection': ['Mean_Reversion_Short'],
            'hammerpattern': ['RSI_Oversold_Long'],
            'engulfingpattern': ['Trend_Following_Long'],
            'dojipattern': ['RSI_Oversold_Long'],
        }
        
        # Try to find a match
        if setup_lower in name_mappings:
            return name_mappings[setup_lower]
        
        # Return original if no mapping found
        return [setup]
    
    def get_setup_parameters(self, setup: str) -> Optional[Dict[str, float]]:
        """Get parameters for a specific setup.
        
        Args:
            setup: Setup name
            
        Returns:
            Dictionary of parameters or None if not found
        """
        if not self.is_loaded:
            return None
        
        # Try exact match first
        params = self.parameters.get('setup_parameters', {}).get(setup)
        if params:
            return params
        
        # Try normalized names
        possible_names = self._normalize_setup_name(setup)
        for name in possible_names:
            params = self.parameters.get('setup_parameters', {}).get(name)
            if params:
                logger.debug(f"Mapped setup '{setup}' to trained setup '{name}'")
                return params
        
        return None
    
    def get_all_setups(self) -> list:
        """Get list of all trained setups."""
        if not self.is_loaded:
            return []
        
        return list(self.parameters.get('setup_parameters', {}).keys())


class EnhancedRollingBayes:
    """Enhanced RollingBayes with trained parameter integration.
    
    This extends the original RollingBayes to use trained parameters
    as informed priors when historical data is available.
    """
    
    def __init__(self, original_rolling_bayes, parameter_loader: TrainedParameterLoader):
        """Initialize enhanced Bayes model.
        
        Args:
            original_rolling_bayes: Original RollingBayes instance from scanner
            parameter_loader: Loaded trained parameters
        """
        self.rolling_bayes = original_rolling_bayes
        self.param_loader = parameter_loader
        
        # Update the alpha/beta priors if we have trained parameters
        if self.param_loader.is_loaded:
            alpha, beta = self.param_loader.get_beta_prior()
            self.rolling_bayes.alpha = alpha
            self.rolling_bayes.beta = beta
            logger.info(f"Updated priors: alpha={alpha:.2f}, beta={beta:.2f}")
    
    def estimate(self, setup: str, ticker: str, timeframe: str = '1d', 
                 volatility: float = None) -> Tuple[float, float, float]:
        """Estimate probability with trained parameter integration.
        
        This method:
        1. Checks if we have trained parameters for this setup
        2. Uses trained parameters as prior
        3. Combines with any real-time data from the portfolio
        
        Args:
            setup: Trading setup name
            ticker: Stock ticker
            timeframe: Timeframe
            volatility: Current volatility
            
        Returns:
            Tuple of (p_win, avg_win, avg_loss)
        """
        # Get original estimate from real trading data
        original_p, original_avg_win, original_avg_loss = self.rolling_bayes.estimate(
            setup, ticker, timeframe, volatility
        )
        
        # Get the number of real trades we have
        key = self.rolling_bayes._get_key(setup, ticker, timeframe)
        counter = self.rolling_bayes.counters[key]
        num_trades = counter['trades']
        
        # If we have enough real trading data, use it primarily
        if num_trades >= 30:
            # Mostly rely on real data
            return original_p, original_avg_win, original_avg_loss
        
        # Get trained parameters for this setup
        trained_params = self.param_loader.get_setup_parameters(setup)
        
        if not trained_params:
            # No trained params for this setup, use original
            return original_p, original_avg_win, original_avg_loss
        
        # Blend trained parameters with real data
        # Weight by confidence (more real trades = more weight on real data)
        confidence_real = min(num_trades / 30.0, 1.0)  # 0 to 1
        confidence_trained = 1.0 - confidence_real
        
        # Blend p_win
        trained_p = trained_params['p_win']
        blended_p = confidence_real * original_p + confidence_trained * trained_p
        
        # Blend average win
        trained_avg_win = trained_params['avg_win']
        blended_avg_win = confidence_real * original_avg_win + confidence_trained * trained_avg_win
        
        # Blend average loss
        trained_avg_loss = trained_params['avg_loss']
        blended_avg_loss = confidence_real * original_avg_loss + confidence_trained * trained_avg_loss
        
        logger.debug(f"Blended estimate for {setup} on {ticker}: "
                    f"p_win={blended_p:.3f} (trained={trained_p:.3f}, real={original_p:.3f}, "
                    f"weight_real={confidence_real:.2f})")
        
        return blended_p, blended_avg_win, blended_avg_loss
    
    def __getattr__(self, name):
        """Delegate all other methods to the original RollingBayes."""
        return getattr(self.rolling_bayes, name)


def integrate_trained_parameters(scanner_instance, parameters_path: str = "trained_parameters/scanner_parameters.json"):
    """Integrate trained parameters into a scanner instance.
    
    This function modifies the scanner's probability model to use
    trained parameters from historical backtesting.
    
    Args:
        scanner_instance: Instance of EVRScanner
        parameters_path: Path to trained parameters file
        
    Returns:
        bool: True if integration successful, False otherwise
    """
    try:
        # Load parameters
        param_loader = TrainedParameterLoader(parameters_path)
        
        if not param_loader.is_loaded:
            logger.warning("Could not load trained parameters, scanner will use defaults")
            return False
        
        # Wrap the existing prob_model with enhanced version
        enhanced_model = EnhancedRollingBayes(scanner_instance.prob_model, param_loader)
        scanner_instance.prob_model = enhanced_model
        
        logger.info("Successfully integrated trained parameters into scanner")
        
        # Log available setups
        setups = param_loader.get_all_setups()
        logger.info(f"Available trained setups: {', '.join(setups)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error integrating parameters: {e}")
        return False


class ParameterMonitor:
    """Monitor and track parameter usage and performance."""
    
    def __init__(self):
        """Initialize parameter monitor."""
        self.usage_stats = defaultdict(lambda: {
            'count': 0,
            'used_trained': 0,
            'used_real': 0,
            'avg_confidence': []
        })
    
    def record_usage(self, setup: str, used_trained: bool, confidence: float):
        """Record parameter usage.
        
        Args:
            setup: Setup name
            used_trained: Whether trained parameters were used
            confidence: Confidence in the estimate (0-1)
        """
        stats = self.usage_stats[setup]
        stats['count'] += 1
        
        if used_trained:
            stats['used_trained'] += 1
        else:
            stats['used_real'] += 1
        
        stats['avg_confidence'].append(confidence)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of parameter usage."""
        summary = {}
        
        for setup, stats in self.usage_stats.items():
            summary[setup] = {
                'total_uses': stats['count'],
                'trained_uses': stats['used_trained'],
                'real_uses': stats['used_real'],
                'avg_confidence': np.mean(stats['avg_confidence']) if stats['avg_confidence'] else 0.0,
                'trained_pct': stats['used_trained'] / stats['count'] if stats['count'] > 0 else 0.0
            }
        
        return summary
    
    def print_summary(self):
        """Print usage summary."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title="Parameter Usage Summary", show_header=True)
        
        table.add_column("Setup", style="cyan")
        table.add_column("Total Uses", justify="right")
        table.add_column("Trained %", justify="right")
        table.add_column("Real %", justify="right")
        table.add_column("Avg Confidence", justify="right")
        
        summary = self.get_summary()
        
        for setup, stats in summary.items():
            table.add_row(
                setup,
                str(stats['total_uses']),
                f"{stats['trained_pct']:.1%}",
                f"{(1-stats['trained_pct']):.1%}",
                f"{stats['avg_confidence']:.2f}"
            )
        
        console.print(table)


# Example usage function
def example_integration():
    """Example of how to integrate parameters into scanner."""
    # This would be called after creating a scanner instance
    # in the official_scanner.py
    
    print("Example integration:")
    print("1. Train parameters: python historical_parameter_trainer.py")
    print("2. In your scanner code, add:")
    print("""
    from parameter_integration import integrate_trained_parameters
    
    # After creating scanner
    scanner = EVRScanner(...)
    
    # Integrate trained parameters
    integrate_trained_parameters(scanner)
    
    # Now scanner will use trained parameters!
    """)


if __name__ == "__main__":
    example_integration()


