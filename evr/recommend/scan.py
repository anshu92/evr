"""Recommendation scanner for EVR."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ...config import Config
from ...data import DataAdapter, YFinanceAdapter, DataCache, ParquetCache, UniverseLoader
from ...features import FeatureGraph
from ...models import RollingBayes, PayoffModel
from ...risk import KellySizing, RiskGuards
from ...setups import SetupRegistry
from ...types import Bars, Features, Signal, TradePlan


class RecommendationScanner:
    """Recommendation scanner for generating trade recommendations."""
    
    def __init__(self, config: Config):
        """Initialize recommendation scanner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.data_adapter = YFinanceAdapter()
        self.cache = ParquetCache(
            cache_dir=config.data.cache_dir,
            ttl_days=config.data.cache_ttl_days
        )
        self.universe_loader = UniverseLoader(config.data.universe_file)
        self.feature_graph = FeatureGraph()
        self.prob_model = RollingBayes(
            window_size=config.prob.window_size,
            min_samples=config.prob.min_samples,
            alpha=config.prob.alpha,
            beta_prior_alpha=config.prob.beta_prior_alpha,
            beta_prior_beta=config.prob.beta_prior_beta,
            use_regimes=config.prob.use_regimes,
            regime_window=config.prob.regime_window,
            regime_threshold=config.prob.regime_threshold,
        )
        self.payoff_model = PayoffModel(
            commission_per_trade=config.risk.commission_per_trade,
            slippage_bps=config.risk.slippage_bps,
            slippage_atr_multiplier=config.risk.slippage_atr_multiplier,
        )
        self.kelly_sizing = KellySizing(
            kelly_fraction=config.risk.kelly_fraction,
            max_kelly_fraction=config.risk.max_kelly_fraction,
            max_position_size=config.risk.max_position_size,
            min_position_size=config.risk.min_position_size,
        )
        self.risk_guards = RiskGuards(
            initial_capital=config.risk.initial_capital,
            max_position_size=config.risk.max_position_size,
            max_sector_exposure=config.risk.max_sector_exposure,
            max_correlation=config.risk.max_correlation,
            daily_loss_limit=config.risk.daily_loss_limit,
            weekly_loss_limit=config.risk.weekly_loss_limit,
            max_drawdown_limit=config.risk.max_drawdown_limit,
            max_positions=config.risk.max_positions,
            min_positions=config.risk.min_positions,
        )
        
        # Setup registry
        self.setup_registry = SetupRegistry()
        self._register_builtin_setups()
        
        # Internal state
        self._bars: Dict[str, Bars] = {}
        self._features: Dict[str, Features] = {}
        self._signals: List[Signal] = []
        self._trade_plans: List[TradePlan] = []
    
    def scan(
        self,
        symbols: Optional[List[str]] = None,
        setups: Optional[List[str]] = None,
        asof_date: Optional[str] = None,
        top_n: int = 10,
    ) -> List[TradePlan]:
        """Scan for trading recommendations.
        
        Args:
            symbols: List of symbols to scan (if None, uses universe)
            setups: List of setup names to use (if None, uses all)
            asof_date: As-of date for scanning (if None, uses today)
            top_n: Number of top recommendations to return
            
        Returns:
            List of top trade plans
        """
        # Get symbols to scan
        if symbols is None:
            symbols = self.universe_loader.get_symbols()
        
        # Get setups to use
        if setups is None:
            setups = self.setup_registry.list_setups()
        
        # Set asof date
        if asof_date is None:
            asof_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Load data
        self._load_data(symbols, asof_date)
        
        # Compute features
        self._compute_features()
        
        # Generate signals
        self._generate_signals(setups, asof_date)
        
        # Process signals into trade plans
        self._process_signals(asof_date)
        
        # Rank and filter trade plans
        ranked_plans = self._rank_trade_plans()
        
        return ranked_plans[:top_n]
    
    def _load_data(self, symbols: List[str], asof_date: str) -> None:
        """Load data for all symbols.
        
        Args:
            symbols: List of symbols
            asof_date: As-of date
        """
        # Calculate start date (enough data for features)
        start_date = pd.Timestamp(asof_date) - pd.Timedelta(days=365)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                # Check cache first
                if self.cache.exists(symbol, start_date_str, asof_date):
                    bars = self.cache.get(symbol, start_date_str, asof_date)
                else:
                    # Download from data source
                    bars = self.data_adapter.get_bars(
                        symbol=symbol,
                        start_date=start_date_str,
                        end_date=asof_date,
                        timeframe=self.config.data.timeframe,
                        adjust_splits=self.config.data.adjust_splits,
                        adjust_dividends=self.config.data.adjust_dividends,
                    )
                    
                    # Cache the data
                    self.cache.put(symbol, bars, start_date_str, asof_date)
                
                # Only keep data if we have enough for features
                if len(bars) >= 50:  # Minimum bars for technical indicators
                    self._bars[symbol] = bars
                    
            except Exception as e:
                print(f"Warning: Failed to load data for {symbol}: {e}")
                continue
    
    def _compute_features(self) -> None:
        """Compute features for all symbols."""
        self.feature_graph.add_technical_indicators()
        
        for symbol, bars in self._bars.items():
            try:
                self.feature_graph.set_data(bars)
                features = self.feature_graph.compute_all()
                self._features[symbol] = features
            except Exception as e:
                print(f"Warning: Failed to compute features for {symbol}: {e}")
                continue
    
    def _generate_signals(self, setups: List[str], asof_date: str) -> None:
        """Generate signals for all symbols and setups.
        
        Args:
            setups: List of setup names
            asof_date: As-of date
        """
        asof_timestamp = pd.Timestamp(asof_date)
        
        for symbol, bars in self._bars.items():
            if symbol not in self._features:
                continue
            
            if asof_timestamp not in bars.index:
                continue
            
            features = self._features[symbol]
            
            for setup_name in setups:
                try:
                    setup = self.setup_registry.get_setup(setup_name)
                    signals = setup.signals(bars, features, symbol, asof_timestamp)
                    
                    for signal in signals:
                        self._signals.append(signal)
                        
                except Exception as e:
                    print(f"Warning: Failed to generate signals for {symbol} with {setup_name}: {e}")
                    continue
    
    def _process_signals(self, asof_date: str) -> None:
        """Process signals into trade plans.
        
        Args:
            asof_date: As-of date
        """
        asof_timestamp = pd.Timestamp(asof_date)
        
        for signal in self._signals:
            # Check risk guards
            if not self._check_risk_guards(signal):
                continue
            
            # Create trade plan
            trade_plan = self._create_trade_plan(signal, asof_timestamp)
            if trade_plan is None:
                continue
            
            self._trade_plans.append(trade_plan)
    
    def _check_risk_guards(self, signal: Signal) -> bool:
        """Check if signal passes risk guards.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if signal passes risk guards
        """
        # Create dummy trade plan for risk checking
        dummy_plan = TradePlan(
            signal=signal,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            position_size=0.0,
            risk_per_trade=0.0,
            expected_return=0.0,
            probability=0.0,
        )
        
        checks = self.risk_guards.check_trade_plan(dummy_plan)
        return checks['all_checks_passed']
    
    def _create_trade_plan(self, signal: Signal, timestamp: pd.Timestamp) -> Optional[TradePlan]:
        """Create trade plan from signal.
        
        Args:
            signal: Trading signal
            timestamp: Current timestamp
            
        Returns:
            Trade plan or None if creation fails
        """
        try:
            bars = self._bars[signal.symbol]
            current_bar = bars.loc[timestamp]
            
            # Get probability estimates
            win_prob, avg_win, avg_loss = self.prob_model.estimate(signal.setup)
            
            # Calculate entry price
            entry_price = current_bar['Close']
            
            # Calculate stop loss and take profit
            atr = self._get_atr(bars, timestamp)
            stop_distance = atr * 2.0  # 2x ATR stop
            
            if signal.direction > 0:  # Long position
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 2.0)  # 2:1 risk-reward
            else:  # Short position
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 2.0)  # 2:1 risk-reward
            
            # Calculate position size
            sizing_info = self.kelly_sizing.size_position(
                trade_plan=None,  # Will be created below
                portfolio_value=self.risk_guards._current_capital,
                win_probability=win_prob,
                avg_win_return=avg_win,
                avg_loss_return=avg_loss,
                current_positions=self.risk_guards._current_positions,
            )
            
            position_size = sizing_info['position_size']
            
            # Calculate expected return
            expected_return = win_prob * avg_win - (1 - win_prob) * avg_loss
            
            # Create trade plan
            trade_plan = TradePlan(
                signal=signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_per_trade=0.02,  # 2% risk per trade
                expected_return=expected_return,
                probability=win_prob,
            )
            
            return trade_plan
            
        except Exception as e:
            print(f"Warning: Failed to create trade plan for {signal.symbol}: {e}")
            return None
    
    def _get_atr(self, bars: Bars, timestamp: pd.Timestamp) -> float:
        """Get ATR value.
        
        Args:
            bars: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            ATR value
        """
        try:
            # Calculate ATR manually
            high = bars['High']
            low = bars['Low']
            close = bars['Close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR as 14-period average
            atr = true_range.rolling(14).mean()
            
            return atr.loc[timestamp]
            
        except (KeyError, IndexError):
            # Fallback to simple range
            current_bar = bars.loc[timestamp]
            return current_bar['High'] - current_bar['Low']
    
    def _rank_trade_plans(self) -> List[TradePlan]:
        """Rank trade plans by expected growth proxy.
        
        Returns:
            List of ranked trade plans
        """
        if not self._trade_plans:
            return []
        
        # Calculate ranking score for each trade plan
        scored_plans = []
        
        for plan in self._trade_plans:
            # Expected growth proxy = probability * expected_return * signal_strength
            score = (
                plan.probability *
                plan.expected_return *
                plan.signal.strength *
                plan.risk_reward_ratio
            )
            
            scored_plans.append((score, plan))
        
        # Sort by score (descending)
        scored_plans.sort(key=lambda x: x[0], reverse=True)
        
        # Return sorted trade plans
        return [plan for score, plan in scored_plans]
    
    def get_scan_summary(self) -> Dict[str, any]:
        """Get scan summary.
        
        Returns:
            Dictionary with scan summary
        """
        return {
            'total_symbols_scanned': len(self._bars),
            'total_signals_generated': len(self._signals),
            'total_trade_plans': len(self._trade_plans),
            'symbols_with_data': list(self._bars.keys()),
            'symbols_with_features': list(self._features.keys()),
            'signals_by_setup': self._count_signals_by_setup(),
            'signals_by_symbol': self._count_signals_by_symbol(),
        }
    
    def _count_signals_by_setup(self) -> Dict[str, int]:
        """Count signals by setup.
        
        Returns:
            Dictionary with signal counts by setup
        """
        counts = {}
        for signal in self._signals:
            setup = signal.setup
            counts[setup] = counts.get(setup, 0) + 1
        return counts
    
    def _count_signals_by_symbol(self) -> Dict[str, int]:
        """Count signals by symbol.
        
        Returns:
            Dictionary with signal counts by symbol
        """
        counts = {}
        for signal in self._signals:
            symbol = signal.symbol
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts
    
    def _register_builtin_setups(self) -> None:
        """Register built-in setups."""
        from ...setups.builtin import SqueezeBreakout, TrendPullback, MeanReversion
        
        self.setup_registry.register('squeeze_breakout', SqueezeBreakout)
        self.setup_registry.register('trend_pullback', TrendPullback)
        self.setup_registry.register('mean_reversion', MeanReversion)
