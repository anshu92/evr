"""Vectorized backtesting engine."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ...config import Config
from ...data import DataAdapter, YFinanceAdapter, DataCache, ParquetCache
from ...features import FeatureGraph
from ...models import RollingBayes, PayoffModel
from ...risk import KellySizing, RiskGuards
from ...setups import SetupRegistry
from ...types import Bars, Features, Signal, TradePlan, TradeResult, Metrics


class BacktestEngine:
    """Vectorized backtesting engine."""
    
    def __init__(self, config: Config):
        """Initialize backtest engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.run_id = str(uuid.uuid4())
        
        # Initialize components
        self.data_adapter = YFinanceAdapter()
        self.cache = ParquetCache(
            cache_dir=config.data.cache_dir,
            ttl_days=config.data.cache_ttl_days
        )
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
        self._trade_results: List[TradeResult] = []
        self._current_positions: Dict[str, TradePlan] = {}
        self._equity_curve: List[float] = []
        self._timestamps: List[pd.Timestamp] = []
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        setups: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Run backtest.
        
        Args:
            symbols: List of symbols to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            setups: List of setup names to use
            
        Returns:
            Dictionary with backtest results
        """
        if setups is None:
            setups = self.setup_registry.list_setups()
        
        # Load data
        self._load_data(symbols, start_date, end_date)
        
        # Compute features
        self._compute_features()
        
        # Run backtest
        self._run_backtest_loop(setups)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            'run_id': self.run_id,
            'config': self.config.to_dict(),
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'setups': setups,
            'trade_results': self._trade_results,
            'equity_curve': self._equity_curve,
            'timestamps': self._timestamps,
            'metrics': metrics,
            'current_positions': self._current_positions,
        }
    
    def _load_data(self, symbols: List[str], start_date: str, end_date: str) -> None:
        """Load data for all symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
        """
        for symbol in symbols:
            # Check cache first
            if self.cache.exists(symbol, start_date, end_date):
                bars = self.cache.get(symbol, start_date, end_date)
            else:
                # Download from data source
                bars = self.data_adapter.get_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=self.config.data.timeframe,
                    adjust_splits=self.config.data.adjust_splits,
                    adjust_dividends=self.config.data.adjust_dividends,
                )
                
                # Cache the data
                self.cache.put(symbol, bars, start_date, end_date)
            
            self._bars[symbol] = bars
    
    def _compute_features(self) -> None:
        """Compute features for all symbols."""
        self.feature_graph.add_technical_indicators()
        
        for symbol, bars in self._bars.items():
            self.feature_graph.set_data(bars)
            features = self.feature_graph.compute_all()
            self._features[symbol] = features
    
    def _run_backtest_loop(self, setups: List[str]) -> None:
        """Run the main backtest loop.
        
        Args:
            setups: List of setup names
        """
        # Get all timestamps
        all_timestamps = set()
        for bars in self._bars.values():
            all_timestamps.update(bars.index)
        
        timestamps = sorted(all_timestamps)
        
        for timestamp in timestamps:
            self._process_timestamp(timestamp, setups)
            self._update_equity_curve()
    
    def _process_timestamp(self, timestamp: pd.Timestamp, setups: List[str]) -> None:
        """Process a single timestamp.
        
        Args:
            timestamp: Current timestamp
            setups: List of setup names
        """
        # Check for exits first
        self._check_exits(timestamp)
        
        # Generate new signals
        self._generate_signals(timestamp, setups)
        
        # Process signals into trade plans
        self._process_signals(timestamp)
        
        # Update probability model
        self._update_probability_model(timestamp)
    
    def _check_exits(self, timestamp: pd.Timestamp) -> None:
        """Check for position exits.
        
        Args:
            timestamp: Current timestamp
        """
        exits = []
        
        for symbol, trade_plan in self._current_positions.items():
            if symbol not in self._bars:
                continue
            
            bars = self._bars[symbol]
            if timestamp not in bars.index:
                continue
            
            current_bar = bars.loc[timestamp]
            exit_price = None
            exit_reason = None
            
            # Check stop loss
            if trade_plan.signal.direction > 0:  # Long position
                if current_bar['Low'] <= trade_plan.stop_loss:
                    exit_price = trade_plan.stop_loss
                    exit_reason = 'stop_loss'
                elif current_bar['High'] >= trade_plan.take_profit:
                    exit_price = trade_plan.take_profit
                    exit_reason = 'take_profit'
            else:  # Short position
                if current_bar['High'] >= trade_plan.stop_loss:
                    exit_price = trade_plan.stop_loss
                    exit_reason = 'stop_loss'
                elif current_bar['Low'] <= trade_plan.take_profit:
                    exit_price = trade_plan.take_profit
                    exit_reason = 'take_profit'
            
            if exit_price is not None:
                exits.append((symbol, trade_plan, exit_price, exit_reason))
        
        # Execute exits
        for symbol, trade_plan, exit_price, exit_reason in exits:
            self._execute_exit(symbol, trade_plan, exit_price, exit_reason, timestamp)
    
    def _execute_exit(
        self,
        symbol: str,
        trade_plan: TradePlan,
        exit_price: float,
        exit_reason: str,
        timestamp: pd.Timestamp,
    ) -> None:
        """Execute position exit.
        
        Args:
            symbol: Stock symbol
            trade_plan: Trade plan
            exit_price: Exit price
            exit_reason: Exit reason
            timestamp: Exit timestamp
        """
        # Calculate P&L
        if trade_plan.signal.direction > 0:  # Long position
            pnl = (exit_price - trade_plan.entry_price) * trade_plan.position_size
        else:  # Short position
            pnl = (trade_plan.entry_price - exit_price) * trade_plan.position_size
        
        # Calculate returns
        returns = pnl / trade_plan.position_size
        
        # Calculate duration
        duration = (timestamp - trade_plan.signal.timestamp).total_seconds() / (24 * 3600)
        
        # Create trade result
        trade_result = TradeResult(
            symbol=symbol,
            entry_timestamp=trade_plan.signal.timestamp,
            exit_timestamp=timestamp,
            direction=trade_plan.signal.direction,
            entry_price=trade_plan.entry_price,
            exit_price=exit_price,
            quantity=trade_plan.position_size,
            pnl=pnl,
            returns=returns,
            duration_days=duration,
            setup=trade_plan.signal.setup,
            metadata={'exit_reason': exit_reason}
        )
        
        # Add to results
        self._trade_results.append(trade_result)
        
        # Update risk guards
        self.risk_guards.add_trade_result(trade_result)
        
        # Remove from current positions
        del self._current_positions[symbol]
    
    def _generate_signals(self, timestamp: pd.Timestamp, setups: List[str]) -> None:
        """Generate signals for current timestamp.
        
        Args:
            timestamp: Current timestamp
            setups: List of setup names
        """
        for symbol, bars in self._bars.items():
            if timestamp not in bars.index:
                continue
            
            if symbol not in self._features:
                continue
            
            features = self._features[symbol]
            
            for setup_name in setups:
                try:
                    setup = self.setup_registry.get_setup(setup_name)
                    signals = setup.signals(bars, features, symbol, timestamp)
                    
                    for signal in signals:
                        self._signals.append(signal)
                        
                except Exception as e:
                    print(f"Warning: Failed to generate signals for {symbol} with {setup_name}: {e}")
                    continue
    
    def _process_signals(self, timestamp: pd.Timestamp) -> None:
        """Process signals into trade plans.
        
        Args:
            timestamp: Current timestamp
        """
        for signal in self._signals:
            if signal.timestamp != timestamp:
                continue
            
            # Skip if already in position
            if signal.symbol in self._current_positions:
                continue
            
            # Check risk guards
            if not self._check_risk_guards(signal):
                continue
            
            # Create trade plan
            trade_plan = self._create_trade_plan(signal, timestamp)
            if trade_plan is None:
                continue
            
            # Add to current positions
            self._current_positions[signal.symbol] = trade_plan
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
            
            # Create trade plan
            trade_plan = TradePlan(
                signal=signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_per_trade=0.02,  # 2% risk per trade
                expected_return=win_prob * avg_win - (1 - win_prob) * avg_loss,
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
    
    def _update_probability_model(self, timestamp: pd.Timestamp) -> None:
        """Update probability model with recent trades.
        
        Args:
            timestamp: Current timestamp
        """
        # Update with trades from the last day
        recent_trades = [
            t for t in self._trade_results
            if (timestamp - t.exit_timestamp).total_seconds() < 24 * 3600
        ]
        
        for trade in recent_trades:
            self.prob_model.update(trade)
    
    def _update_equity_curve(self) -> None:
        """Update equity curve."""
        current_capital = self.risk_guards._current_capital
        self._equity_curve.append(current_capital)
    
    def _calculate_metrics(self) -> Metrics:
        """Calculate backtest metrics.
        
        Returns:
            Metrics object
        """
        if not self._trade_results:
            return self._create_empty_metrics()
        
        # Calculate basic metrics
        total_trades = len(self._trade_results)
        winning_trades = [t for t in self._trade_results if t.is_winner]
        losing_trades = [t for t in self._trade_results if t.is_loser]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean([t.returns for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t.returns for t in losing_trades])) if losing_trades else 0.0
        
        # Calculate total return and CAGR
        initial_capital = self.config.risk.initial_capital
        final_capital = self._equity_curve[-1] if self._equity_curve else initial_capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate CAGR
        if self._timestamps:
            start_date = self._timestamps[0]
            end_date = self._timestamps[-1]
            years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
            cagr = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
        else:
            cagr = 0.0
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate Sharpe ratio
        if self._equity_curve:
            returns = pd.Series(self._equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate other metrics
        profit_factor = self._calculate_profit_factor()
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # Calculate turnover
        turnover = self._calculate_turnover()
        
        # Calculate average trade duration
        avg_duration = np.mean([t.duration_days for t in self._trade_results]) if self._trade_results else 0.0
        
        # Calculate best and worst months
        best_month, worst_month = self._calculate_monthly_returns()
        
        # Calculate VaR and Expected Shortfall
        var_95, expected_shortfall = self._calculate_risk_metrics()
        
        # Calculate consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_trades()
        
        return Metrics(
            total_return=total_return,
            cagr=cagr,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0.0,  # Placeholder
            calmar_ratio=cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max([t.returns for t in winning_trades]) if winning_trades else 0.0,
            largest_loss=min([t.returns for t in losing_trades]) if losing_trades else 0.0,
            profit_factor=profit_factor,
            expectancy=expectancy,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            turnover=turnover,
            avg_trade_duration=avg_duration,
            best_month=best_month,
            worst_month=worst_month,
            start_date=self._timestamps[0] if self._timestamps else pd.Timestamp.now(),
            end_date=self._timestamps[-1] if self._timestamps else pd.Timestamp.now(),
        )
    
    def _create_empty_metrics(self) -> Metrics:
        """Create empty metrics.
        
        Returns:
            Empty metrics object
        """
        return Metrics(
            total_return=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            turnover=0.0,
            avg_trade_duration=0.0,
            best_month=0.0,
            worst_month=0.0,
            start_date=pd.Timestamp.now(),
            end_date=pd.Timestamp.now(),
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown
        """
        if not self._equity_curve:
            return 0.0
        
        equity = pd.Series(self._equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor.
        
        Returns:
            Profit factor
        """
        if not self._trade_results:
            return 0.0
        
        total_wins = sum(t.pnl for t in self._trade_results if t.is_winner)
        total_losses = abs(sum(t.pnl for t in self._trade_results if t.is_loser))
        
        return total_wins / total_losses if total_losses > 0 else 0.0
    
    def _calculate_turnover(self) -> float:
        """Calculate turnover.
        
        Returns:
            Turnover
        """
        if not self._trade_results:
            return 0.0
        
        total_volume = sum(abs(t.pnl) for t in self._trade_results)
        initial_capital = self.config.risk.initial_capital
        
        return total_volume / initial_capital if initial_capital > 0 else 0.0
    
    def _calculate_monthly_returns(self) -> tuple[float, float]:
        """Calculate best and worst monthly returns.
        
        Returns:
            Tuple of (best_month, worst_month)
        """
        if not self._equity_curve or not self._timestamps:
            return 0.0, 0.0
        
        # Group by month and calculate returns
        df = pd.DataFrame({
            'equity': self._equity_curve,
            'timestamp': self._timestamps
        })
        df.set_index('timestamp', inplace=True)
        
        monthly_returns = df['equity'].resample('M').last().pct_change().dropna()
        
        if monthly_returns.empty:
            return 0.0, 0.0
        
        return monthly_returns.max(), monthly_returns.min()
    
    def _calculate_risk_metrics(self) -> tuple[float, float]:
        """Calculate VaR and Expected Shortfall.
        
        Returns:
            Tuple of (VaR_95, Expected_Shortfall)
        """
        if not self._trade_results:
            return 0.0, 0.0
        
        returns = [t.returns for t in self._trade_results]
        returns_series = pd.Series(returns)
        
        var_95 = returns_series.quantile(0.05)
        expected_shortfall = returns_series[returns_series <= var_95].mean()
        
        return abs(var_95), abs(expected_shortfall)
    
    def _calculate_consecutive_trades(self) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses.
        
        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not self._trade_results:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self._trade_results:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _register_builtin_setups(self) -> None:
        """Register built-in setups."""
        from ...setups.builtin import SqueezeBreakout, TrendPullback, MeanReversion
        
        self.setup_registry.register('squeeze_breakout', SqueezeBreakout)
        self.setup_registry.register('trend_pullback', TrendPullback)
        self.setup_registry.register('mean_reversion', MeanReversion)
