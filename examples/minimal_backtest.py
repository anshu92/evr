#!/usr/bin/env python3
"""Minimal backtest example demonstrating EVR usage."""

from evr.config import load_config
from evr.backtest import BacktestEngine


def main():
    """Run a minimal backtest example."""
    # Load configuration
    config = load_config()
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest
    print("Running backtest...")
    results = engine.run_backtest(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2020-01-01",
        end_date="2025-01-01",
        setups=["squeeze_breakout", "trend_pullback", "mean_reversion"],
    )
    
    # Display results
    metrics = results.get('metrics', {})
    
    print(f"\nBacktest Results:")
    print("-" * 40)
    print(f"Run ID: {results['run_id']}")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Symbols: {', '.join(results['symbols'])}")
    print(f"Setups: {', '.join(results['setups'])}")
    print()
    
    print("Performance Metrics:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"CAGR: {metrics.cagr:.2%}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print()
    
    print("Trade Statistics:")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Winning Trades: {metrics.winning_trades}")
    print(f"Losing Trades: {metrics.losing_trades}")
    print(f"Average Win: {metrics.avg_win:.2%}")
    print(f"Average Loss: {metrics.avg_loss:.2%}")
    print(f"Largest Win: {metrics.largest_win:.2%}")
    print(f"Largest Loss: {metrics.largest_loss:.2%}")
    print()
    
    print("Risk Metrics:")
    print(f"VaR (95%): {metrics.var_95:.2%}")
    print(f"Expected Shortfall: {metrics.expected_shortfall:.2%}")
    print(f"Max Consecutive Wins: {metrics.max_consecutive_wins}")
    print(f"Max Consecutive Losses: {metrics.max_consecutive_losses}")
    print()
    
    print("Additional Metrics:")
    print(f"Turnover: {metrics.turnover:.2f}")
    print(f"Average Trade Duration: {metrics.avg_trade_duration:.1f} days")
    print(f"Best Month: {metrics.best_month:.2%}")
    print(f"Worst Month: {metrics.worst_month:.2%}")


if __name__ == "__main__":
    main()
