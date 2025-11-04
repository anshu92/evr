#!/usr/bin/env python3
"""
Test Script for Historical Parameter Training System

Quick tests to verify the system is working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - run: pip install pandas")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - run: pip install numpy")
        return False
    
    try:
        import yfinance
        print("  ✓ yfinance")
    except ImportError:
        print("  ✗ yfinance - run: pip install yfinance")
        return False
    
    try:
        from rich.console import Console
        print("  ✓ rich")
    except ImportError:
        print("  ✗ rich - run: pip install rich")
        return False
    
    try:
        import sklearn
        print("  ✓ scikit-learn (optional)")
    except ImportError:
        print("  ⚠ scikit-learn (optional) - run: pip install scikit-learn")
    
    return True


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        import historical_parameter_trainer
        print("  ✓ historical_parameter_trainer")
    except ImportError as e:
        print(f"  ✗ historical_parameter_trainer - {e}")
        return False
    
    try:
        import parameter_integration
        print("  ✓ parameter_integration")
    except ImportError as e:
        print(f"  ✗ parameter_integration - {e}")
        return False
    
    try:
        import run_parameter_training
        print("  ✓ run_parameter_training")
    except ImportError as e:
        print(f"  ✗ run_parameter_training - {e}")
        return False
    
    return True


def test_classes():
    """Test that main classes can be instantiated."""
    print("\nTesting class instantiation...")
    
    try:
        from historical_parameter_trainer import (
            TechnicalIndicators,
            SignalGenerator,
            TradeSimulator,
            ParameterEstimator
        )
        
        # Test instantiation
        indicators = TechnicalIndicators()
        print("  ✓ TechnicalIndicators")
        
        generator = SignalGenerator()
        print("  ✓ SignalGenerator")
        
        simulator = TradeSimulator()
        print("  ✓ TradeSimulator")
        
        estimator = ParameterEstimator()
        print("  ✓ ParameterEstimator")
        
    except Exception as e:
        print(f"  ✗ Class instantiation failed - {e}")
        return False
    
    try:
        from parameter_integration import TrainedParameterLoader
        
        loader = TrainedParameterLoader("nonexistent.json")
        print("  ✓ TrainedParameterLoader")
        
    except Exception as e:
        print(f"  ✗ TrainedParameterLoader failed - {e}")
        return False
    
    return True


def test_data_fetch():
    """Test that we can fetch a single ticker's data."""
    print("\nTesting data fetch...")
    
    try:
        import yfinance as yf
        
        # Fetch a single ticker
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo")
        
        if not data.empty:
            print(f"  ✓ Fetched {len(data)} bars for AAPL")
        else:
            print("  ✗ Empty data returned")
            return False
            
    except Exception as e:
        print(f"  ✗ Data fetch failed - {e}")
        return False
    
    return True


def test_indicator_calculation():
    """Test that indicators can be calculated."""
    print("\nTesting indicator calculation...")
    
    try:
        import yfinance as yf
        import pandas as pd
        from historical_parameter_trainer import TechnicalIndicators
        
        # Fetch data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="3mo")
        
        if data.empty:
            print("  ✗ No data to test indicators")
            return False
        
        # Calculate indicators
        indicators = TechnicalIndicators()
        
        rsi = indicators.calculate_rsi(data)
        print(f"  ✓ RSI (last value: {rsi.iloc[-1]:.2f})")
        
        macd, signal, hist = indicators.calculate_macd(data)
        print(f"  ✓ MACD (last value: {macd.iloc[-1]:.4f})")
        
        upper, middle, lower = indicators.calculate_bollinger_bands(data)
        print(f"  ✓ Bollinger Bands (last: {lower.iloc[-1]:.2f} - {upper.iloc[-1]:.2f})")
        
        atr = indicators.calculate_atr(data)
        print(f"  ✓ ATR (last value: {atr.iloc[-1]:.2f})")
        
    except Exception as e:
        print(f"  ✗ Indicator calculation failed - {e}")
        return False
    
    return True


def test_signal_generation():
    """Test that signals can be generated."""
    print("\nTesting signal generation...")
    
    try:
        import yfinance as yf
        from historical_parameter_trainer import SignalGenerator
        
        # Fetch data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1y")
        
        if len(data) < 200:
            print("  ✗ Insufficient data for signal generation")
            return False
        
        # Generate signals
        generator = SignalGenerator()
        signals = generator.generate_signals(data, "AAPL")
        
        print(f"  ✓ Generated {len(signals)} signals for AAPL")
        
        if len(signals) > 0:
            setup_counts = {}
            for signal in signals:
                setup = signal['setup']
                setup_counts[setup] = setup_counts.get(setup, 0) + 1
            
            for setup, count in setup_counts.items():
                print(f"    • {setup}: {count} signals")
        
    except Exception as e:
        print(f"  ✗ Signal generation failed - {e}")
        return False
    
    return True


def test_trade_simulation():
    """Test that a trade can be simulated."""
    print("\nTesting trade simulation...")
    
    try:
        import yfinance as yf
        from historical_parameter_trainer import SignalGenerator, TradeSimulator
        from datetime import datetime
        
        # Fetch data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1y")
        
        # Generate signals
        generator = SignalGenerator()
        signals = generator.generate_signals(data, "AAPL")
        
        if len(signals) == 0:
            print("  ⚠ No signals to test trade simulation")
            return True
        
        # Simulate first signal
        signal = signals[0]
        simulator = TradeSimulator()
        
        # Get future data
        signal_date = signal['date']
        future_data = data[data.index > signal_date]
        
        if future_data.empty:
            print("  ⚠ No future data for simulation")
            return True
        
        # Simulate trade
        result = simulator.simulate_trade(signal, future_data)
        
        if result:
            print(f"  ✓ Simulated trade: {result.setup}")
            print(f"    • Entry: ${result.entry_price:.2f}")
            print(f"    • Exit: ${result.exit_price:.2f}")
            print(f"    • Return: {result.return_pct:.2%}")
            print(f"    • Result: {'WIN' if result.is_win else 'LOSS'}")
            print(f"    • Exit Reason: {result.exit_reason}")
        else:
            print("  ✗ Trade simulation returned None")
            return False
        
    except Exception as e:
        print(f"  ✗ Trade simulation failed - {e}")
        return False
    
    return True


def test_parameter_estimation():
    """Test that parameters can be estimated."""
    print("\nTesting parameter estimation...")
    
    try:
        from historical_parameter_trainer import ParameterEstimator, TradeResult
        from datetime import datetime
        
        # Create mock trades
        trades = [
            TradeResult(
                ticker="AAPL",
                setup="RSI_Oversold_Long",
                entry_date=datetime.now(),
                entry_price=150.0,
                exit_date=datetime.now(),
                exit_price=155.0,
                direction=1,
                return_pct=0.0333,
                r_multiple=1.67,
                is_win=True,
                stop_price=147.0,
                target_price=156.0,
                exit_reason="TARGET"
            ),
            TradeResult(
                ticker="AAPL",
                setup="RSI_Oversold_Long",
                entry_date=datetime.now(),
                entry_price=150.0,
                exit_date=datetime.now(),
                exit_price=147.0,
                direction=1,
                return_pct=-0.02,
                r_multiple=-1.0,
                is_win=False,
                stop_price=147.0,
                target_price=156.0,
                exit_reason="STOP"
            )
        ]
        
        # Estimate parameters
        estimator = ParameterEstimator()
        stats = estimator.calculate_statistics(trades, by_setup=True, by_ticker=False)
        
        for (setup, ticker), stat in stats.items():
            print(f"  ✓ Estimated parameters for {setup}")
            print(f"    • P(Win): {stat.p_win:.2%}")
            print(f"    • Avg Win: {stat.avg_win:.2%}")
            print(f"    • Avg Loss: {stat.avg_loss:.2%}")
            print(f"    • Expectancy: {stat.expectancy:.2%}")
        
    except Exception as e:
        print(f"  ✗ Parameter estimation failed - {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Historical Parameter Training System - Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Modules", test_modules()))
    results.append(("Classes", test_classes()))
    results.append(("Data Fetch", test_data_fetch()))
    results.append(("Indicators", test_indicator_calculation()))
    results.append(("Signals", test_signal_generation()))
    results.append(("Simulation", test_trade_simulation()))
    results.append(("Estimation", test_parameter_estimation()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python run_parameter_training.py")
        print("2. Run scanner: python official_scanner.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before proceeding.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements_training.txt")
        print("2. Check internet connection (for data fetch)")
        print("3. Verify all files are present")
        return 1


if __name__ == "__main__":
    sys.exit(main())


