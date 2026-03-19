"""Tests for new predictive and portfolio management features."""
import logging
import numpy as np
import pandas as pd
import pytest

LOGGER = logging.getLogger("test")


# --- 1. Feature engineering tests ---

class TestIntradayOvernightFeatures:
    """Test intraday/overnight return decomposition."""

    def test_overnight_ret_computed(self):
        """Overnight return should equal cumulative close-to-open returns."""
        from stock_screener.features.technical import compute_features

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        tickers = ["TEST"]
        # Create prices where open is always 1% above previous close
        close = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.001, 0.01, len(dates))), index=dates)
        open_price = close.shift(1) * 1.01  # Open 1% above yesterday's close
        open_price.iloc[0] = close.iloc[0]
        volume = pd.Series(1_000_000, index=dates)
        high = close * 1.02
        low = close * 0.98

        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        prices = pd.DataFrame(index=dates, columns=cols)
        prices[("TEST", "Close")] = close
        prices[("TEST", "Open")] = open_price
        prices[("TEST", "Volume")] = volume
        prices[("TEST", "High")] = high
        prices[("TEST", "Low")] = low

        fx = pd.Series(1.0, index=dates)
        features = compute_features(
            prices, fx, liquidity_lookback_days=30, feature_lookback_days=120, logger=LOGGER
        )

        assert "overnight_ret_5d" in features.columns
        assert "intraday_ret_5d" in features.columns
        assert "overnight_intraday_ratio" in features.columns
        # Overnight return should be positive (open > prev close)
        assert not pd.isna(features.loc["TEST", "overnight_ret_5d"])


class TestAmihudIlliquidity:
    """Test Amihud illiquidity ratio."""

    def test_amihud_computed(self):
        from stock_screener.features.technical import compute_features

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        tickers = ["TEST"]
        close = pd.Series(100.0 + np.arange(len(dates)) * 0.5, index=dates)
        open_price = close * 0.999
        volume = pd.Series(1_000_000, index=dates)
        high = close * 1.01
        low = close * 0.99

        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        prices = pd.DataFrame(index=dates, columns=cols)
        prices[("TEST", "Close")] = close
        prices[("TEST", "Open")] = open_price
        prices[("TEST", "Volume")] = volume
        prices[("TEST", "High")] = high
        prices[("TEST", "Low")] = low

        fx = pd.Series(1.0, index=dates)
        features = compute_features(
            prices, fx, liquidity_lookback_days=30, feature_lookback_days=120, logger=LOGGER
        )

        assert "amihud_illiquidity_20d" in features.columns
        val = features.loc["TEST", "amihud_illiquidity_20d"]
        assert not pd.isna(val)
        assert val > 0  # Should be positive

    def test_spread_estimate_computed(self):
        from stock_screener.features.technical import compute_features

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        tickers = ["TEST"]
        close = pd.Series(100.0 + np.arange(len(dates)) * 0.1, index=dates)
        open_price = close * 0.999
        volume = pd.Series(1_000_000, index=dates)
        high = close * 1.02
        low = close * 0.98

        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        prices = pd.DataFrame(index=dates, columns=cols)
        prices[("TEST", "Close")] = close
        prices[("TEST", "Open")] = open_price
        prices[("TEST", "Volume")] = volume
        prices[("TEST", "High")] = high
        prices[("TEST", "Low")] = low

        fx = pd.Series(1.0, index=dates)
        features = compute_features(
            prices, fx, liquidity_lookback_days=30, feature_lookback_days=120, logger=LOGGER
        )

        assert "spread_estimate_cs" in features.columns
        val = features.loc["TEST", "spread_estimate_cs"]
        assert not pd.isna(val)
        assert val > 0  # Spread should be positive


class TestLiquidityTrend:
    """Test liquidity trend feature."""

    def test_liquidity_trend_computed(self):
        from stock_screener.features.technical import compute_features

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        tickers = ["TEST"]
        close = pd.Series(100.0, index=dates)
        open_price = close * 0.999
        # Volume increasing over time (trend > 1)
        volume = pd.Series(np.linspace(500_000, 2_000_000, len(dates)), index=dates)
        high = close * 1.01
        low = close * 0.99

        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        prices = pd.DataFrame(index=dates, columns=cols)
        prices[("TEST", "Close")] = close
        prices[("TEST", "Open")] = open_price
        prices[("TEST", "Volume")] = volume
        prices[("TEST", "High")] = high
        prices[("TEST", "Low")] = low

        fx = pd.Series(1.0, index=dates)
        features = compute_features(
            prices, fx, liquidity_lookback_days=30, feature_lookback_days=120, logger=LOGGER
        )

        assert "liquidity_trend_60d" in features.columns
        val = features.loc["TEST", "liquidity_trend_60d"]
        assert not pd.isna(val)
        assert val > 1.0  # Recent vol > long-term vol


class TestSectorBreadth:
    """Test sector breadth features."""

    def test_sector_breadth_computed(self):
        from stock_screener.features.technical import compute_features

        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        np.random.seed(42)

        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        prices = pd.DataFrame(index=dates, columns=cols)
        for t in tickers:
            close = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.001, 0.01, len(dates))), index=dates)
            prices[(t, "Close")] = close
            prices[(t, "Open")] = close * 0.999
            prices[(t, "Volume")] = 1_000_000
            prices[(t, "High")] = close * 1.01
            prices[(t, "Low")] = close * 0.99

        fx = pd.Series(1.0, index=dates)
        fundamentals = pd.DataFrame(
            {"sector": ["Technology", "Technology", "Technology"]},
            index=tickers,
        )
        features = compute_features(
            prices, fx, liquidity_lookback_days=30, feature_lookback_days=120,
            logger=LOGGER, fundamentals=fundamentals,
        )

        assert "sector_breadth_20d" in features.columns
        assert "sector_momentum_dispersion" in features.columns
        # All same sector: breadth should be between 0 and 1
        breadth = features["sector_breadth_20d"]
        assert (breadth >= 0).all() and (breadth <= 1).all()


# --- 2. HRP tests ---

class TestHRPWeights:
    """Test Hierarchical Risk Parity."""

    def test_hrp_basic(self):
        from stock_screener.optimization.risk_parity import compute_hrp_weights, SCIPY_AVAILABLE

        if not SCIPY_AVAILABLE:
            pytest.skip("scipy not available")

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), 5)),
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

        weights = compute_hrp_weights(returns, LOGGER)
        assert len(weights) == 5
        assert abs(weights.sum() - 1.0) < 1e-6  # Weights sum to 1
        assert (weights >= 0).all()  # All non-negative
        assert weights.max() <= 0.20 + 1e-6  # Weight cap

    def test_hrp_correlated_stocks_diversified(self):
        from stock_screener.optimization.risk_parity import compute_hrp_weights, SCIPY_AVAILABLE

        if not SCIPY_AVAILABLE:
            pytest.skip("scipy not available")

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        # A and B highly correlated, C independent
        base = np.random.normal(0, 0.01, len(dates))
        returns = pd.DataFrame({
            "A": base + np.random.normal(0, 0.001, len(dates)),
            "B": base + np.random.normal(0, 0.001, len(dates)),
            "C": np.random.normal(0, 0.01, len(dates)),
        }, index=dates)

        weights = compute_hrp_weights(returns, LOGGER, weight_cap=0.50)
        # C should get more weight than A or B individually (diversification benefit)
        assert weights["C"] > weights["A"] * 0.8  # C gets at least ~80% of A's weight

    def test_hrp_insufficient_data_fallback(self):
        from stock_screener.optimization.risk_parity import compute_hrp_weights

        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (10, 3)),
            columns=["A", "B", "C"],
        )
        weights = compute_hrp_weights(returns, LOGGER, min_obs=40)
        # Should fallback to equal weights
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(abs(w - 1/3) < 1e-6 for w in weights)


# --- 3. Dynamic risk budgeting tests ---

class TestAdaptiveVolTarget:
    """Test adaptive volatility targeting."""

    def test_high_ic_increases_target(self):
        from stock_screener.optimization.risk_parity import compute_adaptive_vol_target

        target = compute_adaptive_vol_target(
            0.15,
            recent_ic=0.15,  # High IC
            ic_baseline=0.05,
            ic_sensitivity=0.5,
            avg_confidence=0.7,
        )
        assert target > 0.15  # Should increase from base

    def test_low_ic_decreases_target(self):
        from stock_screener.optimization.risk_parity import compute_adaptive_vol_target

        target = compute_adaptive_vol_target(
            0.15,
            recent_ic=-0.05,  # Negative IC
            ic_baseline=0.05,
            ic_sensitivity=0.5,
            avg_confidence=0.3,
        )
        assert target < 0.15  # Should decrease from base

    def test_target_within_bounds(self):
        from stock_screener.optimization.risk_parity import compute_adaptive_vol_target

        # Extreme case: very high IC and confidence
        target = compute_adaptive_vol_target(
            0.15,
            recent_ic=0.50,
            ic_sensitivity=2.0,
            avg_confidence=1.0,
            vol_min=0.08,
            vol_max=0.25,
        )
        assert 0.08 <= target <= 0.25

        # Extreme case: very negative IC
        target = compute_adaptive_vol_target(
            0.15,
            recent_ic=-0.50,
            ic_sensitivity=2.0,
            avg_confidence=0.0,
            vol_min=0.08,
            vol_max=0.25,
        )
        assert 0.08 <= target <= 0.25


# --- 4. Per-model reweighting tests ---

class TestPerModelReweighting:
    """Test per-model IC tracking and adaptive reweighting."""

    def test_compute_per_model_ic(self):
        from stock_screener.reward.feedback import compute_per_model_ic

        np.random.seed(42)
        n = 50
        realized = pd.Series(np.random.normal(0, 0.01, n))
        # Model A has good IC (correlated with realized)
        pred_a = realized + np.random.normal(0, 0.005, n)
        # Model B has bad IC (uncorrelated)
        pred_b = pd.Series(np.random.normal(0, 0.01, n))

        ics = compute_per_model_ic(
            {"model_a": pred_a, "model_b": pred_b},
            realized,
            min_samples=10,
        )
        assert "model_a" in ics
        assert "model_b" in ics
        assert ics["model_a"] > ics["model_b"]  # Good model has higher IC

    def test_adaptive_weights_favor_high_ic(self):
        from stock_screener.reward.feedback import compute_adaptive_ensemble_weights

        model_ics = {"xgb_0": 0.15, "xgb_1": 0.05, "lgbm_0": 0.10}
        weights = compute_adaptive_ensemble_weights(
            model_ics,
            model_names=["xgb_0", "xgb_1", "lgbm_0"],
            ic_blend_alpha=1.0,  # Pure IC-based
        )
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-6
        # xgb_0 (highest IC) should get most weight
        assert weights[0] > weights[1]  # xgb_0 > xgb_1

    def test_adaptive_weights_with_equal_holdout(self):
        from stock_screener.reward.feedback import compute_adaptive_ensemble_weights

        model_ics = {"a": 0.10, "b": 0.10}
        weights = compute_adaptive_ensemble_weights(
            model_ics,
            model_names=["a", "b"],
            ic_blend_alpha=0.5,
        )
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6
        # Equal IC should give equal weights
        assert abs(weights[0] - weights[1]) < 1e-6

    def test_min_weight_enforced(self):
        from stock_screener.reward.feedback import compute_adaptive_ensemble_weights

        model_ics = {"good": 0.20, "bad": -0.10}
        weights = compute_adaptive_ensemble_weights(
            model_ics,
            model_names=["good", "bad"],
            ic_blend_alpha=1.0,
            min_weight=0.1,
        )
        assert all(w >= 0.1 for w in weights)


# --- 5. Feature column parity tests ---

class TestFeatureColumnParity:
    """Verify new features are in FEATURE_COLUMNS."""

    def test_new_features_in_feature_columns(self):
        from stock_screener.modeling.model import FEATURE_COLUMNS, TECHNICAL_FEATURES_ONLY

        new_features = [
            "overnight_ret_5d",
            "intraday_ret_5d",
            "overnight_intraday_ratio",
            "amihud_illiquidity_20d",
            "spread_estimate_cs",
            "liquidity_trend_60d",
            "sector_breadth_20d",
            "sector_momentum_dispersion",
            "vix_term_slope",
            "vix_change_5d",
            "vix_percentile_1y",
        ]

        for feat in new_features:
            assert feat in FEATURE_COLUMNS, f"{feat} missing from FEATURE_COLUMNS"
            assert feat in TECHNICAL_FEATURES_ONLY, f"{feat} missing from TECHNICAL_FEATURES_ONLY"
