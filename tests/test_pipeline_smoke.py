import logging
from datetime import datetime, timezone

import pandas as pd

from stock_screener.config import Config
from stock_screener.pipeline import daily as daily_mod
from stock_screener.portfolio.manager import TradeAction, TradePlan
from stock_screener.portfolio.state import PortfolioState
from stock_screener.utils import Universe


def test_run_daily_smoke(monkeypatch, tmp_path):
    logger = logging.getLogger("test")

    def _fake_universe(*args, **kwargs):
        return Universe(tickers=["AAPL", "MSFT"], meta={"source": "test"})

    def _fake_fx(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        return pd.Series(1.35, index=idx)

    def _fake_prices(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        cols = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close", "Volume"]])
        prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
        prices[("AAPL", "Close")] = [100, 101, 102, 103, 104]
        prices[("MSFT", "Close")] = [200, 201, 202, 203, 204]
        prices[("AAPL", "Volume")] = 2_000_000
        prices[("MSFT", "Volume")] = 2_500_000
        return prices

    def _fake_features(*args, **kwargs):
        return pd.DataFrame(
            {
                "last_close_cad": [140.0, 275.0],
                "avg_dollar_volume_cad": [10_000_000.0, 12_000_000.0],
                "vol_60d_ann": [0.2, 0.25],
                "ret_60d": [0.1, 0.08],
                "ret_120d": [0.2, 0.16],
                "ma20_ratio": [0.02, -0.01],
                "rsi_14": [55.0, 48.0],
                "score": [1.2, 1.0],
                "n_days": [120, 120],
                "last_date": ["2025-01-07", "2025-01-07"],
                "market_vol_regime": [1.0, 1.0],
                "market_trend_20d": [0.02, 0.02],
            },
            index=["AAPL", "MSFT"],
        )

    def _identity_weights(*args, **kwargs):
        df = kwargs.get("features")
        if df is None and args:
            df = args[0]
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        if "weight" not in out.columns:
            out["weight"] = [0.5, 0.5][: len(out)]
        return out

    class _FakePM:
        def __init__(self, *args, **kwargs):
            pass

        def apply_exits(self, *args, **kwargs):
            return []

        def build_trade_plan(self, *, state, screened, weights, prices_cad, scored=None, features=None, blocked_buys=None):
            return TradePlan(actions=[], holdings=weights.copy())

    monkeypatch.setattr(daily_mod, "fetch_us_universe", _fake_universe)
    monkeypatch.setattr(daily_mod, "fetch_tsx_universe", _fake_universe)
    monkeypatch.setattr(daily_mod, "fetch_usdcad", _fake_fx)
    monkeypatch.setattr(daily_mod, "download_price_history", _fake_prices)
    monkeypatch.setattr(daily_mod, "fetch_fundamentals", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(daily_mod, "fetch_macro_indicators", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(daily_mod, "compute_features", _fake_features)
    monkeypatch.setattr(daily_mod, "score_universe", lambda features, **k: features.sort_values("score", ascending=False))
    monkeypatch.setattr(daily_mod, "select_sector_neutral", lambda df, **k: df)
    monkeypatch.setattr(daily_mod, "apply_entry_filters", lambda df, **k: (df, {"rejected_count": 0}))
    monkeypatch.setattr(daily_mod, "compute_inverse_vol_weights", _identity_weights)
    monkeypatch.setattr(daily_mod, "compute_correlation_aware_weights", _identity_weights)
    monkeypatch.setattr(daily_mod, "apply_confidence_weighting", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_conviction_sizing", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_liquidity_adjustment", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_correlation_limits", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_beta_adjustment", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_max_position_cap", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_min_position_filter", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_regime_exposure", lambda w, **k: (w, 0.0, {"enabled": True, "scalar": 1.0}))
    monkeypatch.setattr(daily_mod, "apply_volatility_targeting", lambda w, **k: (w, 0.0))
    monkeypatch.setattr(
        daily_mod,
        "load_portfolio_state",
        lambda *a, **k: PortfolioState(cash_cad=500.0, positions=[], last_updated=datetime.now(tz=timezone.utc)),
    )
    monkeypatch.setattr(daily_mod, "save_portfolio_state", lambda *a, **k: None)
    monkeypatch.setattr(daily_mod, "render_reports", lambda **k: None)
    monkeypatch.setattr(daily_mod, "PortfolioManager", _FakePM)

    cfg = Config(
        cache_dir=str(tmp_path / "cache"),
        data_cache_dir=str(tmp_path / "data_cache"),
        reports_dir=str(tmp_path / "reports"),
        portfolio_state_path=str(tmp_path / "state.json"),
        use_ml=False,
        reward_model_enabled=False,
        dynamic_portfolio_sizing=False,
        volatility_targeting=False,
        regime_exposure_enabled=False,
        drawdown_management=False,
        correlation_limits=False,
        liquidity_adjustment=False,
        conviction_sizing=False,
        beta_adjustment=False,
    )
    daily_mod.run_daily(cfg, logger)


def test_run_daily_blocks_partial_sell_reentry(monkeypatch, tmp_path):
    logger = logging.getLogger("test")

    def _fake_universe(*args, **kwargs):
        return Universe(tickers=["AAPL", "MSFT"], meta={"source": "test"})

    def _fake_fx(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        return pd.Series(1.35, index=idx)

    def _fake_prices(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        cols = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close", "Volume"]])
        prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
        prices[("AAPL", "Close")] = [100, 101, 102, 103, 104]
        prices[("MSFT", "Close")] = [200, 201, 202, 203, 204]
        prices[("AAPL", "Volume")] = 2_000_000
        prices[("MSFT", "Volume")] = 2_500_000
        return prices

    def _fake_features(*args, **kwargs):
        return pd.DataFrame(
            {
                "last_close_cad": [140.0, 275.0],
                "avg_dollar_volume_cad": [10_000_000.0, 12_000_000.0],
                "vol_60d_ann": [0.2, 0.25],
                "ret_60d": [0.1, 0.08],
                "ret_120d": [0.2, 0.16],
                "ma20_ratio": [0.02, -0.01],
                "rsi_14": [55.0, 48.0],
                "score": [1.2, 1.0],
                "n_days": [120, 120],
                "last_date": ["2025-01-07", "2025-01-07"],
                "market_vol_regime": [1.0, 1.0],
                "market_trend_20d": [0.02, 0.02],
            },
            index=["AAPL", "MSFT"],
        )

    def _identity_weights(*args, **kwargs):
        df = kwargs.get("features")
        if df is None and args:
            df = args[0]
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        if "weight" not in out.columns:
            out["weight"] = [0.5, 0.5][: len(out)]
        return out

    class _FakePM:
        def __init__(self, *args, **kwargs):
            pass

        def apply_exits(self, *args, **kwargs):
            return [
                TradeAction(
                    ticker="AAPL",
                    action="SELL_PARTIAL",
                    reason="PEAK_TEST",
                    shares=1.0,
                    price_cad=140.0,
                    days_held=3,
                )
            ]

        def build_trade_plan(self, *, state, screened, weights, prices_cad, scored=None, features=None, blocked_buys=None):
            assert blocked_buys is not None
            assert "AAPL" in {str(t).upper() for t in blocked_buys}
            return TradePlan(actions=[], holdings=weights.copy())

    monkeypatch.setattr(daily_mod, "fetch_us_universe", _fake_universe)
    monkeypatch.setattr(daily_mod, "fetch_tsx_universe", _fake_universe)
    monkeypatch.setattr(daily_mod, "fetch_usdcad", _fake_fx)
    monkeypatch.setattr(daily_mod, "download_price_history", _fake_prices)
    monkeypatch.setattr(daily_mod, "fetch_fundamentals", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(daily_mod, "fetch_macro_indicators", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(daily_mod, "compute_features", _fake_features)
    monkeypatch.setattr(daily_mod, "score_universe", lambda features, **k: features.sort_values("score", ascending=False))
    monkeypatch.setattr(daily_mod, "select_sector_neutral", lambda df, **k: df)
    monkeypatch.setattr(daily_mod, "apply_entry_filters", lambda df, **k: (df, {"rejected_count": 0}))
    monkeypatch.setattr(daily_mod, "compute_inverse_vol_weights", _identity_weights)
    monkeypatch.setattr(daily_mod, "compute_correlation_aware_weights", _identity_weights)
    monkeypatch.setattr(daily_mod, "apply_confidence_weighting", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_conviction_sizing", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_liquidity_adjustment", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_correlation_limits", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_beta_adjustment", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_max_position_cap", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_min_position_filter", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_regime_exposure", lambda w, **k: (w, 0.0, {"enabled": True, "scalar": 1.0}))
    monkeypatch.setattr(daily_mod, "apply_volatility_targeting", lambda w, **k: (w, 0.0))
    monkeypatch.setattr(
        daily_mod,
        "load_portfolio_state",
        lambda *a, **k: PortfolioState(cash_cad=500.0, positions=[], last_updated=datetime.now(tz=timezone.utc)),
    )
    monkeypatch.setattr(daily_mod, "save_portfolio_state", lambda *a, **k: None)
    monkeypatch.setattr(daily_mod, "render_reports", lambda **k: None)
    monkeypatch.setattr(daily_mod, "PortfolioManager", _FakePM)

    cfg = Config(
        cache_dir=str(tmp_path / "cache"),
        data_cache_dir=str(tmp_path / "data_cache"),
        reports_dir=str(tmp_path / "reports"),
        portfolio_state_path=str(tmp_path / "state.json"),
        use_ml=False,
        reward_model_enabled=False,
        dynamic_portfolio_sizing=False,
        volatility_targeting=False,
        regime_exposure_enabled=False,
        drawdown_management=False,
        correlation_limits=False,
        liquidity_adjustment=False,
        conviction_sizing=False,
        beta_adjustment=False,
    )
    daily_mod.run_daily(cfg, logger)
