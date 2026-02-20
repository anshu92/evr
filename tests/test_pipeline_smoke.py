import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from stock_screener.config import Config
from stock_screener.pipeline import daily as daily_mod
from stock_screener.portfolio.manager import TradeAction, TradePlan
from stock_screener.portfolio.state import PortfolioState, Position
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


def test_resolve_portfolio_state_path_prefers_repo_anchor(tmp_path):
    repo_root = tmp_path / "repo"
    cwd = tmp_path / "cwd"
    repo_root.mkdir()
    cwd.mkdir()
    rel = "state.json"
    (repo_root / rel).write_text("{}", encoding="utf-8")
    (cwd / rel).write_text("{}", encoding="utf-8")

    resolved = daily_mod._resolve_portfolio_state_path(rel, repo_root=repo_root, cwd=cwd)
    assert resolved == (repo_root / rel).resolve()


def test_resolve_portfolio_state_path_uses_existing_cwd_when_repo_missing(tmp_path):
    repo_root = tmp_path / "repo"
    cwd = tmp_path / "cwd"
    repo_root.mkdir()
    cwd.mkdir()
    rel = "state.json"
    (cwd / rel).write_text("{}", encoding="utf-8")

    resolved = daily_mod._resolve_portfolio_state_path(rel, repo_root=repo_root, cwd=cwd)
    assert resolved == (cwd / rel).resolve()


def test_run_daily_emits_exposure_metrics(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured: dict[str, object] = {}

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
            out["weight"] = [0.6, 0.2][: len(out)]  # Intentional under-investment to track cash_weight
        return out

    class _FakePM:
        def __init__(self, *args, **kwargs):
            pass

        def apply_exits(self, *args, **kwargs):
            return []

        def build_trade_plan(self, *, state, screened, weights, prices_cad, scored=None, features=None, blocked_buys=None):
            return TradePlan(actions=[], holdings=weights.copy())

    def _capture_render_reports(**kwargs):
        captured["run_meta"] = kwargs.get("run_meta", {})

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
    monkeypatch.setattr(daily_mod, "render_reports", _capture_render_reports)
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
        exposure_policy="allow_cash_no_upscale",
        target_gross_exposure=1.0,
        allow_leverage=False,
    )
    daily_mod.run_daily(cfg, logger)

    run_meta = captured.get("run_meta", {})
    before = run_meta.get("exposure_control_before_rebalance", {})
    after = run_meta.get("exposure_control_after_rebalance", {})
    assert set(["gross_exposure", "net_exposure", "cash_weight"]).issubset(before.keys())
    assert set(["gross_exposure", "net_exposure", "cash_weight"]).issubset(after.keys())
    assert float(after["gross_exposure"]) <= 1.0 + 1e-9
    projection = run_meta.get("optimizer_projection_audit", {})
    assert isinstance(projection, dict)
    pre_vs_final = projection.get("pre_rebalance_vs_final", {})
    assert set(["l1_distance", "l2_distance", "dropped_tickers"]).issubset(pre_vs_final.keys())
    assert isinstance(projection.get("notional_drops_with_reasons", []), list)


def test_run_daily_fails_when_exit_state_persistence_fails(monkeypatch, tmp_path):
    logger = logging.getLogger("test")

    def _fake_universe(*args, **kwargs):
        return Universe(tickers=["AAPL"], meta={"source": "test"})

    def _fake_fx(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        return pd.Series(1.35, index=idx)

    def _fake_prices(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        cols = pd.MultiIndex.from_product([["AAPL"], ["Close", "Volume"]])
        prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
        prices[("AAPL", "Close")] = [100, 101, 102, 103, 104]
        prices[("AAPL", "Volume")] = 2_000_000
        return prices

    def _fake_features(*args, **kwargs):
        return pd.DataFrame(
            {
                "last_close_cad": [140.0],
                "avg_dollar_volume_cad": [10_000_000.0],
                "vol_60d_ann": [0.2],
                "ret_60d": [0.1],
                "ret_120d": [0.2],
                "ma20_ratio": [0.02],
                "rsi_14": [55.0],
                "score": [1.2],
                "n_days": [120],
                "last_date": ["2025-01-07"],
                "market_vol_regime": [1.0],
                "market_trend_20d": [0.02],
            },
            index=["AAPL"],
        )

    def _identity_weights(*args, **kwargs):
        df = kwargs.get("features")
        if df is None and args:
            df = args[0]
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        if "weight" not in out.columns:
            out["weight"] = [1.0][: len(out)]
        return out

    class _FakePM:
        def __init__(self, *args, **kwargs):
            pass

        def apply_exits(self, *args, **kwargs):
            return [
                TradeAction(
                    ticker="AAPL",
                    action="SELL",
                    reason="STOP_LOSS",
                    shares=1.0,
                    price_cad=140.0,
                    days_held=2,
                )
            ]

        def build_trade_plan(self, *args, **kwargs):
            raise AssertionError("build_trade_plan should not run when exit persistence fails")

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
        lambda *a, **k: PortfolioState(
            cash_cad=500.0,
            positions=[
                Position(
                    ticker="AAPL",
                    entry_price=100.0,
                    entry_date=datetime.now(tz=timezone.utc),
                    shares=1.0,
                )
            ],
            last_updated=datetime.now(tz=timezone.utc),
        ),
    )
    monkeypatch.setattr(daily_mod, "append_portfolio_events", lambda *a, **k: 1)
    monkeypatch.setattr(daily_mod, "save_portfolio_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
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
    with pytest.raises(RuntimeError, match="disk full"):
        daily_mod.run_daily(cfg, logger)


def test_run_daily_ml_failure_forces_hold_only_when_baseline_not_allowed(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured: dict[str, object] = {}

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
            captured["weights_index"] = list(weights.index.astype(str))
            return TradePlan(actions=[], holdings=weights.copy())

    def _capture_render_reports(**kwargs):
        captured["run_meta"] = kwargs.get("run_meta", {})

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
    monkeypatch.setattr(daily_mod, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model missing")))
    monkeypatch.setattr(daily_mod, "apply_regime_exposure", lambda w, **k: (w, 0.0, {"enabled": True, "scalar": 1.0}))
    monkeypatch.setattr(daily_mod, "apply_volatility_targeting", lambda w, **k: (w, 0.0))
    monkeypatch.setattr(
        daily_mod,
        "load_portfolio_state",
        lambda *a, **k: PortfolioState(
            cash_cad=200.0,
            positions=[
                Position(
                    ticker="AAPL",
                    entry_price=140.0,
                    entry_date=datetime.now(tz=timezone.utc),
                    shares=1.0,
                )
            ],
            last_updated=datetime.now(tz=timezone.utc),
        ),
    )
    monkeypatch.setattr(daily_mod, "save_portfolio_state", lambda *a, **k: None)
    monkeypatch.setattr(daily_mod, "render_reports", _capture_render_reports)
    monkeypatch.setattr(daily_mod, "PortfolioManager", _FakePM)

    cfg = Config(
        cache_dir=str(tmp_path / "cache"),
        data_cache_dir=str(tmp_path / "data_cache"),
        reports_dir=str(tmp_path / "reports"),
        portfolio_state_path=str(tmp_path / "state.json"),
        use_ml=True,
        allow_baseline_trading=False,
        model_path=str(tmp_path / "missing.pkl"),
        reward_model_enabled=False,
        dynamic_portfolio_sizing=False,
        volatility_targeting=False,
        regime_exposure_enabled=False,
        drawdown_management=False,
        correlation_limits=False,
        liquidity_adjustment=False,
        conviction_sizing=False,
        beta_adjustment=False,
        unified_optimizer_enabled=False,
        instrument_sleeve_constraints_enabled=False,
    )
    daily_mod.run_daily(cfg, logger)

    run_meta = captured.get("run_meta", {})
    mode_payload = run_meta.get("strategy_mode", {})
    assert mode_payload.get("mode") == "HOLD_ONLY"
    assert mode_payload.get("ml_expected") is True
    assert mode_payload.get("ml_available") is False
    assert captured.get("weights_index") == ["AAPL"]


def test_run_daily_ml_failure_can_fall_back_to_baseline_when_allowed(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured: dict[str, object] = {}

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
            captured["weights_index"] = list(weights.index.astype(str))
            return TradePlan(actions=[], holdings=weights.copy())

    def _capture_render_reports(**kwargs):
        captured["run_meta"] = kwargs.get("run_meta", {})

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
    monkeypatch.setattr(daily_mod, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model missing")))
    monkeypatch.setattr(daily_mod, "apply_regime_exposure", lambda w, **k: (w, 0.0, {"enabled": True, "scalar": 1.0}))
    monkeypatch.setattr(daily_mod, "apply_volatility_targeting", lambda w, **k: (w, 0.0))
    monkeypatch.setattr(
        daily_mod,
        "load_portfolio_state",
        lambda *a, **k: PortfolioState(cash_cad=500.0, positions=[], last_updated=datetime.now(tz=timezone.utc)),
    )
    monkeypatch.setattr(daily_mod, "save_portfolio_state", lambda *a, **k: None)
    monkeypatch.setattr(daily_mod, "render_reports", _capture_render_reports)
    monkeypatch.setattr(daily_mod, "PortfolioManager", _FakePM)

    cfg = Config(
        cache_dir=str(tmp_path / "cache"),
        data_cache_dir=str(tmp_path / "data_cache"),
        reports_dir=str(tmp_path / "reports"),
        portfolio_state_path=str(tmp_path / "state.json"),
        use_ml=True,
        allow_baseline_trading=True,
        model_path=str(tmp_path / "missing.pkl"),
        reward_model_enabled=False,
        dynamic_portfolio_sizing=False,
        volatility_targeting=False,
        regime_exposure_enabled=False,
        drawdown_management=False,
        correlation_limits=False,
        liquidity_adjustment=False,
        conviction_sizing=False,
        beta_adjustment=False,
        unified_optimizer_enabled=False,
        instrument_sleeve_constraints_enabled=False,
    )
    daily_mod.run_daily(cfg, logger)

    run_meta = captured.get("run_meta", {})
    mode_payload = run_meta.get("strategy_mode", {})
    assert mode_payload.get("mode") == "BASELINE"
    assert mode_payload.get("ml_expected") is True
    assert mode_payload.get("ml_available") is False
    assert len(captured.get("weights_index", [])) >= 1


def test_run_daily_can_force_hold_only_on_entry_stress_guard(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured: dict[str, object] = {}

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
                "market_vol_regime": [2.0, 2.0],
                "market_breadth": [0.30, 0.30],
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
            out["weight"] = [0.6, 0.4][: len(out)]
        return out

    class _FakePM:
        def __init__(self, *args, **kwargs):
            pass

        def apply_exits(self, *args, **kwargs):
            return []

        def build_trade_plan(self, *, state, screened, weights, prices_cad, scored=None, features=None, blocked_buys=None):
            captured["weights_index"] = list(weights.index.astype(str))
            return TradePlan(actions=[], holdings=weights.copy())

    def _capture_render_reports(**kwargs):
        captured["run_meta"] = kwargs.get("run_meta", {})

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
        lambda *a, **k: PortfolioState(
            cash_cad=200.0,
            positions=[
                Position(
                    ticker="AAPL",
                    entry_price=140.0,
                    entry_date=datetime.now(tz=timezone.utc),
                    shares=1.0,
                )
            ],
            last_updated=datetime.now(tz=timezone.utc),
        ),
    )
    monkeypatch.setattr(daily_mod, "save_portfolio_state", lambda *a, **k: None)
    monkeypatch.setattr(daily_mod, "render_reports", _capture_render_reports)
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
        unified_optimizer_enabled=False,
        instrument_sleeve_constraints_enabled=False,
        entry_stress_guard_enabled=True,
        entry_stress_hold_only_enabled=True,
        entry_stress_max_vol_stress=0.65,
        entry_stress_min_breadth=0.45,
    )
    daily_mod.run_daily(cfg, logger)

    run_meta = captured.get("run_meta", {})
    mode_payload = run_meta.get("strategy_mode", {})
    assert mode_payload.get("mode") == "HOLD_ONLY"
    assert mode_payload.get("reason") == "entry_stress_guard_hold_only"
    assert mode_payload.get("entry_stress_guard_triggered") is True
    assert captured.get("weights_index") == ["AAPL"]


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


def test_run_daily_drops_hold_for_ticker_with_same_run_partial_sell(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured_actions = []

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
            return TradePlan(
                actions=[
                    TradeAction(
                        ticker="AAPL",
                        action="HOLD",
                        reason="IN_TARGET",
                        shares=1.0,
                        price_cad=140.0,
                        days_held=3,
                    )
                ],
                holdings=weights.copy(),
            )

    def _capture_render_reports(**kwargs):
        captured_actions[:] = kwargs.get("trade_actions", [])

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
    monkeypatch.setattr(daily_mod, "render_reports", _capture_render_reports)
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

    sells = [a for a in captured_actions if a.action in ("SELL", "SELL_PARTIAL") and a.ticker == "AAPL"]
    holds = [a for a in captured_actions if a.action == "HOLD" and a.ticker == "AAPL"]
    assert len(sells) == 1
    assert len(holds) == 0


def test_run_daily_does_not_flag_fractional_shares_as_legacy_placeholder(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    save_calls = {"n": 0}

    def _fake_universe(*args, **kwargs):
        return Universe(tickers=["AAPL"], meta={"source": "test"})

    def _fake_fx(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        return pd.Series(1.35, index=idx)

    def _fake_prices(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        cols = pd.MultiIndex.from_product([["AAPL"], ["Close", "Volume"]])
        prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
        prices[("AAPL", "Close")] = [100, 101, 102, 103, 104]
        prices[("AAPL", "Volume")] = 2_000_000
        return prices

    def _fake_features(*args, **kwargs):
        return pd.DataFrame(
            {
                "last_close_cad": [140.0],
                "avg_dollar_volume_cad": [10_000_000.0],
                "vol_60d_ann": [0.2],
                "ret_60d": [0.1],
                "ret_120d": [0.2],
                "ma20_ratio": [0.02],
                "rsi_14": [55.0],
                "score": [1.2],
                "n_days": [120],
                "last_date": ["2025-01-07"],
                "market_vol_regime": [1.0],
                "market_trend_20d": [0.02],
            },
            index=["AAPL"],
        )

    def _identity_weights(*args, **kwargs):
        df = kwargs.get("features")
        if df is None and args:
            df = args[0]
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        if "weight" not in out.columns:
            out["weight"] = [1.0][: len(out)]
        return out

    class _FakePM:
        def __init__(self, *args, **kwargs):
            pass

        def apply_exits(self, *args, **kwargs):
            return []

        def build_trade_plan(self, *, state, screened, weights, prices_cad, scored=None, features=None, blocked_buys=None):
            return TradePlan(actions=[], holdings=weights.copy())

    def _count_saves(*args, **kwargs):
        save_calls["n"] += 1

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
        lambda *a, **k: PortfolioState(
            cash_cad=20_000.0,
            positions=[
                Position(
                    ticker="AAPL",
                    entry_price=100.0,
                    entry_date=datetime.now(tz=timezone.utc),
                    shares=1.2,
                )
            ],
            last_updated=datetime.now(tz=timezone.utc),
        ),
    )
    monkeypatch.setattr(daily_mod, "save_portfolio_state", _count_saves)
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
    # No reset-save and no exit-save should occur.
    assert save_calls["n"] == 0


def test_run_daily_liquidity_uses_live_equity(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured: dict[str, float] = {}

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

    def _capture_liquidity(weights, *args, **kwargs):
        captured["portfolio_value"] = float(kwargs.get("portfolio_value", float("nan")))
        return weights

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
    monkeypatch.setattr(daily_mod, "apply_liquidity_adjustment", _capture_liquidity)
    monkeypatch.setattr(daily_mod, "apply_correlation_limits", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_beta_adjustment", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_max_position_cap", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_min_position_filter", lambda w, *a, **k: w)
    monkeypatch.setattr(daily_mod, "apply_regime_exposure", lambda w, **k: (w, 0.0, {"enabled": True, "scalar": 1.0}))
    monkeypatch.setattr(daily_mod, "apply_volatility_targeting", lambda w, **k: (w, 0.0))
    monkeypatch.setattr(
        daily_mod,
        "load_portfolio_state",
        lambda *a, **k: PortfolioState(
            cash_cad=100.0,
            positions=[
                Position(
                    ticker="AAPL",
                    entry_price=100.0,
                    entry_date=datetime.now(tz=timezone.utc),
                    shares=2.0,
                )
            ],
            last_updated=datetime.now(tz=timezone.utc),
        ),
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
        liquidity_adjustment=True,
        conviction_sizing=False,
        beta_adjustment=False,
        unified_optimizer_enabled=False,
    )
    daily_mod.run_daily(cfg, logger)
    # live equity = cash 100 + AAPL market value 2 * 140
    assert captured["portfolio_value"] == 380.0


def test_run_daily_sanitizes_conflicting_sell_buy_actions(monkeypatch, tmp_path):
    logger = logging.getLogger("test")
    captured_actions: list[TradeAction] = []

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
                    action="SELL",
                    reason="PEAK_TARGET(day2,entry)",
                    shares=1.0,
                    price_cad=140.0,
                    days_held=2,
                )
            ]

        def build_trade_plan(self, *, state, screened, weights, prices_cad, scored=None, features=None, blocked_buys=None):
            # Simulate an inconsistent planner output to validate pipeline-level sanitation.
            return TradePlan(
                actions=[
                    TradeAction(
                        ticker="AAPL",
                        action="BUY",
                        reason="TOP_RANKED",
                        shares=1.0,
                        price_cad=140.0,
                    ),
                    TradeAction(
                        ticker="AAPL",
                        action="HOLD",
                        reason="IN_TARGET",
                        shares=1.0,
                        price_cad=140.0,
                    ),
                    TradeAction(
                        ticker="MSFT",
                        action="BUY",
                        reason="TOP_RANKED",
                        shares=0.5,
                        price_cad=275.0,
                    ),
                ],
                holdings=weights.copy(),
            )

    def _capture_render_reports(**kwargs):
        captured_actions[:] = kwargs.get("trade_actions", [])

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
    monkeypatch.setattr(daily_mod, "render_reports", _capture_render_reports)
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

    aapl_actions = [a.action for a in captured_actions if a.ticker == "AAPL"]
    msft_actions = [a.action for a in captured_actions if a.ticker == "MSFT"]
    assert aapl_actions == ["SELL"]
    assert msft_actions == ["BUY"]
