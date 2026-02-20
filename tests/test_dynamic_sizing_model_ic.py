from stock_screener.config import Config
from stock_screener.pipeline.daily import _extract_model_holdout_ic


def test_extract_model_holdout_ic_prefers_regressor_payload():
    metadata = {
        "regressor": {"holdout": {"mean_ic": 0.203}},
        "holdout": {"mean_ic": 0.05},
    }
    assert _extract_model_holdout_ic(metadata) == 0.203


def test_extract_model_holdout_ic_supports_legacy_payload():
    metadata = {"holdout": {"mean_ic": "0.117"}}
    assert _extract_model_holdout_ic(metadata) == 0.117


def test_extract_model_holdout_ic_handles_missing_or_invalid_values():
    assert _extract_model_holdout_ic(None) is None
    assert _extract_model_holdout_ic({"regressor": {"holdout": {"mean_ic": "nan"}}}) is None
    assert _extract_model_holdout_ic({"regressor": {"holdout": {"mean_ic": "bad"}}}) is None


def test_dynamic_size_min_pred_return_default_is_aligned(monkeypatch):
    monkeypatch.delenv("DYNAMIC_SIZE_MIN_PRED_RETURN", raising=False)
    assert Config().dynamic_size_min_pred_return == 0.01
    assert Config.from_env().dynamic_size_min_pred_return == 0.01

    monkeypatch.setenv("DYNAMIC_SIZE_MIN_PRED_RETURN", "0.025")
    assert Config.from_env().dynamic_size_min_pred_return == 0.025


def test_dynamic_size_max_positions_is_capped_by_portfolio_size(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_SIZE", "5")
    monkeypatch.setenv("DYNAMIC_SIZE_MAX_POSITIONS", "50")
    assert Config.from_env().dynamic_size_max_positions == 5

    monkeypatch.setenv("PORTFOLIO_SIZE", "3")
    monkeypatch.setenv("DYNAMIC_SIZE_MAX_POSITIONS", "10")
    assert Config.from_env().dynamic_size_max_positions == 3


def test_portfolio_size_cap_allows_broader_diversification(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_SIZE", "50")
    cfg = Config.from_env()
    assert cfg.portfolio_size == 12
    assert cfg.dynamic_size_max_positions == 12


def test_entry_dynamic_threshold_defaults_and_env(monkeypatch):
    monkeypatch.delenv("ENTRY_DYNAMIC_THRESHOLDS_ENABLED", raising=False)
    monkeypatch.delenv("ENTRY_MIN_PRED_RETURN_FLOOR", raising=False)
    cfg_default = Config.from_env()
    assert cfg_default.entry_dynamic_thresholds_enabled is True
    assert cfg_default.entry_min_pred_return_floor == 0.0025

    monkeypatch.setenv("ENTRY_DYNAMIC_THRESHOLDS_ENABLED", "0")
    monkeypatch.setenv("ENTRY_MIN_PRED_RETURN_FLOOR", "0.004")
    cfg_env = Config.from_env()
    assert cfg_env.entry_dynamic_thresholds_enabled is False
    assert cfg_env.entry_min_pred_return_floor == 0.004


def test_instrument_sleeve_defaults_and_env(monkeypatch):
    monkeypatch.delenv("INSTRUMENT_SLEEVE_CONSTRAINTS_ENABLED", raising=False)
    monkeypatch.delenv("INSTRUMENT_FUND_MAX_WEIGHT", raising=False)
    monkeypatch.delenv("INSTRUMENT_EQUITY_MIN_WEIGHT", raising=False)
    cfg_default = Config.from_env()
    assert cfg_default.instrument_sleeve_constraints_enabled is True
    assert cfg_default.instrument_fund_max_weight == 0.35
    assert cfg_default.instrument_equity_min_weight == 0.50

    monkeypatch.setenv("INSTRUMENT_SLEEVE_CONSTRAINTS_ENABLED", "0")
    monkeypatch.setenv("INSTRUMENT_FUND_MAX_WEIGHT", "0.25")
    monkeypatch.setenv("INSTRUMENT_EQUITY_MIN_WEIGHT", "0.65")
    cfg_env = Config.from_env()
    assert cfg_env.instrument_sleeve_constraints_enabled is False
    assert cfg_env.instrument_fund_max_weight == 0.25
    assert cfg_env.instrument_equity_min_weight == 0.65


def test_rotate_on_missing_data_defaults_off_and_reads_env(monkeypatch):
    monkeypatch.delenv("ROTATE_ON_MISSING_DATA", raising=False)
    assert Config.from_env().rotate_on_missing_data is False

    monkeypatch.setenv("ROTATE_ON_MISSING_DATA", "1")
    assert Config.from_env().rotate_on_missing_data is True
