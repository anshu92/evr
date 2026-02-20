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
    monkeypatch.delenv("ENTRY_DYNAMIC_MIN_CONF_TOP_DECILE_SPREAD", raising=False)
    monkeypatch.delenv("ENTRY_DYNAMIC_MIN_PRED_TOP_DECILE_SPREAD", raising=False)
    monkeypatch.delenv("ENTRY_STRESS_GUARD_ENABLED", raising=False)
    monkeypatch.delenv("ENTRY_STRESS_HOLD_ONLY_ENABLED", raising=False)
    monkeypatch.delenv("RET_PER_DAY_MIN_PEAK_DAYS", raising=False)
    monkeypatch.delenv("RET_PER_DAY_MAX_PEAK_DAYS", raising=False)
    monkeypatch.delenv("RET_PER_DAY_SMOOTHING_K", raising=False)
    monkeypatch.delenv("RET_PER_DAY_SHIFT_ALERT_ENABLED", raising=False)
    monkeypatch.delenv("RET_PER_DAY_SHIFT_ALERT_MIN_SAMPLES", raising=False)
    cfg_default = Config.from_env()
    assert cfg_default.entry_dynamic_thresholds_enabled is True
    assert cfg_default.entry_min_pred_return_floor == 0.0025
    assert cfg_default.entry_dynamic_min_conf_top_decile_spread == 0.03
    assert cfg_default.entry_dynamic_min_pred_top_decile_spread == 0.002
    assert cfg_default.entry_stress_guard_enabled is True
    assert cfg_default.entry_stress_hold_only_enabled is False
    assert cfg_default.ret_per_day_min_peak_days == 1.0
    assert cfg_default.ret_per_day_max_peak_days == 10.0
    assert cfg_default.ret_per_day_smoothing_k == 1.0
    assert cfg_default.ret_per_day_shift_alert_enabled is True
    assert cfg_default.ret_per_day_shift_alert_min_samples == 20

    monkeypatch.setenv("ENTRY_DYNAMIC_THRESHOLDS_ENABLED", "0")
    monkeypatch.setenv("ENTRY_MIN_PRED_RETURN_FLOOR", "0.004")
    monkeypatch.setenv("ENTRY_DYNAMIC_MIN_CONF_TOP_DECILE_SPREAD", "0.015")
    monkeypatch.setenv("ENTRY_DYNAMIC_MIN_PRED_TOP_DECILE_SPREAD", "0.001")
    monkeypatch.setenv("ENTRY_STRESS_GUARD_ENABLED", "0")
    monkeypatch.setenv("ENTRY_STRESS_HOLD_ONLY_ENABLED", "1")
    monkeypatch.setenv("RET_PER_DAY_MIN_PEAK_DAYS", "2.0")
    monkeypatch.setenv("RET_PER_DAY_MAX_PEAK_DAYS", "9.0")
    monkeypatch.setenv("RET_PER_DAY_SMOOTHING_K", "0.5")
    monkeypatch.setenv("RET_PER_DAY_SHIFT_ALERT_ENABLED", "0")
    monkeypatch.setenv("RET_PER_DAY_SHIFT_ALERT_MIN_SAMPLES", "12")
    cfg_env = Config.from_env()
    assert cfg_env.entry_dynamic_thresholds_enabled is False
    assert cfg_env.entry_min_pred_return_floor == 0.004
    assert cfg_env.entry_dynamic_min_conf_top_decile_spread == 0.015
    assert cfg_env.entry_dynamic_min_pred_top_decile_spread == 0.001
    assert cfg_env.entry_stress_guard_enabled is False
    assert cfg_env.entry_stress_hold_only_enabled is True
    assert cfg_env.ret_per_day_min_peak_days == 2.0
    assert cfg_env.ret_per_day_max_peak_days == 9.0
    assert cfg_env.ret_per_day_smoothing_k == 0.5
    assert cfg_env.ret_per_day_shift_alert_enabled is False
    assert cfg_env.ret_per_day_shift_alert_min_samples == 12


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
