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
