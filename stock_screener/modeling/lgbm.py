from __future__ import annotations

try:
    import lightgbm as lgb
except Exception as _lgb_err:  # pragma: no cover
    lgb = None
    _LGB_IMPORT_ERROR = _lgb_err


def _require_lgb() -> None:
    if lgb is None:  # pragma: no cover
        raise RuntimeError(
            "LightGBM is not available in this environment. "
            f"Original error: {_LGB_IMPORT_ERROR}"
        )


def build_lgbm_regressor(random_state: int = 42):
    """Build LightGBM regressor for return prediction."""
    _require_lgb()
    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=32,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=random_state,
        verbose=-1,
    )


def build_lgbm_ranker(random_state: int = 42):
    """Build LightGBM ranker for cross-sectional ranking."""
    _require_lgb()
    return lgb.LGBMRanker(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=32,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=random_state,
        verbose=-1,
    )
