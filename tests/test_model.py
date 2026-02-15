import logging
import numpy as np
import pandas as pd
import pytest

from stock_screener.modeling.model import load_model, save_model, build_model, FEATURE_COLUMNS


def test_model_save_and_load(tmp_path):
    """Test that models can be saved and loaded correctly."""
    logger = logging.getLogger("test")
    
    # Create a simple model
    try:
        model = build_model(random_state=42)
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("XGBoost not available in test environment")
        raise
    
    # Create dummy training data
    X = pd.DataFrame(np.random.randn(100, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y = pd.Series(np.random.randn(100))
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    model_path = tmp_path / "test_model.json"
    save_model(model, model_path)
    
    # Load the model
    loaded_model = load_model(model_path)
    
    # Verify predictions match
    import xgboost as xgb
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(xgb.DMatrix(X))
    
    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


def test_model_handles_missing_features():
    """Test that _coerce_features handles missing columns."""
    from stock_screener.modeling.model import _coerce_features
    
    # Create DataFrame with only some features
    df = pd.DataFrame({
        "ret_20d": [0.1, 0.2, 0.3],
        "vol_60d_ann": [0.2, 0.25, 0.3],
    })
    
    # Coerce features should add missing columns as NaN
    result = _coerce_features(df)
    
    assert len(result.columns) == len(FEATURE_COLUMNS)
    assert "ret_20d" in result.columns
    assert "vol_60d_ann" in result.columns
    assert pd.isna(result["ret_5d"]).all()  # Should be NaN
