"""Tests for state-of-the-art model enhancements."""
import logging
import numpy as np
import pandas as pd
import pytest

from stock_screener.modeling.model import (
    build_model,
    build_lgbm_model,
    predict,
    predict_ensemble_with_uncertainty,
    TECHNICAL_FEATURES_ONLY,
)


def test_lightgbm_model_creation():
    """Test that LightGBM model can be created."""
    try:
        model = build_lgbm_model(random_state=42)
        assert model is not None
        assert hasattr(model, 'fit')
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("LightGBM not available in test environment")
        raise


def test_mixed_ensemble_predictions():
    """Test that mixed XGB+LGB ensemble works."""
    try:
        # Create dummy data
        X = pd.DataFrame(
            np.random.randn(100, len(TECHNICAL_FEATURES_ONLY)),
            columns=TECHNICAL_FEATURES_ONLY
        )
        y = pd.Series(np.random.randn(100))
        
        # Train a small XGBoost model
        xgb_model = build_model(random_state=42)
        xgb_model.set_params(n_estimators=10)
        xgb_model.fit(X, y)
        
        # Train a small LightGBM model
        lgbm_model = build_lgbm_model(random_state=42)
        lgbm_model.set_params(n_estimators=10)
        lgbm_model.fit(X, y)
        
        models = [xgb_model, lgbm_model]
        
        # Test predictions
        pred_df = predict_ensemble_with_uncertainty(models, None, X)
        
        assert "pred_return" in pred_df.columns
        assert "pred_uncertainty" in pred_df.columns
        assert "pred_confidence" in pred_df.columns
        
        # Confidence should be 0-1
        assert (pred_df["pred_confidence"] >= 0).all()
        assert (pred_df["pred_confidence"] <= 1).all()
        
        # Uncertainty should be non-negative
        assert (pred_df["pred_uncertainty"] >= 0).all()
        
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("XGBoost or LightGBM not available in test environment")
        raise


def test_uncertainty_reflects_disagreement():
    """Test that uncertainty increases when models disagree."""
    try:
        X = pd.DataFrame(
            np.random.randn(100, len(TECHNICAL_FEATURES_ONLY)),
            columns=TECHNICAL_FEATURES_ONLY
        )
        y1 = pd.Series(np.random.randn(100))
        y2 = pd.Series(np.random.randn(100)) * 10  # Very different targets
        
        # Train two models on different targets
        model1 = build_model(random_state=1)
        model1.set_params(n_estimators=10)
        model1.fit(X, y1)
        
        model2 = build_model(random_state=2)
        model2.set_params(n_estimators=10)
        model2.fit(X, y2)
        
        models = [model1, model2]
        pred_df = predict_ensemble_with_uncertainty(models, None, X)
        
        # Models trained on different targets should have high uncertainty
        assert pred_df["pred_uncertainty"].mean() > 0
        
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip("XGBoost not available in test environment")
        raise


def test_confidence_weighting():
    """Test confidence-weighted position sizing."""
    from stock_screener.optimization.risk_parity import apply_confidence_weighting
    
    logger = logging.getLogger("test")
    
    # Create portfolio weights
    weights_df = pd.DataFrame({
        "weight": [0.25, 0.25, 0.25, 0.25],
        "pred_confidence": [0.9, 0.7, 0.5, 0.3],  # Varying confidence
    }, index=["A", "B", "C", "D"])
    
    # Apply confidence weighting
    adjusted = apply_confidence_weighting(weights_df, None, confidence_floor=0.3, logger=logger)
    
    # High confidence should get more weight
    assert adjusted.loc["A", "weight"] > adjusted.loc["D", "weight"]
    
    # Weights should still sum to 1
    assert abs(adjusted["weight"].sum() - 1.0) < 0.01
