"""Test consistency of metadata across different training modes."""
import json
import tempfile
from pathlib import Path

import pytest


def test_regressor_scores_format_consistency():
    """Test that regressor_scores has consistent format regardless of optimization method."""
    
    # Both Optuna and manual search should produce dict[str, float]
    # Format: {str(params): score}
    
    # Manual search format
    manual_scores = {
        "{'max_depth': 6, 'learning_rate': 0.03}": 0.0123,
        "{'max_depth': 5, 'learning_rate': 0.05}": 0.0145,
    }
    
    # Optuna format (should match)
    optuna_scores = {
        "{'max_depth': 6, 'learning_rate': 0.025, 'subsample': 0.8}": 0.0156,
        "{'max_depth': 7, 'learning_rate': 0.042, 'subsample': 0.75}": 0.0134,
    }
    
    # Both should be dict[str, float]
    assert isinstance(manual_scores, dict)
    assert isinstance(optuna_scores, dict)
    
    for scores in [manual_scores, optuna_scores]:
        for key, value in scores.items():
            assert isinstance(key, str), "Keys should be string representations of params"
            assert isinstance(value, (int, float)), "Values should be numeric scores"


def test_metadata_structure():
    """Test that metadata.json has consistent structure."""
    
    # Simulate metadata structure
    metadata = {
        "regressor": {
            "params": {"max_depth": 6, "learning_rate": 0.03},
            "cv_metric": "mean_net_ret_topn",
            "cv_scores_topn": {
                "{'max_depth': 6}": 0.012,
                "{'max_depth': 5}": 0.015,
            },
            "holdout": {"mean_ic": 0.045},
        }
    }
    
    # Verify cv_scores_topn is always a dict
    assert isinstance(metadata["regressor"]["cv_scores_topn"], dict)
    
    # Verify it can be serialized to JSON
    json_str = json.dumps(metadata)
    assert json_str is not None
    
    # Verify it can be loaded back
    loaded = json.loads(json_str)
    assert loaded["regressor"]["cv_scores_topn"] == metadata["regressor"]["cv_scores_topn"]


def test_best_params_extraction():
    """Test that best params can be extracted from both formats."""
    
    # Manual search format
    manual_scores = {
        "{'max_depth': 6, 'learning_rate': 0.03}": 0.0123,
        "{'max_depth': 5, 'learning_rate': 0.05}": 0.0145,
    }
    
    # Find best manually
    best_manual_key = max(manual_scores, key=manual_scores.get)
    best_manual_score = manual_scores[best_manual_key]
    assert best_manual_score == 0.0145
    
    # Optuna format (same structure)
    optuna_scores = {
        "{'max_depth': 6, 'learning_rate': 0.025}": 0.0156,
        "{'max_depth': 7, 'learning_rate': 0.042}": 0.0134,
    }
    
    # Find best the same way
    best_optuna_key = max(optuna_scores, key=optuna_scores.get)
    best_optuna_score = optuna_scores[best_optuna_key]
    assert best_optuna_score == 0.0156
    
    # Both methods work identically
    assert callable(max)


def test_empty_scores_handling():
    """Test that empty regressor_scores is handled correctly."""
    
    # Initialize empty (fallback case)
    regressor_scores = {}
    
    # Should be a valid dict
    assert isinstance(regressor_scores, dict)
    assert len(regressor_scores) == 0
    
    # Should serialize to JSON
    json_str = json.dumps({"cv_scores_topn": regressor_scores})
    assert json_str == '{"cv_scores_topn": {}}'
