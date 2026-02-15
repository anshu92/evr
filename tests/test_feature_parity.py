import logging

import pandas as pd
import pytest

from stock_screener.modeling.model import compute_feature_schema_hash
from stock_screener.pipeline.daily import _validate_feature_parity


def test_feature_schema_hash_stable():
    cols = ["a", "b", "c"]
    h1 = compute_feature_schema_hash(cols)
    h2 = compute_feature_schema_hash(cols)
    assert h1 == h2
    assert len(h1) == 64


def test_validate_feature_parity_strict_raises():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(RuntimeError):
        _validate_feature_parity(
            df,
            ["a", "b", "c"],
            strict=True,
            logger=logging.getLogger("test"),
        )


def test_validate_feature_parity_non_strict_returns_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    missing = _validate_feature_parity(
        df,
        ["a", "b", "c"],
        strict=False,
        logger=logging.getLogger("test"),
    )
    assert missing == ["c"]
