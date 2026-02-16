import warnings

import pandas as pd

from stock_screener.modeling.transform import normalize_features_cross_section


def test_normalize_features_handles_all_nan_date_groups_without_runtime_warnings():
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-01-05"),
                pd.Timestamp("2026-01-05"),
                pd.Timestamp("2026-01-06"),
                pd.Timestamp("2026-01-06"),
            ],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "ret_20d": [0.1, -0.1, 0.2, -0.2],
            # First date group is all NaN; second date group has finite values.
            "ret_60d": [float("nan"), float("nan"), 0.3, -0.3],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out = normalize_features_cross_section(
            df,
            date_col="date",
            feature_cols=["ret_20d", "ret_60d"],
        )

    runtime_msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not any("Mean of empty slice" in msg for msg in runtime_msgs)
    assert out.loc[out["date"] == pd.Timestamp("2026-01-05"), "ret_60d"].isna().all()
