import pandas as pd

from stock_screener.modeling.transform import normalize_features_cross_section


def test_regime_and_macro_columns_are_excluded_from_cross_sectional_normalization():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05")],
            "ticker": ["AAA", "BBB"],
            "ret_20d": [0.10, -0.10],
            "market_trend_20d": [0.03, 0.03],
            "market_vol_regime": [1.20, 1.20],
            "market_breadth": [0.65, 0.65],
            "vix": [22.0, 22.0],
        }
    )
    out = normalize_features_cross_section(df, date_col="date")
    assert float(out["market_trend_20d"].iloc[0]) == float(df["market_trend_20d"].iloc[0])
    assert float(out["market_vol_regime"].iloc[0]) == float(df["market_vol_regime"].iloc[0])
    assert float(out["market_breadth"].iloc[0]) == float(df["market_breadth"].iloc[0])
    assert float(out["vix"].iloc[0]) == float(df["vix"].iloc[0])
