import logging

import pandas as pd

from stock_screener.screening.screener import score_universe, screen_universe


def _base_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_days": [120, 120, 120, 120, 120, 120],
            "last_close_cad": [10, 11, 12, 0.5, 15, 16],  # one below price filter
            "avg_dollar_volume_cad": [1_000_000, 900_000, 800_000, 1_000_000, 10_000, 700_000],  # one below vol
            "ret_60d": [0, 0, 0, 0, 0, 0],
            "ret_120d": [0, 0, 0, 0, 0, 0],
            "ma20_ratio": [0, 0, 0, 0, 0, 0],
            "vol_60d_ann": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "pred_return": [0.2, 0.1, -0.1, 0.3, 0.4, 0.0],
        },
        index=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
    )


def test_score_universe_filters_and_order():
    logger = logging.getLogger("test")
    df = _base_features()

    scored = score_universe(
        features=df,
        min_price_cad=2.0,
        min_avg_dollar_volume_cad=250_000.0,
        logger=logger,
    )

    # DDD (price) and EEE (liquidity) filtered out
    assert set(scored.index) == {"AAA", "BBB", "CCC", "FFF"}

    # With baseline == 0 and ML signal present, ordering follows pred_return zscore
    assert scored.index[0] == "AAA"
    assert scored.index[-1] == "CCC"

    screened = screen_universe(
        features=df,
        min_price_cad=2.0,
        min_avg_dollar_volume_cad=250_000.0,
        top_n=2,
        logger=logger,
    )
    assert len(screened) == 2
    assert list(screened.index) == list(scored.index[:2])
