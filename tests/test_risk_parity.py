import logging

import pandas as pd

from stock_screener.optimization.risk_parity import compute_inverse_vol_weights


def test_weight_cap_allows_cash_when_infeasible():
    logger = logging.getLogger("test")
    df = pd.DataFrame(
        {
            "vol_60d_ann": [0.2, 0.2, 0.2, 0.2, 0.2],
        },
        index=["A", "B", "C", "D", "E"],
    )

    out = compute_inverse_vol_weights(
        features=df,
        portfolio_size=5,
        weight_cap=0.10,
        logger=logger,
    )

    weights = out["weight"]
    assert (weights <= 0.10 + 1e-9).all()
    # Cap is infeasible (0.10 * 5 = 0.50), so sum should be <= 0.50
    assert weights.sum() <= 0.50 + 1e-9
