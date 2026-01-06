## GitHub Workflows

This repo has **two** GitHub Actions workflows:

- **Train model** (every ~3 days): `[.github/workflows/train-stock-screener-model.yml](/Users/sahooa3/Documents/git/evr/.github/workflows/train-stock-screener-model.yml)`
- **Daily trading run** (weekdays): `[.github/workflows/daily-stock-screener.yml](/Users/sahooa3/Documents/git/evr/.github/workflows/daily-stock-screener.yml)`

Both workflows use **Actions cache** (no repo commits).

### 1) Train model workflow (every ~3 days)

- **What it does**:
  - Trains the single stock-ranking ML model (`models/ensemble/manifest.json` + members)
  - Uses a **5-day label horizon** (`LABEL_HORIZON_DAYS=5`) to match the max holding period
  - Uploads the model artifact and also stores it in cache for the daily workflow

- **Schedule**:
  - Runs daily at 02:00 UTC, but is gated to execute only every 3rd day.
  - Can be run any time via `workflow_dispatch`.

### 2) Daily trading workflow (weekdays)

- **What it does**:
  - Builds a **US + TSX/TSXV** ticker universe
  - Downloads prices, computes features in **CAD**
  - Ranks candidates (ML if available; baseline fallback otherwise)
  - Produces a **stateful** trading plan with:
    - **SELL** when `days_held >= 5` (`TIME_EXIT`)
    - Optional `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT` exits if configured
    - **BUY** to fill portfolio slots from top-ranked names
    - **HOLD** for positions that remain in the target set
  - Emails `reports/daily_email.html` and attaches:
    - `reports/daily_report.txt`
    - `reports/portfolio_weights.csv`
    - `reports/trade_actions.json`

### Portfolio state

- **State file**: `screener_portfolio_state.json`
- This file is persisted via **Actions cache** and included in the run artifacts.

### Required secrets

- `EMAIL_USERNAME`
- `EMAIL_PASSWORD`
- Optional: `EMAIL_TO` (defaults to `EMAIL_USERNAME`)

### Useful environment variables

- **Universe/screening**:
  - `MAX_TICKERS`, `TOP_N`, `PORTFOLIO_SIZE`, `WEIGHT_CAP`, `MIN_PRICE_CAD`, `MIN_AVG_DOLLAR_VOLUME_CAD`
- **Model**:
  - `USE_ML=1`, `MODEL_PATH=models/ensemble/manifest.json`, `LABEL_HORIZON_DAYS=5`
- **Trading**:
  - `MAX_HOLDING_DAYS=5`
  - Optional: `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`
  - `PORTFOLIO_STATE_PATH=screener_portfolio_state.json`

