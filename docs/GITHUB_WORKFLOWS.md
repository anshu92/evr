## GitHub Workflows

This repo has **two** GitHub Actions workflows:

- **Train model** (weekly): `.github/workflows/train-stock-screener-model.yml`
- **Daily trading run** (3x per weekday): `.github/workflows/daily-stock-screener.yml`

Both workflows use **Actions cache** (no repo commits).

### 1) Train model workflow (weekly)

- **What it does**:
  - Trains the single stock-ranking ML model (`models/ensemble/manifest.json` + members)
  - Uses a **5-day label horizon** (`LABEL_HORIZON_DAYS=5`) to match the max holding period
  - Uploads the model artifact and also stores it in cache for the daily workflow

- **Schedule**:
  - Runs weekly on **Sunday at 02:00 UTC**.
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

- **Runtime budget controls**:
  - `MAX_DAILY_RUNTIME_MINUTES=12` (default)
  - strict feature schema parity (`STRICT_FEATURE_PARITY=1`)
  - fallback model training disabled by default (`ALLOW_FALLBACK_TRAINING=0`)

- **Cache strategy**:
  - uses stable hash-based cache keys (not per-run IDs) for better cache reuse
  - telemetry artifact emitted at `reports/telemetry/actions_telemetry.json`

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
