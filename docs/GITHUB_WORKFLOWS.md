## GitHub Workflows

This repository currently uses three GitHub Actions workflows:

- `daily-stock-screener.yml`: weekday daily portfolio run and email
- `train-stock-screener-model.yml`: weekly model training/promotion
- `reset-portfolio-state.yml`: manual cache/state reset utility

Detailed pipeline audit: `docs/PORTFOLIO_PIPELINE_AUDIT.md`

## Daily Workflow

File: `.github/workflows/daily-stock-screener.yml`

Schedule:

- Weekdays at `12:00 UTC`, `15:00 UTC`, and `19:30 UTC`
- Manual trigger via `workflow_dispatch`

Core flow:

1. Checkout + Python setup.
2. Restore caches for pip, data/state, and model directory.
3. Find the latest successful training artifact and download model files.
4. Run `python -m stock_screener.cli daily --log-level INFO`.
5. Emit telemetry to `reports/telemetry/actions_telemetry.json`.
6. Upload run artifacts (`reports/`, `cache/last_run_meta.json`, `screener_portfolio_state.json`).
7. Email the HTML report and CSV/JSON attachments.
8. Open a GitHub issue if the workflow fails.

Runtime/controls:

- `MAX_DAILY_RUNTIME_MINUTES=12`
- `STRICT_FEATURE_PARITY=1`
- `USE_ML=1` with fallback behavior when model is unavailable
- Dynamic no-trade-band and turnover controls are enabled by default

## Training Workflow

File: `.github/workflows/train-stock-screener-model.yml`

Schedule:

- Weekly on Sunday `02:00 UTC`
- Manual trigger via `workflow_dispatch`

Core flow:

1. Checkout + setup + cache restore.
2. Train model via `python -m stock_screener.cli train-model`.
3. Evaluate promotion gate result from `models/ensemble/metrics.json`.
4. Upload model artifact only when promotion gates pass.

Notes:

- Weekly cadence is intentional.
- Promotion gates can be enforced/relaxed using repo vars and env settings.

## Reset Workflow

File: `.github/workflows/reset-portfolio-state.yml`

Trigger:

- Manual `workflow_dispatch` with `initial_cash_cad` and `dry_run`.

Core flow:

1. List/delete daily data caches.
2. Create fresh `screener_portfolio_state.json` and `.bak`.
3. Seed a reset cache key for next daily run.
4. Upload reset state artifact and summary.

## State, Caches, and Artifacts

- Portfolio state path in daily runs: `screener_portfolio_state.json`.
- State is also mirrored to `.bak` by runtime code.
- Daily artifacts include reports and state snapshot for audit/debugging.
- Reward logs/policy are stored under `cache/` by the pipeline.
- Mutable portfolio/data cache now uses restore/save semantics:
  - restore via `actions/cache/restore@v4`
  - save via `actions/cache/save@v4` with run-unique key suffix
  - restore key prefix remains stable for cross-run recovery

Important:

- Daily cache restore keys are hash-based and stable; save keys include run-unique suffixes.
- Detailed design and remaining risks are documented in `docs/PORTFOLIO_PIPELINE_AUDIT.md`.

## Required Secrets

- `EMAIL_USERNAME`
- `EMAIL_PASSWORD`
- Optional `EMAIL_TO` (defaults to `EMAIL_USERNAME`)
