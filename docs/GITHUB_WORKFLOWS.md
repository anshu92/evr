## GitHub Workflows

This repository currently uses five GitHub Actions workflows:

- `daily-stock-screener.yml`: weekday daily portfolio run and email
- `train-stock-screener-model.yml`: weekly model training/promotion
- `reset-portfolio-state.yml`: manual cache/state reset utility
- `prune-actions-caches.yml`: weekly cache GC for run-unique daily data keys
- `verify-daily-session-coverage.yml`: weekday self-check for missed daily session windows

Detailed pipeline audit: `docs/PORTFOLIO_PIPELINE_AUDIT.md`
Complete daily flow chart: `docs/DAILY_RUN_FLOW_CHART.md`

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
6. Compute state/reward/action health counters and write `$GITHUB_STEP_SUMMARY` (including strategy mode, adaptive entry thresholds, ret_per_day distribution/shift telemetry, instrument sleeve shifts, and weight-projection drift/notional-drop telemetry).
7. Upload run artifacts (`reports/`, `cache/last_run_meta.json`, `screener_portfolio_state.json`, `screener_portfolio_state.json.events.jsonl`).
8. Email the HTML report and CSV/JSON attachments.
9. Open a GitHub issue if the workflow fails.

Runtime/controls:

- `MAX_DAILY_RUNTIME_MINUTES=12`
- `STRICT_FEATURE_PARITY=1`
- `USE_ML=1` with explicit strategy-mode gate when ML is unavailable:
  - default `HOLD_ONLY` (no new buys)
  - `BASELINE` only when `ALLOW_BASELINE_TRADING=1`
- Portfolio construction defaults: `PORTFOLIO_SIZE=8`, `DYNAMIC_SIZE_MAX_POSITIONS=8`, `WEIGHT_CAP=0.20`
- Instrument sleeve constraints enabled by default (`INSTRUMENT_FUND_MAX_WEIGHT=0.35`, `INSTRUMENT_EQUITY_MIN_WEIGHT=0.50`)
- Adaptive entry thresholds enabled (`ENTRY_DYNAMIC_THRESHOLDS_ENABLED=1`) with percentile-based relax-only floors
- Dynamic entry guardrails enabled:
  - cross-sectional spread minimums (`ENTRY_DYNAMIC_MIN_CONF_TOP_DECILE_SPREAD`, `ENTRY_DYNAMIC_MIN_PRED_TOP_DECILE_SPREAD`)
  - stress guard tightening on weak regime/breadth (`ENTRY_STRESS_*`)
  - optional hard entry freeze via `ENTRY_STRESS_HOLD_ONLY_ENABLED=1`
- Dynamic no-trade-band and turnover controls are enabled by default
- Rotation churn guard enabled (`ROTATION_COOLDOWN_DAYS=2`)
- Exposure policy defaults to explicit cash-aware control (`EXPOSURE_POLICY=allow_cash_no_upscale`, `TARGET_GROSS_EXPOSURE=1.0`, `ALLOW_LEVERAGE=0`)
- Concurrency guard enabled: one daily run at a time (`concurrency.group: daily-stock-screener`)

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
2. Create fresh `screener_portfolio_state.json`, `.bak`, and empty `.events.jsonl`.
3. Seed a reset cache key for next daily run.
4. Upload reset state artifact and summary.

## Cache Prune Workflow

File: `.github/workflows/prune-actions-caches.yml`

Schedule/trigger:

- Weekly on Sunday `03:15 UTC`
- Manual `workflow_dispatch` with `dry_run`, `keep_latest`, and `max_age_days`

Core flow:

1. List all Actions caches and filter daily mutable keys (`-daily-screener-data-`).
2. Keep most-recent keys and remove stale/overflow keys.
3. Write prune metrics to `$GITHUB_STEP_SUMMARY`.

## Session Coverage Workflow

File: `.github/workflows/verify-daily-session-coverage.yml`

Schedule/trigger:

- Weekdays at `22:15 UTC` (after the pre-close run should have finished)
- Manual `workflow_dispatch` with optional `date_utc` override

Core flow:

1. Pull completed successful runs for `daily-stock-screener.yml` on the target UTC day.
2. Bucket successful runs into expected windows: `PRE_MARKET`, `MID_DAY`, `PRE_CLOSE`.
3. Fail if any window is missing and optionally open an ops issue.
4. Write coverage metrics to `$GITHUB_STEP_SUMMARY`.

## State, Caches, and Artifacts

- Portfolio state path in daily runs: `screener_portfolio_state.json`.
- State is mirrored to `.bak` by runtime code.
- Position-changing events are appended to `screener_portfolio_state.json.events.jsonl`.
- Daily artifacts include reports and state snapshot for audit/debugging.
- Reward logs/policy are stored under `cache/` by the pipeline.
- Mutable portfolio/data cache now uses restore/save semantics:
  - restore via `actions/cache/restore@v4`
  - save via `actions/cache/save@v4` with run-unique key suffix
  - restore key prefix remains stable for cross-run recovery
- Weekly cache GC prevents unbounded growth of run-unique daily cache entries.

Important:

- Daily cache restore keys are hash-based and stable; save keys include run-unique suffixes.
- Detailed design and remaining risks are documented in `docs/PORTFOLIO_PIPELINE_AUDIT.md`.

## Required Secrets

- `EMAIL_USERNAME`
- `EMAIL_PASSWORD`
- Optional `EMAIL_TO` (defaults to `EMAIL_USERNAME`)
