# Portfolio Pipeline Audit (2026-02-19)

## Scope

This document audits the end-to-end portfolio management pipeline and its GitHub Actions orchestration, with a focus on:

- Daily portfolio lifecycle and trade decision flow
- Cross-run state persistence
- ML model handoff from training to daily execution
- Operational and logic risks

Primary code/workflow references:

- `.github/workflows/daily-stock-screener.yml`
- `.github/workflows/train-stock-screener-model.yml`
- `.github/workflows/reset-portfolio-state.yml`
- `stock_screener/pipeline/daily.py`
- `stock_screener/portfolio/manager.py`
- `stock_screener/portfolio/state.py`

## End-to-End Flow

### 1) GitHub Triggers and Runtime Orchestration

Daily workflow (`.github/workflows/daily-stock-screener.yml`):

- Scheduled weekday runs at `12:00`, `15:00`, and `19:30` UTC (`.github/workflows/daily-stock-screener.yml:19`, `.github/workflows/daily-stock-screener.yml:22`, `.github/workflows/daily-stock-screener.yml:25`)
- Restores caches for pip, data/state, and models (`.github/workflows/daily-stock-screener.yml:49`, `.github/workflows/daily-stock-screener.yml:58`, `.github/workflows/daily-stock-screener.yml:70`)
- Pulls latest train artifact by querying workflow runs (`.github/workflows/daily-stock-screener.yml:80`)
- Executes `python -m stock_screener.cli daily` (`.github/workflows/daily-stock-screener.yml:227`)
- Uploads run artifacts, sends email, opens issue on failure (`.github/workflows/daily-stock-screener.yml:258`, `.github/workflows/daily-stock-screener.yml:281`, `.github/workflows/daily-stock-screener.yml:297`)

Training workflow (`.github/workflows/train-stock-screener-model.yml`):

- Weekly Sunday retraining (`.github/workflows/train-stock-screener-model.yml:8`)
- Trains model and evaluates promotion gates (`.github/workflows/train-stock-screener-model.yml:58`, `.github/workflows/train-stock-screener-model.yml:107`)
- Uploads model artifact only when promotion passes (`.github/workflows/train-stock-screener-model.yml:133`)

Reset workflow (`.github/workflows/reset-portfolio-state.yml`):

- Manual cache purge + fresh state seeding (`.github/workflows/reset-portfolio-state.yml:31`, `.github/workflows/reset-portfolio-state.yml:69`, `.github/workflows/reset-portfolio-state.yml:93`)

### 2) Daily Pipeline Execution (`run_daily`)

Stage order in `stock_screener/pipeline/daily.py`:

1. Load config and runtime budget guard (`stock_screener/pipeline/daily.py:585`, `stock_screener/pipeline/daily.py:229`)
2. Fetch US/TSX universe, FX, prices, fundamentals, macro, then build features (`stock_screener/pipeline/daily.py:601`, `stock_screener/pipeline/daily.py:618`, `stock_screener/pipeline/daily.py:625`, `stock_screener/pipeline/daily.py:634`, `stock_screener/pipeline/daily.py:640`, `stock_screener/pipeline/daily.py:644`)
3. Optional ML path: load ensemble + metadata, validate feature parity, predict returns/uncertainty, apply calibration/regime/quantile layers (`stock_screener/pipeline/daily.py:655`, `stock_screener/pipeline/daily.py:699`, `stock_screener/pipeline/daily.py:709`)
4. Score and filter universe, sector-neutral selection, entry filters (`stock_screener/pipeline/daily.py:890`, `stock_screener/pipeline/daily.py:903`, `stock_screener/pipeline/daily.py:916`)
5. Dynamic portfolio sizing and initial weights (`stock_screener/pipeline/daily.py:940`, `stock_screener/pipeline/daily.py:971`)
6. Load persistent state, staleness check, migration safeguard (`stock_screener/pipeline/daily.py:981`, `stock_screener/pipeline/daily.py:994`, `stock_screener/pipeline/daily.py:1013`)
7. Optimization and exposure controls (unified optimizer, regime, vol targeting, drawdown, reward-policy scalars) (`stock_screener/pipeline/daily.py:1061`, `stock_screener/pipeline/daily.py:1187`, `stock_screener/pipeline/daily.py:1204`, `stock_screener/pipeline/daily.py:1231`, `stock_screener/pipeline/daily.py:1417`)
8. Portfolio actions: exits, no-trade-band rebalance controls, trade plan generation (`stock_screener/pipeline/daily.py:1502`, `stock_screener/pipeline/daily.py:1529`, `stock_screener/pipeline/daily.py:1558`)
9. Report rendering, reward/action log writes, policy save, run metadata save (`stock_screener/pipeline/daily.py:1730`, `stock_screener/pipeline/daily.py:1745`, `stock_screener/pipeline/daily.py:1857`, `stock_screener/pipeline/daily.py:1975`, `stock_screener/pipeline/daily.py:1983`)

### 3) Portfolio State and Trade Logic

State I/O reliability:

- JSON read/write with lock and atomic replace (`stock_screener/portfolio/state.py:49`, `stock_screener/portfolio/state.py:82`, `stock_screener/portfolio/state.py:139`, `stock_screener/portfolio/state.py:217`)

Action lifecycle in `PortfolioManager`:

- Exit pass first (sell/partial sell/hold keep list) (`stock_screener/portfolio/manager.py:685`)
- Rotation sells for out-of-target names with churn controls (`stock_screener/portfolio/manager.py:986`)
- Buys for missing target names with fractional-share support (`stock_screener/portfolio/manager.py:1078`)
- HOLD actions emitted for untouched open positions (`stock_screener/portfolio/manager.py:1193`)
- PnL snapshot appended and state persisted (`stock_screener/portfolio/manager.py:1286`)

## Issues Identified

| Severity | Issue | Evidence | Impact | Status |
|---|---|---|---|
| Critical | Mutable state was cached under a stable exact key via a single `actions/cache@v4` step, risking stale state on exact-key hits. | Daily workflow cache strategy plus state staleness guard in `stock_screener/pipeline/daily.py:1000` | Portfolio state/reward state could appear frozen across runs. | Fixed in this revision (restore/save split + run-unique save key). |
| High | Holding-period settings (`MAX_HOLDING_DAYS`, `MAX_HOLDING_DAYS_HARD`) were not enforced as hard exits in `apply_exits`. | `stock_screener/portfolio/manager.py` exit path before this fix | Positions could stay open indefinitely if other exits did not trigger. | Fixed in this revision (`HARD_MAX_HOLD(...)` liquidation rule). |
| Medium | Peak-target logic could fail to act for legacy positions missing `entry_pred_peak_days`. | `stock_screener/portfolio/manager.py` peak-target branch before this fix | Legacy holdings might never trigger updated-only peak exits. | Fixed in this revision (`updated_only` actionable boundary). |
| Medium | Model artifact lookup scanned only 5 runs and did not explicitly prefer default branch. | `.github/workflows/daily-stock-screener.yml` artifact lookup script before this fix | Possible suboptimal artifact selection. | Fixed in this revision (default-branch filter + paginated search). |
| Medium | Documentation drift existed around workflow count and exit behavior. | Prior docs state | Misaligned operational expectations. | Fixed in this revision (`README.md`, `docs/GITHUB_WORKFLOWS.md`). |

## Logical Improvements (Current State)

Implemented in this revision:

1. Daily mutable cache switched to restore/save pattern with run-unique save key.
2. Hard max-hold liquidation enforced in `PortfolioManager.apply_exits`.
3. Updated-only peak exit boundary added for legacy positions without entry peak metadata.
4. Model artifact lookup hardened to default branch and paginated run search.
5. Workflow/README docs aligned to current behavior.

Remaining recommended improvements:

1. Add `concurrency` group to the daily workflow to prevent schedule overlap.
2. Emit explicit state-age/reward-log counters in workflow summary for easier ops triage.
3. Consider a soft max-hold rule (`SOFT_MAX_HOLD`) as a second-layer tenure guardrail.
