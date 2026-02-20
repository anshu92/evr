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
- `.github/workflows/prune-actions-caches.yml`
- `.github/workflows/verify-daily-session-coverage.yml`
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
- Emits telemetry + workflow health counters to summary (`.github/workflows/daily-stock-screener.yml:252`, `.github/workflows/daily-stock-screener.yml:281`, `.github/workflows/daily-stock-screener.yml:392`)
- Uploads run artifacts, sends email, opens issue on failure (`.github/workflows/daily-stock-screener.yml:405`, `.github/workflows/daily-stock-screener.yml:439`, `.github/workflows/daily-stock-screener.yml:455`)

Training workflow (`.github/workflows/train-stock-screener-model.yml`):

- Weekly Sunday retraining (`.github/workflows/train-stock-screener-model.yml:8`)
- Trains model and evaluates promotion gates (`.github/workflows/train-stock-screener-model.yml:58`, `.github/workflows/train-stock-screener-model.yml:107`)
- Uploads model artifact only when promotion passes (`.github/workflows/train-stock-screener-model.yml:133`)

Reset workflow (`.github/workflows/reset-portfolio-state.yml`):

- Manual cache purge + fresh state seeding (`.github/workflows/reset-portfolio-state.yml:31`, `.github/workflows/reset-portfolio-state.yml:69`, `.github/workflows/reset-portfolio-state.yml:93`)

Cache prune workflow (`.github/workflows/prune-actions-caches.yml`):

- Weekly and manual cache GC for run-unique daily data keys
- Keeps recent keys and removes stale/overflow entries

Session coverage workflow (`.github/workflows/verify-daily-session-coverage.yml`):

- Weekday post-close verification of expected daily session windows
- Validates at least one successful run in each window (`PRE_MARKET`, `MID_DAY`, `PRE_CLOSE`)
- Opens an issue if any expected session window is missing

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
| High | Holding-period settings were not enforced as explicit tenure exits before other rules. | `stock_screener/portfolio/manager.py` exit path before this fix | Positions could stay open too long and capital could recycle slowly. | Fixed in this revision (`SOFT_MAX_HOLD(...)` + `HARD_MAX_HOLD(...)`). |
| Medium | Peak-target logic and HOLD guidance were inconsistent for legacy positions missing `entry_pred_peak_days`. | `stock_screener/portfolio/manager.py` peak-target + HOLD expected-sell path before this fix | Guidance could disagree with execution timing. | Fixed in this revision (`updated_only` actionable boundary and aligned HOLD sell-date semantics). |
| Medium | Model artifact lookup scanned only 5 runs and did not explicitly prefer default branch. | `.github/workflows/daily-stock-screener.yml` artifact lookup script before this fix | Possible suboptimal artifact selection. | Fixed in this revision (default-branch filter + paginated search). |
| Medium | Run-unique daily cache keys had no automated pruning strategy. | Daily cache save key includes `${{ github.run_id }}` | Unbounded cache growth and avoidable Actions storage pressure. | Fixed in this revision (`prune-actions-caches.yml`). |
| Medium | Workflow summary lacked state/reward/action health counters. | Prior daily workflow summary behavior | Slower operational triage when runs degrade. | Fixed in this revision (`$GITHUB_STEP_SUMMARY` counters). |
| Medium | No automated check ensured each weekday daily session window completed successfully. | Prior workflow set had no post-close coverage verifier | Silent missed runs could leave the day partially unmanaged. | Fixed in this revision (`verify-daily-session-coverage.yml`). |
| Medium | Documentation drift existed around workflow count and exit behavior. | Prior docs state | Misaligned operational expectations. | Fixed in this revision (`README.md`, `docs/GITHUB_WORKFLOWS.md`, this audit). |

## Logical Improvements (Current State)

Implemented in this revision:

1. Daily mutable cache switched to restore/save pattern with run-unique save key.
2. Soft and hard tenure exits enforced in `PortfolioManager.apply_exits`.
3. Updated-only peak exit boundary added for legacy positions without entry peak metadata.
4. HOLD expected-sell-date logic aligned with updated-only peak semantics.
5. Model artifact lookup hardened to default branch and paginated run search.
6. Daily workflow now emits state/reward/action health counters in `$GITHUB_STEP_SUMMARY`.
7. Weekly/manual cache-prune workflow added for run-unique daily cache keys.
8. Weekday session-coverage verifier added for `PRE_MARKET` / `MID_DAY` / `PRE_CLOSE` windows.
9. Workflow docs aligned to current behavior.

Implemented in follow-up revision (2026-02-20):

10. Entry confirmation now supports adaptive, relax-only percentile thresholds (`entry_dynamic_*`) with hard floors to avoid over-filtering when prediction distributions compress.
11. Dynamic portfolio sizing now reuses effective entry thresholds so sizing aggressiveness stays aligned with the active entry gate.
12. Portfolio breadth ceiling widened (config cap `12`) and daily workflow defaults updated to `PORTFOLIO_SIZE=8`, `DYNAMIC_SIZE_MAX_POSITIONS=8`, `WEIGHT_CAP=0.20` to reduce structural under-investment.
13. `LOW_DAILY_RETURN` exits now have a residual-alpha guard (`low_daily_return_hold_min_pred_return`) so positions are not force-exited when live model signal remains strong.
14. Targeted regression tests added for adaptive entry thresholds, widened portfolio-size cap behavior, and residual-alpha low-daily-return override semantics.
15. Instrument sleeve constraints added to portfolio construction (`instrument_fund_max_weight`, `instrument_equity_min_weight`) so ETF/fund exposure is capped and equity sleeve retains minimum allocation.
16. Daily workflow health summary now reports adaptive entry thresholds and instrument sleeve shifts from `cache/last_run_meta.json` for easier production monitoring.
17. Screening ML path no longer applies an extra `ret_per_day / vol` divisor when `ret_per_day` is available; this removes duplicated per-name volatility penalization while keeping hard vol caps and portfolio-level risk controls (`stock_screener/screening/screener.py`, `tests/test_scoring.py`, `docs/RESEARCH_GROUNDED_SIMPLIFICATION.md`).
18. With unified optimizer enabled, post-optimizer rebalance no longer applies an additional turnover shrinkage penalty; hysteresis and min-notional gates remain active to avoid duplicate turnover suppression (`stock_screener/pipeline/daily.py`, `tests/test_rebalance_hysteresis_dynamic.py`, `docs/RESEARCH_GROUNDED_SIMPLIFICATION.md`).

Remaining recommended improvements:

1. Optional: route session-coverage alerts to chat/incident tooling in addition to GitHub issues.
