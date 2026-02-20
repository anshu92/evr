# Daily Run Flow Chart (Complete Gates and Logic)

Last updated: 2026-02-20  
Primary sources:

- `.github/workflows/daily-stock-screener.yml`
- `stock_screener/pipeline/daily.py`
- `stock_screener/screening/screener.py`
- `stock_screener/portfolio/manager.py`

## 1) GitHub Actions Orchestration Flow

```mermaid
flowchart TD
    A[Trigger: cron 12:00/15:00/19:30 UTC or workflow_dispatch] --> B[Concurrency lock: daily-stock-screener]
    B --> C[Checkout + setup Python 3.11]
    C --> D[Restore caches: pip + data/state + models]
    D --> E[Find latest successful train artifact on default branch]
    E --> F{Artifact found?}
    F -- Yes --> G[Download artifact zip via GitHub API<br/>extract to models/ensemble]
    F -- No --> H[Skip model download]
    G --> I[Install dependencies]
    H --> I
    I --> J[Ensure model exists step]
    J --> K{models/ensemble/manifest.json exists?}
    K -- Yes --> L[Continue]
    K -- No + ALLOW_FALLBACK_TRAINING=1 --> M[Run fallback train-model]
    K -- No + ALLOW_FALLBACK_TRAINING=0 --> N[Continue without model<br/>daily pipeline ML block degrades gracefully]
    M --> L
    N --> L
    L --> O[Run: python -m stock_screener.cli daily --log-level INFO]
    O --> P[Collect telemetry json]
    P --> Q[Compute health counters from state/run_meta/reward logs/actions]
    Q --> R[Write $GITHUB_STEP_SUMMARY]
    R --> S[Save data/state cache with run-unique key]
    S --> T[Upload artifacts: reports + last_run_meta + portfolio state]
    T --> U[Determine session tag: Pre-Market / Mid-Day / Pre-Close]
    U --> V[Send email: always(), continue-on-error]
    V --> W{Workflow failed?}
    W -- Yes --> X[Create GitHub issue]
    W -- No --> Y[Done]
    X --> Y
```

## 2) `run_daily` End-to-End Flow

```mermaid
flowchart TD
    A[run_daily(cfg, logger)] --> B[Init paths/dirs/run_meta]
    B --> C[Fetch US + TSX universes]
    C --> C1[Runtime gate: universe]
    C1 --> D[Cap ticker list by MAX_TICKERS]
    D --> E[Fetch FX]
    E --> F[Download prices]
    F --> F1[Runtime gate: price_download]
    F1 --> G[Fetch fundamentals + macro]
    G --> H[Compute features]
    H --> H1[Runtime gate: feature_build]
    H1 --> I{USE_ML?}

    I -- No --> J[Skip ML, keep baseline features]
    I -- Yes --> K[ML inference block (fail-soft try/except)]
    K --> K1[Load metadata if present + apply target encodings]
    K1 --> K2[Normalize features for ML]
    K2 --> K3[Feature parity check<br/>strict mode can raise inside ML block]
    K3 --> K4{MODEL_PATH is manifest?}
    K4 -- No --> K5[Load single model + predict pred_return]
    K4 -- Yes --> K6[Load ensemble + uncertainty]
    K6 --> K7[Optional regime specialists + gating]
    K7 --> K8[Optional linear recalibration]
    K8 --> K9[Optional rank-preserving calibration map]
    K9 --> K10[Optional quantile models + LCB/spread]
    K10 --> K11[Predict peak days]
    K11 --> K12[Compute guarded ret_per_day:<br/>pred_return / (clamp(pred_peak_days)+k)]
    K12 --> K12A[Log ret_per_day + peak-day distribution stats]
    K12A --> K12B[Shift alert vs previous run stats]
    K5 --> K13[Resolve strategy mode gate<br/>ML | BASELINE | HOLD_ONLY]
    K12B --> K13
    K --> K13
    J --> K13
    K13 --> L[Score universe]

    L --> L1[score_universe filters: n_days, price, liquidity, vol cap]
    L1 --> M{sector_neutral_selection?}
    M -- Yes --> N[select_sector_neutral(top_n)]
    M -- No --> O[top_n by score]
    N --> P[Compute effective entry thresholds]
    O --> P
    P --> P1{dynamic entry thresholds enabled + enough names?}
    P1 -- Yes --> P2[Relax-only percentile thresholds with floors]
    P1 -- No --> P3[Use static entry thresholds]
    P2 --> Q[apply_entry_filters]
    P3 --> Q
    Q --> Q1[Entry gates: confidence, pred_return, vol, 5d momentum, momentum alignment]
    Q1 --> Q2[Runtime gate: screening]
    Q2 --> R{screened empty?}
    R -- Yes --> R1[target_weights empty]
    R -- No --> S[Build candidate weights]

    S --> S1{dynamic_portfolio_sizing?}
    S1 -- Yes --> S2[Compute dynamic portfolio size using IC + confidence + pred_return]
    S1 -- No --> S3[Use static portfolio_size]
    S2 --> S4{use_correlation_weights?}
    S3 --> S4
    S4 -- Yes --> S5[compute_correlation_aware_weights]
    S4 -- No --> S6[compute_inverse_vol_weights]

    R1 --> T[Load portfolio state]
    S5 --> T
    S6 --> T

    T --> T1[Staleness check + legacy placeholder migration reset]
    T1 --> U{unified_optimizer_enabled?}
    U -- Yes --> U1[optimize_unified_portfolio<br/>risk/turnover/cost/beta/corr constraints]
    U -- No --> U2[Sequential transforms<br/>confidence->uncertainty->conviction->liquidity->corr->beta->caps]
    U1 --> V[Instrument sleeve constraints]
    U2 --> V
    V --> W{regime_exposure_enabled?}
    W -- Yes --> W1[Apply regime exposure scalar]
    W -- No --> X
    W1 --> X{volatility_targeting?}
    X -- Yes --> X1[Apply portfolio volatility targeting]
    X -- No --> Y
    X1 --> Y{drawdown_management and pnl_history?}
    Y -- Yes --> Y1[Apply drawdown scalar]
    Y -- No --> Z
    Y1 --> Z{reward_model_enabled?}
    Z -- Yes --> Z1[Reward policy block (fail-soft try/except)<br/>load logs/policy, update IC/state, select action,<br/>scale exposure+conviction]
    Z -- No --> AA
    Z1 --> AA{target weights eliminated?}
    AA -- Yes --> AA1[Fallback rebuild top min(3, screened) weights]
    AA -- No --> AB
    AA1 --> AB[Build adaptive PortfolioManager params]
    AB --> AC[apply_exits(stateful)]
    AC --> AD[Persist exit events + state<br/>fail-stop on persistence error]
    AD --> AE[Rebalance controls/hysteresis]
    AE --> AF{unified optimizer active?}
    AF -- Yes --> AF1[Disable post-optimizer turnover shrinkage]
    AF -- No --> AF2[Allow turnover shrinkage]
    AF1 --> AG{strategy_mode == HOLD_ONLY?}
    AF2 --> AG
    AG -- Yes --> AG1[Override targets to current open holdings only]
    AG -- No --> AG2[Keep model-derived targets]
    AG1 --> AH
    AG2 --> AH
    AH[build_trade_plan with blocked_buys from same-run exits]
    AH --> AI[Merge exit actions + sanitize conflicts/dupes]
    AI --> AJ[Build holdings report frame from all open positions]
    AJ --> AK[render_reports]
    AK --> AK1[Runtime gate: reporting]
    AK1 --> AL{reward_model_enabled and log objects present?}
    AL -- Yes --> AL1[Save reward prediction/close entries (fail-soft)]
    AL -- No --> AM
    AL1 --> AL2[Save action-level reward entries (fail-soft)]
    AL2 --> AL3[Save reward policy (fail-soft)]
    AL3 --> AM[Write cache/last_run_meta.json]
    AM --> AN[Done]
```

## 3) Portfolio Exit Gate Order (`PortfolioManager.apply_exits`)

Exit rules are evaluated in strict order; first matching rule exits the position.

```mermaid
flowchart TD
    A[For each OPEN position with valid price] --> B[Compute trading days held]
    B --> C{days >= adjusted_hard_hold?}
    C -- Yes --> C1[SELL HARD_MAX_HOLD]
    C -- No --> D{days >= adjusted_max_hold and no extend signal?}
    D -- Yes --> D1[SELL SOFT_MAX_HOLD]
    D -- No --> E{peak_based_exit target reached?}
    E -- Yes --> E1[SELL PEAK_TARGET]
    E -- No --> F{stop-loss hit? (vol-adjusted or fixed)}
    F -- Yes --> F1[SELL STOP_LOSS]
    F -- No --> G{take-profit hit?}
    G -- Yes --> G1[SELL TAKE_PROFIT]
    G -- No --> H{quick-profit rule hit?}
    H -- Yes --> H1[SELL QUICK_PROFIT]
    H -- No --> I{low daily return and no pred_return override?}
    I -- Yes --> I1[SELL LOW_DAILY_RETURN]
    I -- No --> J{momentum decay from peak?}
    J -- Yes --> J1[SELL MOMENTUM_DECAY]
    J -- No --> K{age urgency enabled and below gain floor?}
    K -- Yes --> K1[SELL AGE_URGENCY]
    K -- No --> L{trailing stop hit?}
    L -- Yes --> L1[SELL TRAILING_STOP]
    L -- No --> M{signal decay (pred_return <= threshold)?}
    M -- Yes --> M1[SELL SIGNAL_DECAY]
    M -- No --> N{peak detection confirms >=2 signals?}
    N -- Yes --> N1[SELL_PARTIAL PEAK_* if min notional and cooldown satisfied]
    N -- No --> O[KEEP/HOLD]
    C1 --> P[Next position]
    D1 --> P
    E1 --> P
    F1 --> P
    G1 --> P
    H1 --> P
    I1 --> P
    J1 --> P
    K1 --> P
    L1 --> P
    M1 --> P
    N1 --> P
    O --> P
```

Notes:

- Holding periods use market-specific trading-day calendars (NYSE/TSX holidays).
- `adjusted_max_hold` and `adjusted_hard_hold` can be volatility-regime scaled.
- Peak target chooses earlier of entry-time and updated prediction; legacy positions use `updated_only` behavior.

## 4) Trade Plan Gate Order (`PortfolioManager.build_trade_plan`)

```mermaid
flowchart TD
    A[Normalize open lots + derive open/target sets] --> A1[Initialize per-ticker decision map<br/>EXIT/REDUCE/HOLD/INCREASE/ENTER]
    A1 --> B[Rotation sell pass for out-of-target opens]
    B --> B1[Rotation eligibility gates:<br/>not pre-decided EXIT/REDUCE,<br/>days_held >= rotation_cooldown_days]
    B1 --> C{Rotate?}
    C -- No --> D[Keep open]
    C -- Yes --> E[SELL with reason:<br/>ROTATION:NEG_PRED / LOW_RANK / NO_DATA]
    E --> F[Churn guards:<br/>min_trade_notional + min_rebalance_weight_delta]
    F --> G[Update state open/closed]
    G --> H[Buy pass for missing target names up to slot limit]
    H --> I[Buy gates:<br/>not blocked_buys,<br/>not pre-decided EXIT/REDUCE,<br/>valid price, min cash, min allocation]
    I --> J[Small-notional gate:<br/>allow one seed buy only if portfolio empty]
    J --> K[Create BUY action (+replaces_ticker for rotation linkage)]
    K --> L[Emit HOLD for remaining open names]
    L --> M[HOLD fields include pred_return and adaptive expected_sell_date]
    M --> N[Append pnl snapshot + append position events + save state]
    N --> O[Return TradePlan]
```

## 5) Rebalance Hysteresis and Turnover Gates (`_apply_rebalance_controls`)

### 5.1 Gate Logic

1. If no target weights: return unchanged.
2. Compute current live weights from open positions and prices.
3. If cold start (no current weights):
4. Keep only weights above `min_trade_notional`.
5. If none survive, keep top target as seed (if at least ~1 CAD), otherwise return empty.
6. If not cold start, compute per-ticker dynamic multiplier:
7. `dyn_mult = 1 + u_w*uncertainty_pct + l_w*illiquidity_pct + v_w*vol_stress`, clipped to `[dyn_min, dyn_max]`.
8. Compute thresholds:
9. `delta_threshold = min_rebalance_weight_delta * dyn_mult`
10. `notional_threshold = min_trade_notional_cad * dyn_mult`
11. If `|target-current| < delta_threshold` or trade notional below threshold: keep current weight.
12. Else move toward target weight.
13. Optional turnover shrinkage:
14. `effective = target - (turnover_penalty_bps * dyn_mult * 1e-4 * delta)` when enabled.
15. Tiny new positions below threshold are zeroed.
16. If everything is removed, fallback to feasible positive target weights.
17. Exposure policy enforcement:
18. Track `gross_exposure`, `net_exposure`, `cash_weight`.
19. Default policy `allow_cash_no_upscale`: keep under-invested weights as cash; only downscale if gross is infeasible (> target).
20. Optional policy `normalize_to_target_gross`: scale to configured gross target.
21. Emit deterministic gate audit with per-run reason-code counts and dropped ticker events.
22. Record `pre_rebalance -> final` drift (`L1`, `L2`, dropped/entered names, top absolute drifts).
23. Record `optimizer -> final` drift when unified optimizer is enabled.
24. Explicitly report names dropped by notional gates with reason codes.

### 5.2 Unified Optimizer Interaction

- When `unified_optimizer_enabled=True`, post-optimizer turnover shrinkage is disabled in `run_daily` to avoid duplicate turnover penalties.
- Hysteresis and notional gates still run.
- Workflow health summary surfaces projection drift (`pre L1/L2`, `optimizer L1/L2`) and notional-drop count.

## 6) Entry and Screening Gate Details

`score_universe` hard pre-filters:

- `n_days >= 90`
- `last_close_cad >= min_price_cad`
- `avg_dollar_volume_cad >= min_avg_dollar_volume_cad`
- `vol_60d_ann <= max_screen_volatility` (when configured)

Scoring branch priority:

- If ML columns exist: `ret_per_day` -> `pred_score` -> `pred_return`, blended with baseline score.
- `ret_per_day` is guarded before scoring:
  - clamp `pred_peak_days` into configured `[min_days, max_days]`
  - smooth denominator with `+k` to avoid small-day blowups
  - log run distribution stats + shift alerts vs previous run metadata
- Else baseline factors only.

`apply_entry_filters` gates:

- `pred_confidence >= min_confidence` (if available)
- `pred_return >= min_pred_return` (if available)
- `vol_60d_ann <= entry_max_volatility` (if available)
- `ret_5d >= entry_min_momentum_5d` (if available)
- Momentum alignment gate: reject bullish predictions with strongly bearish short-term momentum.

Dynamic entry threshold gate (`_compute_effective_entry_thresholds`):

- Activated only when enabled and candidate count >= `entry_dynamic_min_candidates`.
- Relax-only percentiles (never stricter than static threshold).
- Floors enforced (`entry_min_confidence_floor`, `entry_min_pred_return_floor`).
- Relaxation blocked when cross-sectional separation is weak:
  - require minimum confidence top-decile spread (`entry_dynamic_min_conf_top_decile_spread`)
  - require minimum predicted-return top-decile spread (`entry_dynamic_min_pred_top_decile_spread`)
- Market stress guard:
  - if vol stress (`market_vol_regime` mapped to `[0,1]`) exceeds `entry_stress_max_vol_stress` or
  - `market_breadth` is below `entry_stress_min_breadth`
  - then thresholds are tightened (`entry_stress_*_tighten_add`)
  - optional hard escalation: `ENTRY_STRESS_HOLD_ONLY_ENABLED=1` sets strategy mode to `HOLD_ONLY`

Strategy mode gate (`run_daily`):

- `ML`: `USE_ML=1` and ML predictions available.
- `BASELINE`: either `USE_ML=0`, or `USE_ML=1` with ML unavailable and `ALLOW_BASELINE_TRADING=1`.
- `HOLD_ONLY`: `USE_ML=1`, ML unavailable, and baseline fallback not allowed.
- In `HOLD_ONLY`, targets are overridden to currently open holdings only; no new buys are created.

## 7) Runtime Budget Gates

`_check_runtime_budget` checkpoints:

- `universe`
- `price_download`
- `feature_build`
- `screening`
- `reporting`

Behavior:

- Raises `TimeoutError` if elapsed minutes > `max_daily_runtime_minutes`.
- Warns when > 80% of budget.

## 8) Fail-Soft vs Fail-Stop Behavior

### 8.1 Fail-Soft (continue run)

- ML loading/inference/calibration exceptions: logs warning, then strategy-mode gate decides `HOLD_ONLY` or explicit baseline fallback.
- Reward policy/logging exceptions: non-fatal warnings.
- Email sending step: `continue-on-error: true`.

### 8.2 Fail-Stop (job fails)

- Runtime budget timeout from `_check_runtime_budget`.
- Exit-transition persistence failure (event append or state save after `apply_exits`).
- Unhandled exceptions outside guarded try/except blocks.
- Workflow-level failure triggers “Create GitHub Issue on Failure”.

## 9) Daily Outputs and Side Effects

Primary outputs:

- `reports/daily_report.txt`
- `reports/daily_email.html`
- `reports/portfolio_weights.csv`
- `reports/trade_actions.json`
- `cache/last_run_meta.json`
- `screener_portfolio_state.json` (and `.bak` managed by state module)
- `screener_portfolio_state.json.events.jsonl` (append-only position-changing action log)

Workflow telemetry outputs:

- `reports/telemetry/actions_telemetry.json`
- `$GITHUB_STEP_SUMMARY` health counters
- Actions artifact bundle (`daily-stock-screener-<run_number>`)
