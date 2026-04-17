# EVR: End-to-End System Architecture Report

**Stack**: Python 3.11 · XGBoost/LightGBM · Multi-Agent LLM · GitHub Actions  
**Portfolio**: $500 CAD paper trading, US + TSX equities  
**Strategy**: LLM-Primary decision engine with ML as fallback, quantitative circuit breakers  
**Last updated**: 2026-04-03

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Phase 1: Data Ingestion](#phase-1-data-ingestion)
3. [Phase 2: Feature Engineering](#phase-2-feature-engineering)
4. [Phase 3: ML Pipeline](#phase-3-ml-pipeline)
5. [Phase 4: LLM Multi-Agent System](#phase-4-llm-multi-agent-system)
6. [Phase 5: LLM-Primary Decision Mode](#phase-5-llm-primary-decision-mode)
7. [Phase 6: Quantitative Circuit Breakers](#phase-6-quantitative-circuit-breakers)
8. [Phase 7: Portfolio Manager — Stateful Exit System](#phase-7-portfolio-manager--stateful-exit-system)
9. [Phase 8: Reward Feedback Loop](#phase-8-reward-feedback-loop)
10. [Phase 9: Reporting](#phase-9-reporting)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Data Flow: Signal → Action](#data-flow-signal--action)
13. [Architecture Assessment](#architecture-assessment)
14. [Configuration Reference](#configuration-reference)
15. [File Map](#file-map)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXECUTION ENVIRONMENT                         │
│                        GitHub Actions (free)                         │
│  Daily: 9 PM EST post-close + 7:30 AM EST pre-market (weekdays)     │
│  Intraday: 3 PM EST (weekdays) · Macro: 10 AM / 12 PM EST (weekdays) │
│  Training: Sunday 9 PM ET weekly                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA INGESTION LAYER                           │
│  yfinance (prices/fundamentals/macro/news)  ·  TSX Directory API     │
│  Reddit (r/wsb, r/stocks, r/CongressStockWatcher)  ·  Finnhub        │
│  RSS (MarketWatch, Seeking Alpha)  ·  CBOE VIX  ·  US Treasuries    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE ENGINEERING                            │
│  70+ per-ticker features  ·  134 model input features                │
│  Cross-sectional normalization  ·  Sector-relative ranks             │
│  VADER sentiment  ·  Insider transaction signals                     │
└─────────────────────────────────────────────────────────────────────┘
                    │                    │
          ┌─────────┘                    └─────────┐
          ▼                                        ▼
┌──────────────────────┐            ┌──────────────────────────────┐
│     ML PIPELINE      │            │      LLM MULTI-AGENT SYSTEM  │
│  XGBoost × 2        │            │  Technical Analyst            │
│  LightGBM × 2       │            │  Fundamental Analyst          │
│  Quantile q10/50/90 │            │  Sentiment Analyst            │
│  Regime specialists  │            │  Bull ↔ Bear Debate           │
│  Peak-day model      │            │  Risk Management Team         │
│  pred_return +       │            │  Portfolio Manager            │
│  pred_confidence     │            │  (Gemini/Groq/OpenRouter)     │
└──────────────────────┘            └──────────────────────────────┘
                    │                    │
                    └────────┬───────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE (LLM-PRIMARY)                      │
│  LLM selects tickers  →  LLM assigns weights (SMALL/MEDIUM/FULL)    │
│  Fallback: ML pipeline runs identically if LLM fails                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   QUANTITATIVE CIRCUIT BREAKERS                      │
│  Market Gate  ·  R:R Filter  ·  Vol Targeting  ·  Drawdown Scalar   │
│  Regime Exposure  ·  Correlation Limits  ·  Beta Constraints         │
│  Max 20% / Min 2% per position  ·  Liquidity floors                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PORTFOLIO MANAGER                                │
│  Stateful positions (atomic JSON + JSONL event log)                  │
│  12-rule exit system  ·  Rotation  ·  Scaling (50%+50%)             │
│  Reward feedback (Bayesian bandit)                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        REPORTING                                     │
│  HTML email (Gmail SMTP)  ·  CSV weights  ·  Text report            │
│  GitHub Actions summary  ·  Failure → GitHub Issue                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Ingestion

```
┌──────────────────────────────────────────────────────────────────┐
│  data/ module                                                     │
│                                                                   │
│  prices.py ──────── yfinance.download() ──► price history 180d   │
│                      batched (50/req), retry+backoff               │
│                      OHLCV in USD → converted to CAD              │
│                                                                   │
│  fundamentals.py ── yf.Ticker().info ──────► per-ticker dict     │
│                      7-day file cache (TTL)                        │
│                      beta, P/E, ROE, D/E, margins, market cap     │
│                                                                   │
│  fx.py ────────────  USDCAD=X ────────────► single float rate    │
│                                                                   │
│  macro.py ─────────  ^VIX, ^VIX3M ────────► vix, vix_term_slope  │
│                      ^TNX, ^IRX ───────────► yield_curve_slope    │
│                      Derived: vix_change_5d, vix_percentile_1y   │
│                                                                   │
│  news.py ──────────  yf.Ticker().news ─────► VADER compound score │
│                      aggregated: sentiment_avg, pos_ratio          │
│                      24h file cache                                │
│                                                                   │
│  news_sources.py ──  Multi-source aggregator:                     │
│    RSS: MarketWatch, Seeking Alpha headlines                       │
│    Reddit: r/wsb, r/stocks (ticker mentions → bullish signal)     │
│    Reddit: r/CongressStockWatcher (congressional trade filings)   │
│    Finnhub: FINNHUB_API_KEY (optional, premium quality)           │
│    → fetch_ticker_news_multi(): ranked by source priority         │
│    → get_wildcard_tickers(): up to 3 Reddit/news injection tickers│
│    → get_congress_traded_tickers(): inject legislator-traded tickers│
└──────────────────────────────────────────────────────────────────┘
```

### Data Sources

| Source | What | How |
|---|---|---|
| **yfinance** | Prices (OHLCV), fundamentals, news, FX, macro | Primary data library; batched downloads with retry/backoff |
| **TSX Directory API** | TSX ticker list | `https://www.tsx.com/json/company-directory/search` |
| **VADER** | News sentiment scoring | `vaderSentiment` library |
| **MarketWatch RSS** | Top financial headlines | `https://feeds.content.dowjones.io/public/rss/mw_topstories` |
| **Seeking Alpha RSS** | Market headlines | `https://seekingalpha.com/market_currents.xml` |
| **Reddit (wsb, stocks)** | Retail sentiment, wildcard tickers | Reddit JSON API with User-Agent |
| **Reddit (CongressStockWatcher)** | Congressional trading signals | Same subreddit scrape |
| **Finnhub** | Ticker news (optional) | `FINNHUB_API_KEY` required |
| **^VIX, ^VIX3M** | CBOE volatility indices | yfinance |
| **^TNX, ^IRX** | 10Y/3M Treasury yields | yfinance |
| **USDCAD=X** | FX rate | yfinance |

**Key design decisions:**
- All prices converted to CAD at point of ingestion — portfolio is CAD-denominated
- Fundamentals are cached 7 days to avoid hammering yfinance rate limits
- News/sentiment is fail-soft — returns NaN if unavailable, not a pipeline error
- Congressional trades and Reddit mentions inject tickers into the universe, not just score them

---

## Phase 2: Feature Engineering

```
features/technical.py: compute_features(prices, fx, fundamentals, macro)
────────────────────────────────────────────────────────────────────
Input:  MultiIndex DataFrame (ticker × {Open,High,Low,Close,Volume})
Output: DataFrame indexed by ticker, 70+ columns (all CAD)

Feature groups computed per ticker:

RETURNS               ret_5d, ret_10d, ret_20d, ret_60d, ret_120d
                      ret_accel_20_120 (momentum acceleration)

VOLATILITY            vol_20d_ann, vol_60d_ann
                      vol_ratio_20_60 (vol regime change signal)
                      vol_anom_30d (vol spike detector)

TECHNICALS            rsi_14 (EMA-smoothed)
                      ma20_ratio, ma50_ratio, ma200_ratio (price/MA)
                      drawdown_60d, dist_52w_high, dist_52w_low

MOMENTUM QUALITY      momentum_reversal = ret_5d - ret_20d
                      ret_consistency_20d (% of up days)
                      up_days_ratio_20d
                      ret_5d_sharpe, ret_20d_sharpe, ret_60d_sharpe

VOLUME                volume_momentum_20d (vol vs 60d avg)
                      volume_surge_5d (5d spike ratio)
                      price_volume_div (divergence signal)
                      avg_dollar_volume_cad (liquidity proxy)
                      amihud_illiquidity_20d
                      spread_estimate_cs (Corwin-Schultz bid-ask)
                      mfi_14 (Money Flow Index)
                      vwma_20_ratio (VWAP proxy)

INTRADAY/OVERNIGHT    overnight_ret_5d, intraday_ret_5d
                      overnight_intraday_ratio

MEAN REVERSION        ma20_zscore, mean_reversion_signal

CROSS-SECTION RANKS   rank_ret_5d .. rank_ma20_zscore (global ranks)
SECTOR RANKS          sector_rank_ret_20d, sector_breadth_20d,
                      sector_momentum_dispersion

FX / MARKET           fx_ret_5d, fx_ret_20d, is_tsx
                      market_vol_regime, market_trend_20d,
                      market_breadth, market_momentum_accel

MACRO                 vix, treasury_10y, treasury_13w,
                      yield_curve_slope, vix_term_slope,
                      vix_change_5d, vix_percentile_1y

INTERACTIONS          sharpe_x_rank, momentum_vol_interaction,
                      rsi_momentum_interaction, size_momentum_interaction

FUNDAMENTALS (134 → model input, computed from fundamentals.py + yfinance):
                      log_market_cap, beta, sector_target_enc,
                      value_score, quality_score, growth_score,
                      pe_discount, roc_growth, value_momentum
```

**Key design decisions:**
- Cross-sectional normalization (`normalize_features_cross_section`) at inference: zscore within the current batch, rank features are already relative
- Sector-relative ranks prevent value traps (a cheap stock in a cheap sector isn't necessarily cheap)
- Target encoding for sector/industry is expanding-window (no future leakage) computed during training, saved to manifest, applied at inference
- All rank features use `pd.rank(pct=True)` — robust to extreme outliers

---

## Phase 3: ML Pipeline

```
Training (weekly, GitHub Actions):
─────────────────────────────────

Historical panel (730d lookback)
        │
        ▼
compute_ticker_features() per (ticker, date)
  └─ same feature set as inference, computed on each bar
        │
        ▼
Label generation:
  target = max_return over next 5 days (peak return)
  - market_relative: alpha = stock_ret - SPY_ret
  - cost_adjusted: subtract 3 bps round-trip
        │
        ▼
Walk-forward splits (n=3, val_window=60d, embargo=10d)
  ├─ Train on past, validate on future (no leakage)
  └─ Embargo gap prevents label lookahead
        │
        ▼
Hyperparameter search (Optuna, 8 trials, 90s):
  XGBoost (×2): rank:pairwise, eta, max_depth, subsample, colsample
  LightGBM (×2): rank_xendcg, n_estimators, num_leaves, feature_fraction
  Quantile models (×3 for q10/q50/q90): separate LightGBM regressors
  Peak-day model: regressor predicting days to peak return
  Regime specialists (×3): bull/neutral/bear separate XGB ensembles
        │
        ▼
Promotion gates (blocks deployment if any fail):
  - IC/day ≥ 0.0002 (information content present)
  - Sharpe (cost-adj) ≥ 0.5
  - Max drawdown ≤ -25%
  - Consistency (% IC>0 days) ≥ 55%
  - Turnover efficiency ≥ 20%
  - Rebalance hysteresis ≥ 35%
        │
        ▼
Artifacts saved to models/ensemble/:
  manifest.json (metadata + calibration map + target encodings)
  xgb_model_0.json, xgb_model_1.json
  lgbm_model_0.bin, lgbm_model_1.bin
  quantile_q10.bin, quantile_q50.bin, quantile_q90.bin
  peak_model.bin
  bull_*.bin, neutral_*.bin, bear_*.bin (regime specialists)

─────────────────────────────────────────────────────────────────────
Inference (daily, at screening time):
─────────────────────────────────────

load_ensemble(manifest.json)
        │
        ▼
predict_ensemble_with_uncertainty(features)
  ├─ weighted average of 4 base models
  │   weights = reward feedback (re-weighted by online IC)
  ├─ pred_uncertainty = std of model outputs (disagreement)
  └─ pred_confidence = 1 - normalized_uncertainty
        │
        ▼ (if regime_specialist_enabled)
compute_regime_gate_weights(market_vol, market_trend, market_breadth)
  → [w_bull, w_neutral, w_bear] via softmax on tanh-normalized signals
predict_regime_gated() = w_bull*bull + w_neutral*neutral + w_bear*bear
        │
        ▼ (if quantile_models_enabled)
predict_quantile_lcb(features)
  LCB = q50 - 0.5 * (q90 - q10)   ← risk-adjusted lower bound
  Used to penalize high-uncertainty predictions
        │
        ▼
predict_peak_days(features) → trading days to predicted peak return
ret_per_day = pred_return / clamp(pred_peak_days + 1, 1, 10)
        │
        ▼
Final score:
  score = 0.60 * zscore(ret_per_day) + 0.40 * baseline_score
  baseline = 0.60*ret_60d + 0.35*ret_120d + 0.10*ma20_ratio
            + 0.10*log_dollar_vol - 0.35*vol_60d
```

**Why this model design:**
- Learning-to-rank (not regression) is well-suited to the problem: we care about relative ordering, not exact price levels
- Peak return target: predicting when to buy and hold to peak is more actionable than end-of-period returns
- LCB scoring: conservative estimate that penalizes model disagreement — never bet full confidence when models disagree
- Regime specialists: bull/bear/neutral ensembles soft-gated by current regime avoids a single model that must work in all conditions

---

## Phase 4: LLM Multi-Agent System

```
Per-ticker analysis: analyze_ticker(ticker, features, news, context)
─────────────────────────────────────────────────────────────────────

Context passed to every agent:
  - ML signals: pred_return, pred_confidence, pred_peak_days, score
  - Technical snapshot: ret_5d/10d/20d/60d, rsi_14, ma20_ratio,
    vol_60d_ann, volume_surge_5d, market_vol_regime
  - Fundamental snapshot: beta, P/E, ROE, D/E, market_cap
  - News: up to 5 recent headlines with dates + VADER sentiment
  - Congressional trading activity (if any)
  - Portfolio context: current holdings, cash available

PHASE 1: Specialized Analysts (4 FAST LLM calls)
───────────────────────────────────────────────────────────────────────
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ Technical Analyst  │  │Fundamental Analyst │  │Sentiment Analyst   │
│ SIGNAL: BUL/NEU/   │  │ SIGNAL: BUL/NEU/   │  │ SIGNAL: BUL/NEU/   │
│         BEAR       │  │         BEAR       │  │         BEAR       │
│ CONVICTION: H/M/L  │  │ CONVICTION: H/M/L  │  │ CONVICTION: H/M/L  │
│ KEY_LEVEL: $X.XX   │  │ KEY_METRIC: P/E=X  │  │ KEY_CATALYST: ...  │
│ ANALYSIS: 2-3 sent │  │ ANALYSIS: 2-3 sent │  │ ANALYSIS: 2-3 sent │
└──────────┬─────────┘  └──────────┬─────────┘  └──────────┬─────────┘
           └──────────────────────▼──────────────────────────┘
                         ┌─────────────────────┐
                         │  Research Manager   │
                         │  SYNTHESIS: 3-4 sent│
                         │  (all three views)  │
                         └──────────┬──────────┘

PHASE 2: Bull/Bear Debate (FAST calls, multi-round)
───────────────────────────────────────────────────────────────────────
Round 1:
  ┌─────────────────────┐     ┌─────────────────────┐
  │   Bull Researcher   │     │   Bear Researcher   │
  │ Opening argument    │     │ Opening argument    │
  │ (3 bullet points)   │     │ (3 risk factors)    │
  └──────────┬──────────┘     └──────────┬──────────┘
Rounds 2–N:
  Bull sees bear argument → rebuttal
  Bear sees bull argument → rebuttal
Final (if N>1):
  Both sides give closing statements

PHASE 3: Risk Management (1 FAST call)
───────────────────────────────────────────────────────────────────────
Input: full debate history + portfolio context
Output: unified 3-way debate internally:
  AGGRESSIVE: "Allocate up to X% given the upside"
  CONSERVATIVE: "Limit exposure, watch for Y"
  NEUTRAL: "Balanced view"
  SYNTHESIS: final risk summary for PM

PHASE 4: Portfolio Manager Decision
───────────────────────────────────────────────────────────────────────
SMART chain (Gemini-2.5-flash → Groq 70B → OpenRouter)

Standard mode output:
  RATING: BUY | OVERWEIGHT | HOLD | UNDERWEIGHT | SELL
  SCORE: -1.0 to +1.0
  REASON: one sentence

Primary mode output (llm_decision_primary=True):
  RATING: BUY | OVERWEIGHT | HOLD | UNDERWEIGHT | SELL
  SCORE: -1.0 to +1.0
  POSITION_SIZE: SMALL | MEDIUM | FULL | NONE
  TARGET_WEIGHT: 0.00-0.20
  STOP_LOSS: 0.05-0.12
  HOLD_DAYS: 1-5
  REASON: one sentence

Portfolio-Level Reasoning (1 SMART call, all tickers at once):
  CONCENTRATION_RISK: LOW | MEDIUM | HIGH
  CORRELATION_FLAG: NONE | MILD | HIGH
  REGIME_CHECK: NEUTRAL | REDUCE_EXPOSURE | INCREASE_EXPOSURE
  TICKER_WEIGHTS: GPRE=0.12, CLMT=0.08, ...
  EXCLUDED: none
  OVERALL: summary sentence

Score blending:
  final_score = 0.70 * ml_score + 0.30 * llm_score
```

### LLM Model Chain

Fail-over on 429 / error — models tried in ranked order for the rest of the run:

**FAST chain (21 models):**

| Tier | Model | Provider | Notes |
|---|---|---|---|
| 1 | llama-3.3-70b-versatile | Groq | 100K TPD |
| 1 | meta-llama/llama-3.3-70b-instruct:free | OpenRouter | Same quality |
| 1 | nousresearch/hermes-3-llama-3.1-405b:free | OpenRouter | 405B |
| 1 | nvidia/nemotron-3-super-120b-a12b:free | OpenRouter | 120B MoE |
| 1 | openai/gpt-oss-120b:free | OpenRouter | 120B GPT-class |
| 2 | qwen/qwen3-32b | Groq | 500K TPD |
| 2 | qwen/qwen3-next-80b-a3b-instruct:free | OpenRouter | 80B MoE |
| 2 | meta-llama/llama-4-scout-17b-16e-instruct | Groq | 500K TPD |
| 2 | qwen/qwen3.6-plus-preview:free | OpenRouter | 1M ctx |
| 3 | moonshotai/kimi-k2-instruct | Groq | 300K TPD |
| 3 | google/gemma-3-27b-it:free | OpenRouter | 131K ctx |
| 4 | llama-3.1-8b-instant | Groq | Fast fallback |

**SMART chain (8 models):** gemini-2.5-flash (Gemini) → llama-3.3-70b (Groq) → ... fallback to fast chain

`_rate_limited_models`: global set — skips exhausted models for rest of run  
`REASONING_MODELS`: set — passes `reasoning_effort="none"` to skip `<think>` output  
`_clean_think_response()`: strips any remaining `<think>` blocks

---

## Phase 5: LLM-Primary Decision Mode

```
if llm_decision_primary=True AND llm_agent_enabled=True AND LLM succeeded:
─────────────────────────────────────────────────────────────────────

select_tickers_llm_primary(screened, decisions, max_positions):
  ├─ BUY / OVERWEIGHT → include (sorted by LLM score descending)
  ├─ HOLD → fill remaining slots (ML score as tiebreaker)
  └─ UNDERWEIGHT / SELL → exclude entirely

compute_llm_primary_weights(selected, decisions, screened, cfg):
  ├─ If LLM provided TARGET_WEIGHT → use directly
  ├─ Else map POSITION_SIZE → weight range:
  │    SMALL  → [5%, 10%]   (conservative position)
  │    MEDIUM → [10%, 15%]  (normal position)
  │    FULL   → [15%, 20%]  (high-conviction position)
  ├─ Interpolate within range using LLM score
  ├─ Apply min 2% / max 20% hard caps
  └─ Normalize to sum = 100%

apply_portfolio_reasoning_enforced(target_weights, reasoning, screened):
  ├─ TICKER_WEIGHTS → adjust weights (advisory, won't add new tickers)
  ├─ EXCLUDED → remove (advisory, won't remove BUY-rated tickers)
  ├─ CONCENTRATION_RISK=HIGH → cap each sector at 30%
  ├─ REGIME_CHECK=REDUCE_EXPOSURE → scale all weights × 0.50
  └─ Re-normalize after all adjustments

FALLBACK (revert entirely to ML pipeline):
  ├─ LLM API failed (no key, timeout, all models 429)
  ├─ LLM returned 0 BUY/OVERWEIGHT tickers
  └─ llm_decision_primary=False
```

---

## Phase 6: Quantitative Circuit Breakers

These run **after** LLM weights are set — hard limits that LLM cannot override:

```
Sequential weight transforms:
─────────────────────────────────────────────────────────────────────

1. INSTRUMENT SLEEVE CONSTRAINTS
   ETF/fund tickers ≤ 35% combined weight
   Equity tickers ≥ 50% combined weight

2. UNIFIED OPTIMIZER (minimize: −alpha + λ_risk*variance
                               + λ_turnover*turnover + λ_cost*costs)
   Constraints:
     max position ≤ 20%
     correlated pairs (ρ ≥ 0.70) combined ≤ 25%
     portfolio beta within ±0.25 of target (1.0)
   Covariance: Ledoit-Wolf shrinkage

3. REGIME EXPOSURE SCALAR
   market_regime_composite = f(vol_regime, market_trend, market_breadth)
   scale = clamp(composite, min=0.7 LLM-primary / 0.5 ML-primary, max=1.2)
   target_weights *= scale

4. VOLATILITY TARGETING
   target_vol = 15% annualized
   scale = min(target_vol / portfolio_vol, 1.0)
   min_scalar = 0.6 in LLM-primary (softer floor)

5. DRAWDOWN MANAGEMENT
   if drawdown > 10%: scalar = max(1 - 5*(dd - 0.10), 0.25)
   target_weights *= scalar

6. REWARD POLICY SCALARS (Bayesian bandit)
   exposure_scalar: 0.3x–1.5x
   conviction_scalar: 0.5x–2.0x

7. HARD POSITION CAPS
   max 20% per position
   min 2% per position (dust filter)

8. PROFESSIONAL TRADING GATES (O'Neil / Minervini / PTJ rules):

   MARKET GATE (check_market_direction):
     Block all new BUYs if ≥2 of 3 hostile:
       vol_regime > 1.8
       market_trend_20d < -5%
       market_breadth < 35%

   R:R GATE (filter_by_risk_reward):
     upside = pred_return × price
     downside = vol_adjusted_stop × price
     Block if R:R < 2.0 (minimum 2:1)

   RISK-BASED SIZING (compute_risk_based_size):
     max_risk = portfolio_value × 0.02 (2% per trade)
     weight = min(max_risk / (price × vol_stop), max_position_pct)

   SCALING (compute_scaled_entry):
     New positions enter at 50% of planned size
     Remaining 50% added on confirmation
```

---

## Phase 7: Portfolio Manager — Stateful Exit System

```
State file: screener_portfolio_state.json (atomic write + .bak)
Event log:  screener_portfolio_state.json.events.jsonl (append-only)
POSIX file locking on all writes

Position fields:
  ticker, entry_price_cad, entry_date, shares (fractional)
  status: OPEN | CLOSED:STOP_LOSS | CLOSED:PEAK_TARGET | ...
  exit_price, exit_date, exit_reason
  highest_price (trailing stop tracking)
  entry_pred_peak_days, entry_pred_return (recorded at entry)
  last_partial_sell_at (cooldown for partial exits)
```

### Exit Priority Order

| Priority | Rule | Trigger | ML Override? |
|---|---|---|---|
| 1 | HARD_MAX_HOLD | days ≥ 5 (hard ceiling) | No |
| 2 | SOFT_MAX_HOLD | days ≥ 3 | Yes, if pred_return ≥ 3% |
| 3 | PEAK_TARGET | days ≥ pred_peak_days | No |
| 4 | STOP_LOSS (vol-adj) | return < −(base 8% × stock_vol/30%), clamped [4%, 15%] | No |
| 5 | TAKE_PROFIT | return ≥ take_profit_pct | No |
| 6 | QUICK_PROFIT | return ≥ 3% within 2 days | No |
| 7 | LOW_DAILY_RETURN | gain/day < 0.8% | Yes, if pred_return ≥ 1% |
| 8 | MOMENTUM_DECAY | gave back 40%+ of peak gain | No |
| 9 | AGE_URGENCY | old + return < 1% | Disabled by default |
| 10 | TRAILING_STOP | below peak × (1 − 8%) | Activates after 5% gain |
| 11 | SIGNAL_DECAY | live pred_return < −2% | No |
| 12 | PEAK_DETECTION (partial) | 2+ of: NEG_PRED, SCORE_DROP, RSI_OB, MA_EXT | 50% sell only; min 10% gain, ≥ 2d held |

### LLM Exit Review (runs before mechanical exits)

```
review_exits() → per-position:
  EXIT + HIGH urgency   → immediate SELL (ML veto disabled)
  EXIT + MEDIUM urgency → SELL unless ML pred_return > 3%
  EXIT + LOW urgency    → advisory only, ignored
  TIGHTEN_STOP          → reduce stop distance by 2%
  HOLD                  → no action
```

### Dynamic Holding Periods

| vol_regime | Soft exit | Hard exit |
|---|---|---|
| 0.5 (calm) | ~4 days | ~6 days |
| 1.0 (normal) | 3 days | 5 days |
| 2.0 (stressed) | ~2 days | ~3 days |

---

## Phase 8: Reward Feedback Loop

```
RewardLog: prediction outcomes
  At entry: log(ticker, pred_return, pred_confidence, date)
  At exit:  log(ticker, actual_return, days_held, exit_reason)
  compute_online_ic(): 60-day rolling IC of pred vs actual

ActionRewardLog: trade outcomes
  At BUY:  log(action, pred_return, price, weight)
  At SELL: log(action, realized_return, days_held)

RewardPolicy (Bayesian Thompson Sampling bandit):
─────────────────────────────────────────────────
State vector:
  ├─ portfolio_sharpe_recent (rolling 10-day)
  ├─ equity_slope (trend in equity curve)
  ├─ win_rate (% profitable trades)
  └─ current_drawdown

4 action dimensions (each has Beta(α, β) posterior):
  ├─ exposure (how much to invest): 0.3x – 1.5x
  ├─ conviction (amplify best picks): 0.5x – 2.0x
  ├─ exit_tightness: 0.5 – 2.0x
  └─ hold_patience: 0.5 – 2.0x

Update rule:
  if daily_return > 0: α += 1  (success)
  if daily_return < 0: β += 1  (failure)
  θ ~ Beta(α, β) sampled each run

compute_ensemble_reward_weights():
  Each ensemble member IC → weights = softmax(IC)
  Better-predicting models get more weight over time
```

---

## Phase 9: Reporting

```
render_reports(reports_dir, run_meta, screened, weights, trade_actions, ...)
─────────────────────────────────────────────────────────────────────
Outputs:
  daily_email.html      ── HTML email (sent via Gmail SMTP/465)
  daily_report.txt      ── machine-readable text report
  portfolio_weights.csv ── weight + metrics table
  trade_actions.json    ── serialized TradeAction list

HTML Email sections (in order):
  Header: "Trading Intelligence Report · LLM-Primary · {date}"
  Equity card: $X CAD · day return · all-time return
  Position card: N open · N actions today
  Risk status card: Normal / Elevated / Critical + DD%

  Risk Dashboard (horizontal bar):
    Drawdown from Peak    (gauge bar)
    Data Freshness        (Daily | Intraday 1h)
    Kill Switch           (Normal | HALTED)
    Market Gate           (Open | Caution | BLOCKED)
    Settlement            (Clear | T+2 Warning)

  Action summary: LLM reasoning behind each action
  Action table: BUY/SELL/HOLD with price, shares, reason

  Multi-Agent Analysis (per ticker card):
    Ticker + LLM Rating + Score
    Verdict (one-line summary from PM)
    PM Decision: RATING, SCORE, SIZE, WEIGHT, STOP, HOLD_DAYS
    Analyst pipeline flow diagram
    Bull argument (left column)
    Bear argument (right column)
    Risk assessment (below debate)

  Portfolio Details: P&L breakdown, holdings table, universe stats
  Model Validation: IC summary (if available)
  Attachments: daily_report.txt, portfolio_weights.csv
```

---

## CI/CD Pipeline

```
Workflows:
─────────────────────────────────────────────────────────────────────

train-stock-screener-model.yml (weekly, Sun 9PM ET)
  Timeout: 45min
  4000 tickers, 730d history, 4 models, 8 Optuna trials
  Promotion gates → blocks bad model from deploying
  Artifact: screener-model-{date}-{run_id}

daily-stock-screener.yml (weekdays, 9PM + 7:30AM ET)
  Timeout: 30min · Concurrency: stock-screener-pipeline (no cancel)
  Steps:
    1. Restore cache: pip + universe + data + state
    2. Find latest model artifact (GitHub REST API)
    3. Download + unzip model to models/ensemble/
    4. Run: python -m stock_screener.cli daily
    5. Compute health counters (inline Python)
    6. Write GitHub Step Summary (markdown dashboard)
    7. Save state cache (keyed by run_id)
    8. Upload artifact (14d TTL)
    9. Email HTML report via Gmail SMTP
   10. On failure: create GitHub Issue

intraday-stock-screener.yml (weekdays, 10AM/12PM/2PM ET)
  Timeout: 15min · cancel-in-progress=true
  Restores state from daily workflow cache
  Runs: python -m stock_screener.cli intraday

Secrets:   GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY,
           FINNHUB_API_KEY, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO
Variables: LLM_AGENT_ENABLED=1, LLM_DECISION_PRIMARY=1, ...
Kill switch: gh variable set TRADING_HALT --body "1"
```

---

## Data Flow: Signal → Action

Example: GPRE on 2026-04-03 (Liberation Day tariff selloff)

```
[Universe fetch]
GPRE ∈ S&P 500 universe

[Feature computation]
ret_5d=3.25%, rsi=52, vol_60d=45%, market_vol_regime=2.26

[ML scoring]
pred_return=2.82%, pred_confidence=88.5%
pred_peak_days=3, ret_per_day=0.94%
score = 0.78 (top decile)

[LLM analyst team]
Technical:    BULLISH (momentum + MA alignment)
Fundamental:  NEUTRAL (energy sector concerns)
Sentiment:    BEARISH (negative headlines about energy)
Research Mgr: "Mixed signals — ML predicts upside but sector
               headwinds are real and vol regime is elevated"

[Debate]
Bull: "3.25% 5-day momentum, ML 88.5% confidence, sector resilience"
Bear: "Oil price pullback, RSI overbought, high vol regime"

[Risk assessment]
"Limit to ~10% of cash given high volatility beta=1.47"

[PM Decision]
RATING: OVERWEIGHT · SCORE: +0.5
POSITION_SIZE: SMALL · TARGET_WEIGHT: 0.08

[Market Gate]
vol_regime=2.26 > 1.8  ← hostile
market_trend=-1.4%     ← hostile (index down sharply)
hostile_count=2 → BLOCKED ← correct: markets fell >3% that day

→ GPRE not purchased ✓
```

---

## Architecture Assessment

### What Works Well

**LLM multi-agent system**: The 4-phase pipeline (analysts → debate → risk → PM) produces qualitatively rich reasoning that a pure ML system cannot. The April 3 email shows agents correctly identifying sector headwinds, insider selling, and overbought technicals as counterweights to ML bullish signals.

**Model chain failover**: The 21-model ranked chain means a rate-limited Groq account (100K token/day) gracefully cascades through OpenRouter alternatives without any code change. The `_rate_limited_models` global set prevents re-hitting exhausted endpoints.

**Quantitative circuit breakers**: The Market Gate correctly blocked all BUYs on April 3 (Liberation Day tariff selloff — one of the worst market days in years). A pure LLM-confidence system might have bought into the falling market.

**State persistence**: Atomic write + JSONL event log + rebuild-from-events makes the portfolio state extremely durable. Even if the JSON file corrupts, events reconstruct it.

**Reward loop**: Online IC tracking and Bayesian bandit feedback mean the system learns from its own mistakes without retraining the ML model.

### Areas for Improvement

**Vol targeting vs small portfolio**: With `target_vol=15%` and individual stocks at 40–70% annualized vol, a 4-stock portfolio gets scaled down to ~25–30% invested. The system is holding too much cash relative to the LLM's conviction. The LLM-primary softening (min_scalar=0.6) helps but the fundamental tension remains — the vol target was designed for a 10–20 stock diversified portfolio.

**Intraday is reactive, not predictive**: Intraday runs use cached daily weights and only fetch current prices — no re-screening, no feature recomputation. For a true intraday strategy you'd want to recompute momentum and volume signals hourly. Current design optimizes for API cost over signal freshness.

**Stop-loss parsing ambiguity**: LLM sometimes outputs `STOP_LOSS: $22.50` (price level) instead of `STOP_LOSS: 0.08` (percentage). The parser needs to distinguish absolute vs relative values.

**Small portfolio budget ($500 CAD)**: Below the minimum viable position for many US stocks. Fractional shares are supported but creates rounding artifacts in weight computations. Acceptable for paper trading; would need scaling for real deployment.

---

## Configuration Reference

Key parameters (full list in `stock_screener/config.py`):

| Parameter | Default | Effect |
|---|---|---|
| `portfolio_budget_cad` | 500 | Total capital |
| `dynamic_size_max_positions` | 8 | Max concurrent holdings |
| `min_pred_return_threshold` | 0.01 | ML minimum alpha signal |
| `min_confidence_threshold` | 0.50 | ML model certainty floor |
| `vol_target` | 0.15 | Portfolio annualized vol target |
| `max_position_pct` | 0.20 | Single-stock cap |
| `stop_loss_pct` | 0.08 | Base stop (vol-adjusted) |
| `max_holding_days` | 3 | Soft exit trigger |
| `max_holding_days_hard` | 5 | Hard ceiling |
| `llm_agent_ml_weight` | 0.70 | ML weight in score blend |
| `llm_agent_llm_weight` | 0.30 | LLM weight in score blend |
| `llm_decision_primary` | True | LLM drives ticker/weight selection |
| `max_vol_regime` | 1.8 | Market gate vol threshold |
| `min_rr_ratio` | 2.0 | Minimum R:R for trade entry |
| `risk_per_trade_pct` | 0.02 | Max 2% capital at risk per trade |
| `trailing_stop_activation_pct` | 0.05 | Trailing stop activates after 5% gain |
| `trailing_stop_distance_pct` | 0.08 | Trailing stop trails 8% below peak |
| `rotation_cooldown_days` | 2 | Min days before rotation sell |
| `reward_model_enabled` | True | Bayesian bandit feedback |
| `quantile_models_enabled` | True | LCB scoring with q10/q50/q90 |
| `regime_specialist_enabled` | True | Bull/neutral/bear specialist ensembles |

---

## File Map

| File | Lines | Role |
|---|---|---|
| [pipeline/daily.py](../stock_screener/pipeline/daily.py) | ~2920 | Master orchestrator (`run_daily`, `run_intraday`) |
| [reporting/render.py](../stock_screener/reporting/render.py) | ~1550 | HTML/text/CSV report generator |
| [agents/trading_agent.py](../stock_screener/agents/trading_agent.py) | ~1800 | Full multi-agent LLM system |
| [modeling/model.py](../stock_screener/modeling/model.py) | ~700 | ML inference |
| [modeling/train.py](../stock_screener/modeling/train.py) | ~800 | Model training |
| [portfolio/manager.py](../stock_screener/portfolio/manager.py) | ~900 | Exit/entry/rotation logic |
| [optimization/risk_parity.py](../stock_screener/optimization/risk_parity.py) | ~1100 | Portfolio weight construction |
| [config.py](../stock_screener/config.py) | ~600 | All system parameters |
| [features/technical.py](../stock_screener/features/technical.py) | ~500 | Feature engineering |
| [agents/config.py](../stock_screener/agents/config.py) | ~188 | LLM model chain |
| [agents/trade_rules.py](../stock_screener/agents/trade_rules.py) | ~200 | Professional trading rules |
| [data/news_sources.py](../stock_screener/data/news_sources.py) | ~400 | Multi-source news aggregator |
| [reward/policy.py](../stock_screener/reward/policy.py) | ~300 | Bayesian bandit |
| [screening/screener.py](../stock_screener/screening/screener.py) | ~400 | Universe scoring and filtering |
