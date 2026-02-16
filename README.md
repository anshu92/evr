# Stock Screener (CAD) + Short-Horizon Trading (max 5 days)

This repo runs a daily **US + TSX/TSXV** stock screener, recommends a portfolio, and produces explicit **BUY/SELL/HOLD** actions with a **max holding period of 5 days**. It also trains a single ML ranking model weekly (Sunday 02:00 UTC).

### What you get each day

- **Email** with:
  - recommended actions (BUY/SELL/HOLD)
  - portfolio weights (CAD)
  - a full text report + machine-readable `trade_actions.json`

### Local run

Install deps:

```bash
pip install -r requirements.txt
```

Train model (optional; the daily run falls back if no model is present):

```bash
LABEL_HORIZON_DAYS=5 python -m stock_screener.cli train-model --log-level INFO
```

Run daily trading pipeline:

```bash
MAX_HOLDING_DAYS=5 PORTFOLIO_STATE_PATH=screener_portfolio_state.json python -m stock_screener.cli daily --log-level INFO
```

Outputs:

- `reports/daily_email.html`
- `reports/daily_report.txt`
- `reports/portfolio_weights.csv`
- `reports/trade_actions.json`

### GitHub Actions

See `docs/GITHUB_WORKFLOWS.md` for workflow schedules, runtime-budget settings, and required secrets.

### End-to-End Workflow Diagram

```mermaid
flowchart TD
  %% ---------- Triggers ----------
  T1[CLI: python -m stock_screener.cli daily] --> D0
  T2[CLI: python -m stock_screener.cli train-model] --> M0
  T3[CLI: python -m stock_screener.cli eval-model] --> E0
  GA1[GitHub Actions: daily-stock-screener.yml<br/>Mon-Fri 3x/day] --> D0
  GA2[GitHub Actions: train-stock-screener-model.yml<br/>Weekly] --> M0

  %% ---------- Daily Pipeline ----------
  subgraph DAILY[Daily Trading Pipeline - stock_screener/pipeline/daily.py]
    D0[Load Config and runtime budget] --> D1[Fetch universes<br/>US: NasdaqTrader<br/>TSX/TSXV: TSX API]
    D1 --> D2[Fetch FX USDCAD and prices via yfinance<br/>fundamentals and macro]
    D2 --> D3[Compute features in CAD<br/>technical, regime, macro, fundamentals]
    D3 --> D4{USE_ML and model present?}

    D4 -- Yes --> D5[Load manifest, metrics, target encodings<br/>validate feature schema hash/parity]
    D5 --> D6[Predict ensemble return and uncertainty<br/>optional: regime-gated specialists<br/>optional: quantile LCB<br/>optional: peak-day model]
    D6 --> D7[Calibrate predictions]
    D4 -- No --> D8[Baseline factor scoring path]
    D7 --> D9
    D8 --> D9

    D9[Score universe<br/>liquidity/price/vol filters<br/>sector-neutral selection<br/>entry filters] --> D10[Dynamic portfolio size from confidence/return/IC]
    D10 --> D11[Initial weights<br/>inverse-vol or correlation-aware]
    D11 --> D12[Portfolio optimization<br/>unified optimizer OR sequential constraints]
    D12 --> D13[Exposure controls<br/>regime scalar, vol targeting, drawdown scalar]
    D13 --> D14[Reward-policy overlay<br/>adaptive exposure/conviction/exit-tightness]
    D14 --> D15[PortfolioManager<br/>apply exits, rebalance hysteresis,<br/>build BUY/SELL/HOLD plan]
    D15 --> D16[Persist state JSON and reward logs]
    D16 --> D17[Render reports<br/>daily_report.txt, daily_email.html,<br/>portfolio_weights.csv, trade_actions.json]
  end

  %% ---------- Training Pipeline ----------
  subgraph TRAIN[Model Training Pipeline - stock_screener/modeling/train.py]
    M0[Load Config] --> M1[Build universe and fetch data<br/>prices, fx, fundamentals, macro]
    M1 --> M2[Compute training features and labels<br/>cost-aware, market-relative, peak options]
    M2 --> M3[Time splits, holdout, CV]
    M3 --> M4[Train ensemble models<br/>XGBoost and LightGBM]
    M4 --> M5[Train regime specialists<br/>bull/neutral/bear]
    M5 --> M6[Train quantile models q10/q50/q90<br/>and peak-timing model]
    M6 --> M7[Evaluate: IC/top-N, realistic backtest,<br/>walk-forward robustness, turnover/cost metrics]
    M7 --> M8[Promotion gates]
    M8 -->|pass| M9[Save models, manifest.json, metrics.json]
    M8 -->|fail & enforce| M10[Block promotion]
  end

  %% ---------- Eval ----------
  subgraph EVAL[Model Evaluation - cli eval-model]
    E0[Load current model and data] --> E1[Compute ranker/regressor metrics]
    E1 --> E2[Log IC and top-N summaries]
  end

  %% ---------- Workflow/Artifact Loop ----------
  M9 --> A1[Upload model artifact and cache]
  A1 --> A2[Daily workflow fetches latest trained artifact]
  A2 --> D4

  D17 --> A3[Upload daily artifact and send email]
  D16 --> A4[screener_portfolio_state.json cached]
  A4 --> D0
```
