# sentinel.
### Predictive Drawdown Containment System for Nifty 50 F&O traders

DTU B.Tech minor project · Department of Applied Mathematics
Team: Lokesh Godara (23/MC/082), Naman Sagar (23/MC/093), Navdeep Singh (23/MC/094) · Guide: Prof. L. N. Das

---

## Why

A SEBI study of individual traders in the equity F&O segment (FY22 to FY24) found that **93% lost money**.
Most of those losses come from behaviour, not bad luck: revenge trading, overtrading, holding bought options
into expiry while time decay eats them, and letting losses run bigger than wins.

Nifty Sentinel has two engines in one dashboard:

1. **Behavioural auditor (looks back):** reads a broker tradebook, finds the habits costing money, and replays
   the history with a daily stop-loss to show what would have been saved.
2. **Predictive containment (looks forward):** reads the live market, forecasts possible losses, and says
   whether to trade at all, how big, and which defined-risk strategy fits today.

---

## The dashboard: six tabs

| Tab | What it does |
|---|---|
| **Nifty 50 Price Chart** | Live candles with EMA 20/50 and the **expected-move band**: the 68% range India VIX implies for today, flagging unusually large moves. |
| **Risk Command Center** | Circuit breaker (green / amber / red), 1-day VaR and expected shortfall, probability of ruin, a 1,000-path **Monte Carlo** cone, market regime, quarter-Kelly position size, and **all six defined-risk strategies** with one-click "add to portfolio". |
| **Option Safety & Chain** | Option chain for any **upcoming expiry** (weekly Tuesdays, monthly last Tuesday), OPTIMAL / SPECULATIVE / HAZARDOUS verdicts with **Greeks and the reason** for each, and an open-interest chart with max pain, support and resistance. |
| **Trade Tester** | TAKE / HEDGE / DROP verdict, a **second opinion from the ML model**, the trade's worst case against today's **daily loss limit**, and a **P&L heatmap** of Nifty moves against days to expiry. |
| **Active Portfolio** | Positions **saved between sessions**, strategies **grouped** with one "Close strategy" button, **Greeks in plain words** ("if Nifty rises 1% you gain about ₹8,900"), stop-loss / target alerts, CSV download, payoff chart, and the tradebook audit (discipline score, habits found, money a daily stop would have saved). |
| **ML & Backtest Analysis** | Take/drop models trained on **real Nifty history** (or simulated trades): Random Forest vs gradient boosting vs a logistic baseline, ROC curves, a backtest with a **confidence slider**, and **"Why this prediction?"** per-feature explanations. |

Above the tabs: a **tradebook upload** (CSV / Excel from Zerodha, Groww, Dhan, Angel One, Upstox) that replaces
the sample in the audit and risk tabs, and a **Demo mode** switch for offline presenting.

---

## Running it

Requires Python 3.10+ (tested on 3.13).

```bash
pip install -r requirements.txt
```

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501`.

On Windows you can instead double-click **`Open Nifty Sentinel.bat`** in the project folder: it starts the app
on port 8510 and opens your browser (if the app is already running, it just opens the browser). Keep its
window open while you use the app; closing it stops the app.

### Offline demo mode (for presentations)
The evening before presenting, with internet:
```bash
python scripts/save_market_snapshot.py
```
This saves today's market data and 5 years of daily Nifty / India VIX history to `data/market_snapshot/`.
Then switch on **Demo mode (offline)** under the logo: the app never touches the network. Every price carries
a badge: **Live** (Yahoo Finance), **Cached** (saved snapshot) or **Simulated** (no feed and no snapshot).
If the live feed drops mid-demo, the app falls back to the snapshot on its own.

### Saved portfolio
Positions are saved to `data/portfolio.json` (not committed to git). Set `SENTINEL_PORTFOLIO_PATH` to use a
different file; the test suite does this so it never touches your real portfolio.

### Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
115 tests, including headless runs that click through every tab.

---

## Machine learning: results and honest limits

**Data.** 33,990 option trades built from 1,230 trading days of real Nifty 50 and India VIX (Nov 2021 to
Nov 2026). Each day: calls and puts at 5 strikes (−2% to +2% from spot) and 3 expiries (2, 7, 14 days),
priced with Black-Scholes at that day's close and VIX, held to expiry, and labelled by what Nifty actually did.

**Evaluation.** Trained on trades before 5 May 2026 and tested on the 3,470 trades after it. Training trades
still open at the cutoff are removed so nothing leaks from the future; cross-validation is time-ordered.

| Unseen trades, May–Nov 2026 | Random Forest | Gradient boosting | Logistic baseline |
|---|---|---|---|
| ROC-AUC | **0.688** | 0.684 | 0.681 |
| Time-ordered CV ROC-AUC | 0.691 | 0.671 | 0.695 |

Only 30.7% of bought options were profitable; among trades the Random Forest rates 40% or more, 45.9% were.

**Backtest** (₹20 lakh, 1 lot per trade, real P&L): taking every trade wipes the account out after 934 trades;
a rule filter after 743; the ML filter at 40% confidence survives at −₹2.1 lakh, and at 45% ends +₹2.8 lakh.
The 45% bar was chosen after seeing these results, so it is optimistic; live use must fix it in advance.

**Limits.** The signal is real but modest (ROC-AUC about 0.69). Option open interest in the chain is simulated.
The model covers bought options held to expiry. A "Simulated trades" switch in the ML tab shows the original
formula-based dataset for comparison.

---

## Project structure

```
├── core/
│   ├── models.py              # Trades, contracts, Greeks, regimes
│   ├── greeks.py              # Black-Scholes pricing and Greeks
│   ├── option_evaluator.py    # Verdicts (named thresholds), scenarios, P&L heatmap grid
│   ├── expiry.py              # Upcoming Nifty expiries (weekly Tuesday, monthly last Tuesday)
│   └── portfolio.py           # Positions, groups, saving, MTM, Greeks in words, stop/target alerts
├── data_feed/
│   ├── live_feed.py           # Nifty / India VIX via yfinance, option chain (simulated OI)
│   └── market_source.py       # Live → snapshot → simulated chain with source labels (Demo mode)
├── trade_analyzer/
│   ├── parser.py              # Multi-broker tradebook parser
│   ├── upload.py              # Safe CSV / Excel upload with friendly errors
│   ├── metrics.py             # Win rate, expectancy, drawdown
│   ├── leakage_detector.py    # Revenge trading, overtrading, expiry decay, risk-reward habits
│   └── counterfactual.py      # What-if replay with a daily stop
├── risk_engine/
│   ├── var_cvar.py            # Historical / parametric VaR and expected shortfall
│   ├── monte_carlo.py         # Bootstrap Monte Carlo and probability of ruin
│   ├── containment.py         # Circuit breaker, adaptive daily limit, Kelly sizing
│   ├── command_center.py      # One snapshot of every risk engine for the Risk Command Center
│   ├── expected_move.py       # VIX-implied expected range
│   └── trade_budget.py        # Does a trade fit today's loss limit?
├── strategy_optimizer/
│   ├── regime_classifier.py   # Trend, volatility, PCR, max pain
│   ├── payoff_calculator.py   # Multi-leg payoff and Greeks
│   └── strategy_recommender.py# Six defined-risk strategies with real legs
├── ml_engine/
│   ├── real_data.py           # Trades from real Nifty / VIX history, time split
│   ├── synthetic_data.py      # Simulated trades (seeded)
│   ├── classifier.py          # Random Forest take/drop model
│   ├── evaluation.py          # Three-model comparison, ROC, CV
│   └── explain.py             # Exact per-feature contributions for a prediction
├── backtesting/
│   ├── engine.py              # Simulated-trade backtest
│   └── threshold.py           # Backtest at a chosen confidence, stops at account wipe-out
├── scripts/save_market_snapshot.py
├── data/                      # Sample tradebook, market snapshot, saved portfolio
├── ui/app.py                  # Streamlit dashboard (charcoal and gold theme)
└── tests/                     # 115 unittest tests
```
