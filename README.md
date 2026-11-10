# 🛡️ NIFTY SENTINEL
### Predictive Drawdown Containment System & Options Alpha Optimizer for Nifty 50 F&O

---

## 📌 Executive Summary
**NIFTY SENTINEL** is a quantitative risk containment and behavioral analytics platform specifically engineered for Indian retail and proprietary Nifty 50 F&O derivatives traders.

According to SEBI studies, **93% of individual F&O options traders in India lose money**, primarily due to unconstrained intraday drawdowns, revenge trading, overtrading spikes, and holding decaying out-of-the-money (OTM) options to zero.

**Nifty Sentinel solves this through two powerful engines**:
1. **Behavioral Leakage Auditor & Counterfactual Engine**: Ingests broker tradebooks (Zerodha, Groww, Dhan, Angel One, Upstox) and diagnoses the exact psychological traps causing losses, simulating the "What-If" equity curve under strict containment rules.
2. **Predictive Drawdown & Market Regime Strategy Selector**: Employs 1,000-path Monte Carlo simulations, Value-at-Risk (VaR / Expected Shortfall), and Nifty market regime detection (India VIX, IV Percentile, PCR, Max Pain) to enforce real-time circuit breakers and recommend defined-risk multi-leg option strategies (Spreads, Condors, Butterflies).

---

## 🚀 Key Features

### 1. 📊 Executive PnL Journey & Quantitative Auditing
- Realized Net PnL, Win Rate %, Profit Factor, Expectancy per trade.
- Interactive **Underwater Drawdown curve** with peak high-water-mark tracking.
- Segmented diagnostics: Call (CE) vs Put (PE), Option Buyers vs Sellers, Expiry (0-DTE) vs Non-Expiry.

### 2. 🔍 6-Point Behavioral Leakage Diagnostics
- **😡 Revenge Trading Trap**: Flags rapid re-entries (<15 mins) after losses and quantifies total capital drained.
- **📈 Overtrading Frequency Decay**: Identifies the threshold where daily trade count degrades win rate and balloons commissions.
- **⏳ 0-DTE Expiry Theta Crush**: Audits long options held into expiry decay and quantifies preventable losses.
- **📉 Martingale / Loss Averaging**: Detects doubling down on falling option premiums.
- **⚖️ Risk-to-Reward Inversion**: Analyzes asymmetric payoff profiles where 1 loss wipes out 4 wins.
- **Behavioral Discipline Score (0–100)** with actionable behavioral prescriptions.

### 3. ⚖️ "What-If" Counterfactual Simulation
- Interactively re-executes past trades under rule-based constraints:
  - Daily Drawdown Hard Stop (e.g. 1.5% or ₹3,000 max daily loss).
  - Maximum trades per day limit (e.g. max 3 trades/day).
  - Option Premium Stop-Loss Cap (e.g. 30% max loss per trade).
- Shows exact **₹ Capital Saved**, **Drawdown Reduction %**, and **Taxes/Brokerage Saved**.

### 4. 🔮 Predictive Monte Carlo & Tail Risk Engine
- **1,000-Path Bootstrap Monte Carlo**: Forecasts the 5th, 25th, 50th (median), 75th, and 95th percentile equity cones over the next 50-250 trades.
- **Probability of Ruin / Account Distress %**.
- **95% Worst-Case Maximum Drawdown**.
- **1-Day 95% and 99% Value-at-Risk (VaR)** & **Conditional VaR (Expected Shortfall)**.

### 5. 🧭 Nifty 50 Market Regime & Defined-Risk Strategy Optimizer
- **Market Regime Classifier**: Combines Nifty Spot, 20/50 EMAs, ADX, India VIX, IV Percentile, PCR, and Max Pain.
- **Defined-Risk Strategies**:
  - *Bull Call Debit Spread* (Bullish + Low IV)
  - *Bull Put Credit Spread* (Bullish + High IV)
  - *Bear Put Debit Spread* (Bearish + Low IV)
  - *Bear Call Credit Spread* (Bearish + High IV)
  - *4-Leg Defined Iron Condor* (Rangebound + High IV)
  - *Iron Butterfly ATM Theta Pin* (Pinning at Max Pain)
- **Interactive Multi-Leg Payoff Diagram** with Expiry and T+0 Greeks curves.
- **Fractional Kelly Position Sizer**: Calculates exact lot sizes and ₹ risk based on mathematical edge.

---

## 🛠️ Installation & Running the Dashboard

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
```bash
streamlit run ui/app.py
```
Open your browser at `http://localhost:8501`.

### Offline demo mode (for presentations)
The day before presenting, with internet, save today's market data:
```bash
python scripts/save_market_snapshot.py
```
Then switch on **Demo mode (offline)** under the logo. The app uses `data/market_snapshot/` and never
touches the network. Every price shows a badge: **Live** (Yahoo Finance), **Cached** (saved snapshot) or
**Simulated** (no feed and no snapshot). If the live feed drops mid-demo, the app falls back to the
snapshot automatically.

### 3. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📁 Project Architecture

The dashboard (`ui/app.py`) has six tabs: **Nifty 50 Price Chart**, **Risk Command Center**, **Option Safety & Chain**,
**Trade Tester**, **Active Portfolio**, and **ML & Backtest Analysis**.

```
├── core/
│   ├── models.py              # Data structures for Trades, Contracts, Greeks & Regimes
│   ├── greeks.py              # Vectorized Black-Scholes pricing & Greeks engine
│   ├── option_evaluator.py    # Take/drop verdict for a single option trade (theta burn, scenarios)
│   └── portfolio.py           # Active positions book with live mark-to-market PnL
├── data_feed/
│   ├── market_source.py       # Live → snapshot → simulated data chain with source labels (Demo mode)
│   └── live_feed.py           # Nifty spot / India VIX / option chain via yfinance (synthetic fallback offline)
├── trade_analyzer/
│   ├── parser.py              # Multi-broker tradebook parser (Zerodha, Groww, Dhan, Angel One, CSV)
│   ├── metrics.py             # Performance statistics (Win rate, Expectancy, Sharpe, Max Drawdown)
│   ├── leakage_detector.py    # 6-point behavioral diagnostics (Revenge trading, Overtrading, etc.)
│   └── counterfactual.py      # "What-If" simulator: Re-executes history with risk caps
├── risk_engine/
│   ├── var_cvar.py            # Historical, Parametric VaR, and Expected Shortfall
│   ├── monte_carlo.py         # Monte Carlo forward simulation & Probability of Ruin
│   ├── containment.py         # Intraday Circuit Breaker & Fractional Kelly Sizer
│   └── command_center.py      # Combines all risk engines into one snapshot for the Risk Command Center tab
├── strategy_optimizer/
│   ├── regime_classifier.py   # Nifty market regime detection (Trend, VIX, PCR, Max Pain)
│   ├── payoff_calculator.py   # Multi-leg option payoff & Greeks curves
│   └── strategy_recommender.py# Defined-risk strategy recommendation engine
├── ml_engine/
│   ├── synthetic_data.py      # Seeded synthetic option-trade generator (features + profit label)
│   ├── classifier.py          # Random Forest take/drop classifier (trained on seed 42)
│   └── evaluation.py          # ROC-AUC, precision/recall, confusion matrix, k-fold CV vs a logistic baseline (unseen seed 99)
├── backtesting/
│   └── engine.py              # Baseline vs rule-based vs ML filter backtest (out-of-sample, seed 7)
├── data/
│   ├── sample_zerodha_trades.csv # Realistic sample tradebook with retail flaws
│   └── sample_nifty_market_data.json # Sample market data
├── ui/
│   └── app.py                 # Streamlit interactive dark financial terminal
└── tests/                     # unittest suite, incl. a headless check that every tab renders
```

### ⚠️ About the ML backtest
The ML model and backtest use **synthetic** trades produced by a scoring formula, with a fixed
₹5,000 win / ₹3,000 loss per trade. The model is trained on one random draw (seed 42) and the
backtest runs on an independent draw (seed 7) it has never seen. Results show how well the model
learns the simulated pattern, not proven performance on real Nifty markets.
