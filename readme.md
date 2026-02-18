# Lag-Based Sector Lagger Engine

A modular, research-grade engine for detecting sector laggers, backtesting rules, and generating trader-friendly trade sheets with a realistic portfolio model (max 3 concurrent trades, long + short).

---

## Project structure

```text
trading_engine/
│
├── data/
│   ├── __init__.py
│   ├── universe.py
│   └── price_loader.py
│
├── signals/
│   ├── __init__.py
│   └── lag_detector.py
│
├── backtest/
│   ├── __init__.py
│   ├── trade_generator.py
│   ├── portfolio.py
│   ├── rule_scoring.py
│   └── sector_analysis.py
│
├── reporting/
│   ├── __init__.py
│   ├── summaries.py
│   ├── rule_reports.py
│   └── trade_sheet.py
│
├── main.py
├── requirements.txt
└── notebooks/
    └── exploration.ipynb

Installation:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Running the engine:

From the project root (where trading_engine/ lives):

python -m trading_engine.main

This will
- Load/update price data (cached in data/historical_prices.parquet)
- Detect lag signals
- Generate trades
- Score rules
- Run a portfolio backtest (max 3 concurrent trades)
+ Save research outputs to research/:
	- all_trades.csv
	- rule_scores.csv
	- portfolio_equity_curve.csv
	- portfolio_used_trades.csv

Key concepts:

Universe - Defined in data/universe.py as sector → tickers.
Signals - Lag events where most of a sector moves strongly and one ticker lags.
Rules - Parameter sets over lookback, group move threshold, participation, lagger max move, entry lag, and hold.
Portfolio model - Max 3 concurrent trades, equal-weight across open trades, long + short allowed.