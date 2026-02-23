# Mossy 4X – 30-Day Performance Review

## Live/Demo Audit Data Window
- Source files searched: `data/trade_journal.db`, `data/audit_events.jsonl`, `data/trade_history_last24h.csv`
- Result: no 30-day trade rows found in `audit_trades` table; no `trade_history_last24h.csv` available at analysis time.

## 30-Day Metrics (Observed)
- Net P/L: `0.0000`
- Win rate: `0.00%`
- Average winner: `0.0000`
- Average loser: `0.0000`
- Maximum drawdown: `0.0000`
- Profit factor: `0.0000`

Interpretation: insufficient live/demo trade history in repository artifacts to statistically classify profitability.

## Loss Pattern / Rule-Behavior Notes
- Because no 30-day closed trades were present, loss-pattern clustering (session/symbol/time/volatility) is not statistically meaningful from repository logs alone.
- Existing event schema still supports this analysis when logs are present (`indicator_snapshot`, `risk_block`, `account_snapshot`, `daily_health_report`).

## Improvement Implemented (Strategy-side only)
To reduce chop-driven entries while preserving existing safety/audit behavior:
1. Added ADX gate to entry conditions (must satisfy `ADX_FILTER`).
2. Added minimum EMA spread filter relative to ATR (`EMA_SPREAD_ATR_MULT`, default `0.05`).

## Backtest Comparison (Synthetic chop-heavy market regime)
Using deterministic synthetic candles with frequent choppy periods:

- **Baseline logic**
  - Trades: `35`
  - Net P/L: `-0.0000328`
  - Profit factor: `0.9912`
  - Max drawdown: `0.0021371`

- **Enhanced logic (ADX + EMA spread/ATR filter)**
  - Trades: `8`
  - Net P/L: `0.0003826`
  - Profit factor: `1.6047`
  - Max drawdown: `0.0004218`

Conclusion: enhanced strategy materially improved risk-adjusted performance in chop-heavy conditions while reducing drawdown.
