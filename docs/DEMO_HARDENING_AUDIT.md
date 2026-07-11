# Mossy 4X Demo Hardening Audit

Date: 2026-07-11

## Decision

Keep Mossy 4X in OANDA practice/demo mode. Do not enable live trading.

The broker contains a hard stop that exits when either `MODE=live` or `OANDA_ENV=live`. This guard remains unchanged.

## Current architecture

- Python worker deployed by Render from `main`.
- OANDA practice REST API for candles, account state, open trades, and demo orders.
- Decision cycle scheduled every 60 seconds.
- Default strategy timeframe: M5.
- Signal inputs include EMA, RSI, MACD, ADX, ATR, session filters, spread filters, cooldowns, and duplicate-position checks.
- Risk manager tracks daily and weekly P/L, drawdown, equity floor, loss streaks, cooldowns, trade counts, and persistent halt state.
- SQLite trade journal stores entries, exits, indicators, gating flags, session metadata, and realised P/L.
- Adaptive tuner can reduce effective risk after poor recent results.
- Status/health reporting exposes scheduler state, cycle age, broker-sync age, equity, and open trades.

## Findings

### Fixed in this hardening branch

1. Render was configured with `AGGRESSIVE_MODE=true` even though this is intended for testing only.
2. Render was configured with `SESSION_MODE=ALWAYS`, permitting off-session entries.
3. Render loss and drawdown limits were looser than the safer defaults.
4. The runtime risk cap allowed up to 1.0% per trade; this branch reduces the cap to 0.5%.
5. Aggressive cooldown was only two candles; this branch restores nine candles.
6. Important runtime switches were implicit. This branch makes TP, adaptive tuning, quiet logs, cache TTL, and reset controls explicit.

### Remaining operational checks in Render

These cannot be verified from GitHub alone:

- Confirm `OANDA_API_KEY` belongs to an OANDA practice account.
- Confirm `OANDA_ACCOUNT_ID` is the matching practice account.
- Confirm Render shows `MODE=demo` and `OANDA_ENV=practice` after deployment.
- Confirm the latest deploy revision matches the merged GitHub commit.
- Confirm heartbeat and decision-cycle logs continue every minute.
- Confirm `/status` reports `scheduler_alive=true` and a recent cycle age.
- Confirm no `LIVE` order messages appear in logs.

## Data persistence warning

The risk state and SQLite trade journal are written under the application data directory. Render instances can lose local filesystem data during replacement or redeployment unless persistent storage is configured. Before relying on long-run performance history, attach a persistent disk in Render and set `MOSSY_STATE_PATH` to its mounted directory, or move the journal to a managed database.

Do not add a disk blindly through the Blueprint until the current Render service and billing/storage options are reviewed in the Render dashboard.

## Demo operating profile

- Mode: demo
- OANDA environment: practice
- Instruments: AUD_USD, GBP_USD
- Session mode: SOFT
- Aggressive mode: disabled
- Risk cap: 0.5% per trade
- Daily loss cap: 1%
- Weekly loss cap: 3%
- Maximum drawdown cap: 5%
- Maximum positions: 3
- Cooldown: 9 candles
- Take profit: enabled
- Adaptive risk reduction: enabled
- Verbose market logs: disabled

## Approval gate before any future live consideration

Live mode must remain blocked until all of the following are satisfied:

1. A meaningful demo sample has been collected across different market conditions.
2. The journal is persistent and performance reports are reproducible.
3. Maximum drawdown and loss-streak behaviour have been verified from real demo fills.
4. Position sizing, stop-loss, take-profit, spread, and currency conversion calculations are independently reviewed.
5. Render restart/deploy recovery is tested without losing risk state.
6. A separate explicit change removes the live-mode guard and receives manual approval.

## Recommended next action

Merge this branch, allow Render to deploy it, then inspect the first 20-30 minutes of logs and the status response. Do not change the OANDA environment from practice.
