# Deployment Notes

## Standard release flow (recommended)

Use this sequence for normal changes:

1. Create a branch and commit your changes.
2. Open a pull request into `main`.
3. Merge the pull request.
4. Let the `mcp-web` service auto-deploy from `main`.
5. Manually deploy the active `mossy-4x-render` worker after checks pass.

Why this is the default here:

- The public `mcp-web` service tracks `main` and auto-deploys.
- The active Singapore worker intentionally has auto-deploy disabled to prevent accidental trading-runtime releases.
- The older `mossy-4x` worker described by the root Blueprint is suspended. Do not sync the Blueprint over the active `mossy-4x-render` service.

## When manual deploy steps are needed

You may need manual intervention only when:

- auto-deploy is disabled in the Render dashboard,
- a deployment is stuck/failed and needs retry,
- you are performing rollback/hotfix operations.

## Mossy 4X read-only MCP

The MCP bridge exposes only sanitised health, fixed guardrails, and committed
algorithm reports. It has no order, optimisation, GitHub-write, secret-write,
or deploy tools.

The active worker needs these service-level values in Render:

- `MOSSY_MCP_STATUS_URL=https://mcp-web-kyhu.onrender.com/internal/runtime-heartbeat`
- `MOSSY_MCP_STATUS_KEY=<the same generated secret configured on mcp-web>`

The shared key belongs only in Render environment settings. Never commit it.

`mcp-web` uses a disposable SQLite cache on Render's free tier. During configured
weekday London/New York sessions, and whenever an open trade exists, the worker
publishes every ten minutes to keep the bridge warm. Safety-state changes publish
immediately. Outside those periods the service may sleep and lose its cache;
runtime health then fails closed as `BLOCKED` until a fresh signed heartbeat is
stored. Static guardrails and repository reports remain available after wake-up.

Do not copy OANDA, GitHub, or Render API credentials into the MCP service. A paid
disk or external datastore is intentionally not provisioned by this setup.

## Aggressive test profile (higher activity, bounded risk)

For short test windows where you want more entries than `SOFT` mode but still avoid reckless sizing:

- `SESSION_MODE=ALWAYS`
- `AGGRESSIVE_MODE=true`
- `AGGRESSIVE_RISK_PCT=0.004` (0.4% risk per trade)
- `AGGRESSIVE_MAX_POSITIONS=3`
- `AGGRESSIVE_COOLDOWN_CANDLES=2`
- `AGGRESSIVE_DAILY_LOSS_CAP_PCT=0.02`
- `AGGRESSIVE_WEEKLY_LOSS_CAP_PCT=0.05`
- `AGGRESSIVE_MAX_DRAWDOWN_CAP_PCT=0.15`

This is intentionally higher-tempo than default, but still constrained by loss caps/drawdown limits.

## Instrument Configuration

- Omit the `instruments` key or set it to `null`/`None` to use the default instrument basket.
- Provide an empty list (`[]`) to pause trading entirely.
- Set `merge_default_instruments` to `true` to append the default instruments after any custom subset, including when the subset is empty.

## Trailing exits

The trailing exit is controlled via environment variables (all optional):

- `TRAIL_ARM_PIPS` (default `8`) — pips at which trailing activates.
- `TRAIL_GIVEBACK_PIPS` (default `4`) — pips given back from the high-water before closing.
- `TRAIL_ARM_CCY` / `TRAIL_GIVEBACK_CCY` — account-currency fallback thresholds when pip math is unavailable (legacy `TRAIL_ARM_USD` / `TRAIL_GIVEBACK_USD` are still honored).
- `TRAIL_USE_PIPS` (default `true`) — prefer pip-based trailing when possible.
- `BE_ARM_PIPS` (default `6`) and `BE_OFFSET_PIPS` (default `1`) — break-even guard once in profit.
- `MIN_CHECK_INTERVAL_SEC` (default `0`) — optional rate-limit for trailing checks.

## Risk limits and exits

- `MAX_OPEN_TRADES` / `MAX_CONCURRENT_POSITIONS` (default `3`) — cap simultaneous positions across instruments.
- Initial protective orders are ATR-based:
  - `SL_ATR_MULT` (default `1.2`) and `TP_ATR_MULT` (default `1.0`) scale the latest ATR to place the stop-loss and take-profit.
  - Per-instrument overrides live in `risk.instrument_atr_multipliers` (defaults set `XAU_USD` to `sl=1.6`, `tp=0.8`).
- Time-stop failsafe (runs before entry gating):
  - `TIME_STOP_MINUTES` (default `90`) — minimum trade age before evaluation.
  - `TIME_STOP_MIN_PIPS` (default `2`) — close if pips stay below this after the time threshold.
  - `TIME_STOP_XAU_ATR_MULT` (default `0.35`) — for `XAU_USD`, minimum pips can float on ATR (`atr/pip_size * multiplier`).

## Entry confirmation

- `USE_MACD_CONFIRMATION` (default `false`) — when enabled, the standard 12/26/9 MACD must agree with the existing EMA/RSI signal. MACD only filters entries; it never opens trades on its own.

## Verbose market logging (temporary debugging)

- `VERBOSE_MARKET_LOGS=false` (default) keeps repetitive `[SCAN]`/`[SIGNAL]` lines at debug level and muted.
- Set `VERBOSE_MARKET_LOGS=true` only during short troubleshooting windows to surface per-instrument scan/signal traces.
- Revert to `VERBOSE_MARKET_LOGS=false` after debugging to reduce log noise and duplicate-looking lines.
