# Deployment Notes

## Instrument Configuration

- Omit the `instruments` key or set it to `null`/`None` to use the default instrument basket.
- Provide an empty list (`[]`) to pause trading entirely.
- Set `merge_default_instruments` to `true` to append the default instruments after any custom subset, including when the subset is empty.

## Trailing exits

The trailing exit is controlled via environment variables (all optional):

- `TRAIL_ARM_PIPS` (default `8`) — pips at which trailing activates.
- `TRAIL_GIVEBACK_PIPS` (default `4`) — pips given back from the high-water before closing.
- `TRAIL_ARM_USD` / `TRAIL_GIVEBACK_USD` — USD fallback thresholds when pip math is unavailable.
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
