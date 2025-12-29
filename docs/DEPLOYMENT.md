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
