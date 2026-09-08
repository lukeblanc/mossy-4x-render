# Starting a fresh demo drawdown run

`MOSSY_DEMO_RUN_ID` is an explicit operator request to re-anchor the maximum
drawdown peak to current broker NAV and release the previous maximum drawdown
halt. It is not an automatic recovery setting. Use a new ID only after the
account owner has approved starting another demo test.

At startup the request requires demo mode, OANDA practice, valid positive NAV,
a broker-confirmed count of zero open trades, and readable/writable risk state.
The ID must contain 1–96 ASCII letters, digits, dots, underscores or hyphens,
starting with a letter or digit. Blank means no request.

The reset and its audit are saved together in the existing atomic risk-state
write. `risk_state.json` retains every consumed ID in `demo_runs`, with its UTC
start time, starting equity and the preceding risk state. Earlier audits remain
alongside later ones. A save failure restores the old in-memory state and keeps
entries blocked until persistence recovers.

The same ID cannot reset the peak or clear another halt on any later restart.
After confirming `[DEMO-RUN] ... status=applied`, clear the environment value
and verify the next startup preserves the new peak and all limits. On Render,
an environment update itself triggers a deployment.

Daily and weekly baselines, realized results, entry counts, cooldowns, position
limits and the journal are preserved. This action does not reconcile historical
journal rows, guarantee a trade, or demonstrate profitability. Normal signal,
session, spread and loss controls continue to govern entries. The existing
`RESET_MAX_DRAWDOWN_HALT` and `RESET_WEEKLY_LOSS_CAP` flags remain disabled by
the Render safe-demo profile.
