# Demo equity stop

Luke approved an AUD 500 equity floor for the OANDA practice/demo worker on
10 November 2026. This replaces the old AUD 5,000 monitoring-only warning with
an execution check. It does not enable live trading or remove other risk,
position, session, signal, or protective-stop checks.

The existing Render worker uses `MODE=demo`, `OANDA_ENV=practice` (the default),
and a persistent `MOSSY_STATE_PATH`. The code defaults to 500; the optional
non-secret dashboard override is `DEMO_EQUITY_FLOOR_AUD=500`. Invalid, non-finite,
or lower demo floors stop startup. The legacy live equity floor is unchanged.

At or below the floor, the worker persists `demo_halted_on_equity_floor` in
`risk_state.json`, blocks new entries, attempts to close existing demo positions,
and requires a fresh broker read to confirm that the account is flat. A failed
close or unavailable confirmation is retried. Existing broker protection halts
are never cleared by this path. A disk-write failure still blocks entries and
does not prevent protective close attempts.

Checks run at startup, on each heartbeat (normally every 30 seconds), and on
each decision cycle. A previously latched stop continues retrying even when NAV
is unavailable. It remains latched after equity rebounds, restarts, daily/weekly
rollovers, and demo-run or ordinary risk resets. Resetting this stop requires a
separate explicit instruction; this change provides no automatic reset.

NAV is broker-reported equity, including unrealised P/L. This is a trigger,
not a guaranteed account minimum: polling delays, market gaps, disconnections,
and closing prices can take equity below AUD 500.

The read-only bridge accepts the configured floor and latch as bounded optional
heartbeat fields without receiving exact equity, credentials, orders, or signals.
Old workers keep their original 5,000 warning until updated. A new worker refuses
to strip the new fields to satisfy an older bridge; deploy the bridge before the
worker. Commit messages skip automatic Render deployment so the tested revision
can be rolled out to those two existing services explicitly. No other worker
should be activated or reconfigured.
