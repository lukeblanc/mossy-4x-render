# PR 157 work order — startup equity drawdown recovery

## Outcome
Move the one-time demo drawdown recovery into the normal `_startup_checks()` flow, after OANDA practice equity and open trades are known. The recovery must reliably clear a stale max-drawdown state without weakening the permanent 5% guard.

## Context
PR #155 created a one-time marker before current equity was available. PR #156 attempted to gate that marker through `sitecustomize.py`, but Render revision `2e5851b07aba987508f60dc7260b6eaf0b01c47c` produced no `[DRAWDOWN-RECOVERY-GATE]` startup log and still reported `one_time_drawdown_reset=false`. Daily rollover may clear `state.max_drawdown_halt` while leaving a stale high `state.peak_equity`, so checking only the boolean halt flag can miss the stale condition.

## Boundaries
- Demo/practice only. Live remains disabled and untouched.
- Do not alter Champion, signal, session, sizing, stop-loss, take-profit, journal, adaptive, or shadow-learning rules.
- Do not change `MAX_DRAWDOWN_CAP_PCT`; it remains 5% in safe demo.
- Only run with zero open trades.
- Use a new persistent one-time marker under the same `DATA_DIR`/`/var/data` resolved by the runtime.
- Recover only when either `state.max_drawdown_halt` is true OR current equity is already at/beyond the persisted peak-equity drawdown threshold. A healthy account inside the 5% window must not have `peak_equity` moved.
- On recovery: clear the halt, set `peak_equity` to current valid equity, persist state, and log exact evidence.
- Remove the unreliable `sitecustomize.py` gate and its tests/docs from PR #156 if they are no longer needed.
- Remove this work-order file before finalizing the PR.

## Proof
Add focused tests proving:
1. Active demo halt recovers once and re-anchors to current equity.
2. Halt flag false but current equity beyond the saved 5% peak threshold also recovers once.
3. Halt flag false and current equity inside the 5% window does not move peak equity.
4. Open trades prevent recovery.
5. Live/non-practice mode is untouched.
6. Subsequent restart does not recover again because the new marker exists.
7. Existing risk-manager tests and safe-demo profile tests pass.

Expected first Render startup evidence when stale:
`[DRAWDOWN-RECOVERY] stale demo drawdown recovered equity=... previous_peak=... reason=active-halt|threshold-breached`

Expected healthy/no-op evidence:
`[DRAWDOWN-RECOVERY] no stale recovery needed ...`
