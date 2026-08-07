# PR deployment checklist

1. Merge this PR into `main`.
2. Allow Render to auto-deploy the new main revision.
3. On startup, verify one of:
   - active halt: `[DRAWdown-RECOVERY-GATE] active halt detected; reset armed` followed by `one_time_drawdown_reset=true` and the risk reset log;
   - no active halt: `[DRAWDOWN-RECOVERY-GATE] no active halt; reset suppressed` and `one_time_drawdown_reset=false`.
4. During an allowed session, confirm `blocked_risk=0` and no `Skipping ... due to max-drawdown` unless a fresh legitimate drawdown occurs.
