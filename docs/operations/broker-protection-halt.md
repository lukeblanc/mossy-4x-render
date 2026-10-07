# Broker protection halt: diagnosis and review

The Singapore demo worker uses two independent signals: scheduler liveness and
permission to open trades. A live scheduler with no open positions can still have
`broker_entry_halted=true`. Read `broker_entry_halt_reason` in `/status` or the
`HEALTH_STATUS` log; observing this property does not clear the latch.
The existing `/status` route may refresh broker positions and run the protection
audit, which can emergency-close an unsafe trade. Only the halt property itself
is side-effect-free; prefer already-emitted logs for strictly read-only inspection.

## Protection evidence

`[BROKER][PROTECTION-AUDIT]` logs a specific `audit_reason` before emergency closure.
A missing stop, a cancelled/mismatched stop, unavailable broker evidence, and a
valid stop above the cash-risk limit are distinct conditions. Only selected
fields are logged; account IDs, credentials and raw broker responses are excluded.

The legacy `unprotected-open-trade` latch category is retained for compatibility.
It does not, by itself, prove that a stop was missing. The cash-risk diagnostic
includes stop ID, units, entry/stop prices, current loss-side conversion factor,
calculated stop exposure and the effective cash limit.

## Conversion headroom

New positions whose quote currency differs from the account currency use at most
98% of the configured cash limit for sizing. With the default A$0.50 limit, the
sizing budget is A$0.49 before units are rounded down. A smaller percentage-risk
request remains smaller, and stricter configured cash limits remain stricter.

The broker still audits against the unchanged hard maximum of A$0.50 (or the
stricter configured limit). Conversion changes exceeding the small reserve can
still trigger the guard. Headroom reduces sensitivity to small conversion moves;
it is not a guaranteed maximum realized loss or proof of profitability.

## Recovery is consequential

Normal open-trade polling and confirmation of a closed trade do not clear the
persisted entry halt. This change intentionally does not add automatic per-cycle
recovery. A failed deletion of the persisted marker also leaves the in-memory
halt active.

The existing startup `connectivity_check()` is **not read-only**: it can attempt
to close unsafe trades and can clear a retained halt following a clean protection
audit. Restarting or redeploying can consequently resume demo entries. Do not use
that method merely to check status or delete the halt marker to bypass the audit.

Before approved recovery:

1. Read exact broker transaction/order evidence for the affected trade and stop;
   reconcile confirmed closure, current exposure and pending entry orders
2. Preserve the demo/practice environment, protective stops, cash cap, position
   limit and persisted daily/weekly/drawdown state
3. Review the specific audit reason. Missing historical diagnostic inputs mean
   the historical cause remains unverified
4. Obtain approval for the controlled deployment/restart or other recovery action
5. After recovery, verify the audit/recovery log and halt field, journal consistency,
   intended sizing and confirmed broker stop before claiming entries are healthy

The September 28 incident had a verified stop at entry and a later generic audit
failure. Conversion drift is a plausible explanation, not a confirmed root cause
without the failing audit payload or broker transaction evidence.

## Read-only MCP monitoring compatibility

The optional heartbeat fields are `broker_entry_halted` and a finite allowlisted
`broker_entry_halt_reason`. Unrecognized persisted reasons are reduced to `other`;
raw text is not transmitted. A new bridge treats an older worker's absent halt
state as unknown and blocks an all-clear supervisory result.

Before sending these fields, the worker checks an authenticated GET capability
response on the configured internal heartbeat URL. An older bridge's 405 response
now returns `unsupported-heartbeat-capabilities` without posting or stripping halt
fields. The legacy payload downgrade has been retired. Failed negotiation does
not update the successful-delivery throttle, so the next publish attempt can
retry immediately after an approved bridge upgrade. Consult worker logs while
monitoring is unavailable; failed publication does not alter trading behavior.
Authentication errors, malformed capability responses, and heartbeat validation
failures are not downgraded or treated as successful delivery.

No new credential or status destination is introduced. The capability endpoint
reports field support only; it cannot clear a latch or submit a trade. Both bridge
and worker changes require their own approved rollout; this patch deploys neither.
