# Audited demo journal cleanup

Older versions recorded some order IDs as trades even when OANDA cancelled the
order. PR #162 prevents new entries without an actual opening fill. This
maintenance operation classifies the existing backlog using broker evidence.

Set `MOSSY_JOURNAL_CLEANUP_ID` to a new, descriptive run ID to request a startup
sweep. It runs only with `MODE=demo`, OANDA practice, and a freshly confirmed flat
broker account. The ordinary risk state and trading limits are preserved. Clear
the request after completion. Updating Render environment variables starts a new
deployment; a separate deploy request is unnecessary.

Before corrections, the operation saves a standalone SQLite backup under
`/var/data/journal-backups/before-<run-id>.sqlite`. The SQLite backup API includes
committed WAL data. Failure to create the backup stops the operation before any
correction or broker request.

The operation freezes the unresolved entry IDs, with a maximum of 500 per run,
and records progress in `journal_cleanup_runs`. A restart with the same ID resumes
an interrupted run; a completed ID does not run again. Broker authentication,
rate-limit or server failures defer the run. Requests are read-only and reuse one
broker connection.

Resolution rules:

- A cancelled market order must match the journal instrument and entry time
  within five minutes, contain no fill linkage, and have a matching OANDA
  `ORDER_CANCEL` transaction. Its reason is logged and the evidence is saved.
- A duplicate order-ID alias requires an explicit link to an actual trade and a
  matching canonical journal row. The alias is retained as audit history.
- A recovered close requires an exact broker trade, confirmed closed state,
  valid price, PnL and close time. It uses the existing exact-fill reconciliation
  checks, including entry/exit chronology and duplicate-outcome protection.
- Insufficient or conflicting evidence leaves the entry unresolved. Completion
  means every frozen candidate was checked, not that every candidate was resolved.

Cancelled orders and duplicate aliases are classified in
`journal_entry_resolutions`, with the complete original row and broker evidence.
The original trade rows and event history are not deleted or assigned invented
closing prices or PnL. Reconciliation, open-trade matching and report open counts
exclude the audited classifications. Reports display their separate audit counts
and exclude unconfirmed closes from verified performance figures.

Verify `[JOURNAL-CLEANUP][COMPLETE]`, including any `unresolved_ids`, and the next
`[ALGO-REPORT][STARTUP]` report. After clearing the request, verify that the classifications
survive restart, broker/risk checks succeed, and ordinary decision cycles resume.
Preserve the backup and audit tables for later inspection.

Some recent, confirmed opening trade IDs returned 404 from OANDA's individual
trade endpoint during the first cleanup. A bounded fallback requests
`GET /trades?ids=<exact-id>&state=ALL&count=1`. It accepts exactly one matching ID,
then applies the same instrument, closed-state, price, PnL and chronology checks.
The fallback is shared by ordinary reconciliation and maintenance. It never
scans all transactions or substitutes another trade from the same instrument.
