# Private OANDA after-cost history

The practice worker now keeps a separate research ledger on its existing disk.
It reads OANDA with GET only, on startup and once an hour in a daemon thread.
The decision loop, order handling, risk settings and existing gross-P&L learning
cohorts are unchanged. No extra service, credentials or paid subscription is used.
`MODE=demo` and `OANDA_ENV=practice` are required; the host is pinned to OANDA
practice and redirects are disabled. `PRACTICE_HISTORY_ENABLED=false` disables
the monitor at the next restart.

## Accounting

An existing `memory/oanda-practice/<account fingerprint>/history-*/trade_history.sqlite`
archive is read as an immutable seed. Later runs request only IDs after the stored
cursor, in batches of at most 1,000, through a fixed account-summary cursor.
If no seed exists, the same bounded ID pagination starts at account creation.
IDs must be complete and unique from creation. Account, currency, chronology,
every balance checkpoint and the fixed-cursor account balance must match.
At most 20,000 new transactions are fetched per run, with a 60-second request
start budget and five-second HTTP timeouts. Exceeding a limit is an error, not
permission to silently truncate. Another run retries from the unchanged cursor.

Closed trades include all partial reductions, entry and exit commission, signed
fill financing, daily financing and guaranteed execution fees. Financing and fees
must reconcile to their broker breakdowns. Spread and currency-conversion detail
are not debited again. Spread is already in executed-price P&L. Unsupported cash
events fail closed. Deposits/withdrawals are excluded from performance.

OANDA reports commission for an order fill, not each trade within that fill.
For fills involving multiple trades, commission is allocated in proportion to
absolute filled units, preserving the exact total. Such trades are labelled
`ALLOCATED_BY_UNITS`; per-trade results and win/loss counts using this allocation
are estimates, even though the account result reconciles exactly. Single-trade
commission is `BROKER_DIRECT`. All calculations use Decimal. Open-trade charges
remain separate from closed-trade performance and are not unrealised P&L.

Execution quality compares the fill to OANDA's contemporaneous bid/ask snapshot,
not to a quote at the original decision time. Missing snapshots remain missing.
Requested stop/target settings and subsequent protection-order events are retained
as history, not asserted to be currently active. Existing journal context is
attached only when broker trade ID, instrument and entry timestamp match.
Missing reasons/versions remain unknown; new entries record the runtime revision
inside their existing gating metadata. Nothing rewrites old journal P&L or
automatically changes the Champion.

## Files and observation

Within the private account directory, `enriched-latest.json` points to:

- `enriched/history.sqlite`: canonical transactions, derived trade records,
  review and cursor, committed together under an exclusive lock.
- `enriched/review.json`: account/closed/open metrics, recent 80, instrument and
  fill-quality comparisons, coverage, timestamps and limitations.
- `enriched/trades.csv` and `enriched/fills.csv`: private inspection exports.
- `enriched/status.json`: latest attempt; read this with the report's generation
  time. A failed attempt leaves the last successful report in place, marked stale
  by its timestamp and the latest failure status. Never present it as a fresh sync.

The directory is mode 0700 and files mode 0600. Nothing is uploaded to GitHub or
the public monitoring bridge. CSV/JSON files are regenerable views; SQLite is the
canonical atomic snapshot. Logs use `[PRACTICE-HISTORY]` and omit credentials and
account IDs. They expose aggregate reconciliation and coverage for authorised
private Render monitoring. A failure cannot halt trading or clear risk state.

For an authorised one-off rebuild, run
`MODE=demo OANDA_ENV=practice python -m src.practice_history` on the worker.
The shell does not necessarily inherit the application's resolved environment;
these values apply only to this GET-only command. This is read-only towards
OANDA and the original journal/seed archive.
After deploy, check the running revision, a VERIFIED history log with zero
allocation residual, unchanged risk limits, and healthy ordinary cycles.

Implementation validation: 1,005 tests passed and one skipped locally, including
cost allocation, financing reconciliation, partial closes, gap/duplicate rejection,
incremental restart, no-op retry, preservation after failed sync and live-mode
blocking. A private one-off run against the saved archive reconciled successfully;
its second run reproduced the same totals without modifying the original archive
or journal. Open entry context is joined independently of exit confirmation;
resolved duplicate/cancelled journal rows are excluded. Verify the deployed
revision and current sync status in private Render logs.

Primary specifications:
https://developer.oanda.com/rest-live-v20/transaction-df/
https://developer.oanda.com/rest-live-v20/transaction-ep/
