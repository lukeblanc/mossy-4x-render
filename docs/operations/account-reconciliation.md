# Read-only account reconciliation

## Scope and interpretation

This change adds independent accounting evidence to the existing weekly operations
reporter. It does not change the Champion, scheduler cadence, entries, exits,
position sizing, risk limits, learning, trade history, or broker positions.
OANDA requests are GET-only, pinned to the practice API, with redirects disabled.
`MODE=demo` and `OANDA_ENV=practice` must both be explicit. The existing
`OANDA_API_KEY`/`OANDA_API_TOKEN` and `OANDA_ACCOUNT_ID`/`ACCOUNT_ID` environment
variables are reused; there is no additional paid service or new API credential.

A report's closed-trade P&L is NOT automatically the entire account return.
Financing, commissions, guaranteed execution fees, dividends, deposits and
withdrawals need their own accounting. Balance is not NAV when trades are open.

The 28 September review compared Monday balance observations with a report ending
on Sunday evening. Its approximately AUD 9.02 difference is an investigation
lead, not proof of missing money, financing expense, or a journal defect. The new
code makes that comparison only on identical account and UTC boundaries.

## Evidence contract

Both journal and broker calculations use **(start, end]**, with explicit UTC
boundaries and nanosecond-aware membership for broker timestamps. Each report:

1. Fetches an AUD account summary and keeps balance and NAV separate.
2. Fetches the unfiltered account transaction index through the report end, then
   every page. The index defaults to account creation; the unfiltered count,
   unique numeric IDs, account identity, page ranges and chronology are checked.
3. Obtains actual `accountBalance` checkpoints immediately before each reporting
   boundary. Missing opening evidence is unknown, never assumed to be zero.
4. Adds `ORDER_FILL.pl`, signed financing, dividends and transfers; subtracts
   positive commission and guaranteed execution fee costs. `halfSpreadCost` and
   nested breakdowns are not subtracted a second time. Each intermediate broker
   balance and the final change must reconcile within AUD 0.01.
5. Matches exact closed trade IDs, instrument, realised P&L and close timestamps
   to broker-confirmed journal rows. Read-only SQLite mode is used; missing DBs
   are not created, old rows are not repaired, and audited resolution rows are
   excluded. A matching aggregate alone is insufficient.

A report is `VERIFIED` only when BOTH ledger accounting and journal matching pass.
`UNVERIFIED` includes unknown transaction types, malformed money, missing pages,
unknown opening balances, stale/mismatched summary cursors, mismatching IDs or
amounts, and unreadable or incomplete journal evidence. Partial reductions need
explicit lifetime allocation and remain `UNVERIFIED`; they are not forced into
full-close statistics. Transactions outside the bot can therefore cause a visible
journal mismatch rather than silently being attributed to the Champion.

Decimal is used for accounting; tolerance is a fixed AUD 0.01, not a percentage.
`account_net_excluding_transfers` is balance movement minus external transfers,
not a mark-to-market return, risk-adjusted result, or proof of strategy edge.
A later current balance is not substituted for a historical closing checkpoint.

## Execution and persistence

The existing reporting thread performs one read-only accounting audit on startup
and includes reconciliation in its normal weekly report. There is no additional
scheduler, agent session, decision-path network request, or automatic correction.
Reports retain the existing trade metrics and trading-readiness classification;
accounting verification is a separate, explicit field.

Network work is limited to 20 pages of at most 1,000 transactions, a 60-second
request-start budget and a 5-second HTTP timeout. Exceeding a budget yields
`UNVERIFIED` rather than a partial result. Large future histories will need a
separately reviewed incremental archive; this version does not truncate silently.

Checkpoint reports are atomically saved with mode 0600 in
`<ALGO_REPORT_DIR>/account-reconciliation/` (normally
`/var/data/algo-reports/account-reconciliation/`). UTC timestamp and content-hash
filenames preserve distinct evidence instead of overwriting the latest report.
They contain a hashed account fingerprint, not the account ID or token. Raw
transactions, tokens and broker response errors are never published by this code.
Existing optional GitHub report publishing remains unchanged.

Startup/weekly logs emit `[ACCOUNT-RECONCILIATION]` with verification status,
window, balance change, result excluding transfers, unexplained delta, journal
P&L delta and safe reason codes. An unavailable audit cannot stop trading or
reset risk state. A transient close not yet journalled may correctly appear
unverified until a later run; it must not be auto-repaired from that discrepancy.

## Release and acceptance

This branch requires Luke's approval to merge and his normal deployment approval.
It is not a deployed or profitability-validated change. After release, verify:

- the running revision matches the approved commit;
- the new log appears without credentials/account IDs;
- the existing AUD 0.50 protections, one position, eight entries/day and
  `SHADOW_AUTO_APPLY=false` are unchanged;
- archived checkpoints match the report's exact window;
- status is VERIFIED only with no unresolved ledger/journal reason, or the
  specific UNVERIFIED reason remains visible for investigation;
- all original journal rows remain unchanged and ordinary cycles remain healthy.

Do not reset drawdown, increase risk, change exits or claim the earlier approximate
difference has been explained just because code/unit tests pass. The scheduler
misfire repair remains a separate engineering change.

## Primary references

- https://developer.oanda.com/rest-live-v20/transaction-df/
- https://developer.oanda.com/rest-live-v20/transaction-ep/
- https://developer.oanda.com/rest-live-v20/account-ep/
