# Quote recording and research memory

The demo worker keeps quote capture and proposal import separate from its entry,
exit, sizing and risk code. The research catalogue does not enable instruments or
promote strategy changes. No credentials or account history belong in this file.

## Quotes

`practice_quotes.sqlite` stores sampled OANDA practice bid/ask for AUD/USD,
GBP/USD, EUR/USD and gold. The collector uses a persistent HTTP client, GET only,
an explicit demo/practice startup gate and a pinned practice host. Redirects are
not followed. Read/connection/protocol errors, malformed responses and HTTP
failures are retried with bounded backoff. Authentication failures wait five
minutes; rate limits respect a bounded numeric Retry-After. Database startup
failures retry without blocking the decision loop.

Fresh executable quotes must have valid bid/ask, a non-future broker timestamp
and age no greater than 15 seconds when received (three normal polling periods).
Non-tradable weekend responses are successful responses, not fresh price rows.
Partial/missing, stale and invalid snapshots are labelled explicitly. Invalid
snapshots cannot partially commit rows. Duplicate quote timestamps are ignored.
Historical gaps are not filled with invented prices; the recorder is sampled,
not tick-complete. The replay ignores legacy stale/future-timestamp rows.

Private `practice_quotes.status.json` and `[PRACTICE_QUOTES]` report state,
observation time, last successful response, consecutive failures, next retry,
per-instrument coverage and first/last stored quote timestamps. Status is emitted
at startup, on state changes and at least once a minute while requests complete.
Logs include only error class/status code, never exception text or response
bodies. A successful response can coexist with an empty history while markets
are closed. Check observation age as well as the state's label.

The delayed-entry replay is a timing diagnostic: it measures the first observed
positive move, changed entry and remaining target distance. It does not simulate
all later stops/exits or prove profitability. Audit interval coverage and gaps,
then freeze an experiment and test unseen data before claiming an advantage.

## Research proposals

On demo startup, `src.research_registry` validates and copies
`research/market_experiments_2026-11-14.json` to private
`memory/research/registry.json`. A receipt logs the catalogue hash, six proposal
IDs, zero validated experiments and disabled execution. Hash equality and the
running revision establish which catalogue was imported. The import preserves
old output on failure. No external news, yield, oil or crypto feed is added.

The catalogue captures the five market proposals from the saved 14 November
research and the delayed-entry idea. It excludes unverified price/news claims.
Every result is null and every status is PROPOSED. The independent promotion
protocol in LEARNING_CONTROL_PLANE.md remains unimplemented; this catalogue is
not that protocol and cannot approve any candidate.

## After-cost cohort review

The separate hourly ledger now groups matched trades by journal run tag in
`review.json`. The private `[PRACTICE-HISTORY-COHORT]` line publishes only the
allowlisted MINI_RUN group, with an explicit after-cost basis. These groups do
not widen or overwrite the learner's eligibility filters or gross-P&L inputs.
Do not assume the same sample without checking eligibility and timestamps.

## Verification

Run worker tests with network sockets blocked and the MCP read-only suite.
After deployment verify the new revision, unchanged SAFE-DEMO/EFFECTIVE-RISK
values, fresh flat practice snapshot, VERIFIED history, research import hash,
and quote response/coverage status. During closed markets, verify successful
non-tradable responses and tests of fresh payloads; do not claim live tradable
quote capture until an actual row is observed after reopening.
