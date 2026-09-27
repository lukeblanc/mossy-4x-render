# Mossy 4X learning control plane

Mossy has two deliberately separate learning lanes.

## Safety lane

The adaptive tuner and setup policy may reduce risk or block a setup. They may
never scale above the configured base risk. Their evidence is limited to the
shared clean cohort: broker-confirmed, closed, finite-P&L outcomes that have not
been invalidated as cancelled orders or duplicate aliases. Legacy
`trade_events` are not learning evidence.

The default learning cohort is pinned to the `MINI_RUN` regime beginning
2026-07-13T12:47:00Z. Blank environment values do not widen that boundary.
Entries must precede their broker-confirmed exits, and future-dated exits are
excluded at the review's as-of time.

## Edge lane

The edge learner is advisory only. It may observe, label, compare and recommend
a registered Challenger. It cannot place an order, edit strategy settings,
promote a Challenger, write GitHub, or deploy.

The first control-plane release records:

- one immutable `learning_opportunities` row for each valid completed M5 bar
  successfully observed, instrument, revision and configuration fingerprint;
- one immutable `runtime_decision_events` row for each exact evaluation event
  referring to that opportunity;
- HOLDs, the first blocking gate, executions, order failures and evaluations
  skipped after uncertain order state;
- only a sanitized aggregate across the MCP boundary. Raw rows, signals,
  prices, account data, identifiers and credentials remain local.

The observer writes to a separate `learning_observations.db` through a bounded,
fail-open queue. This prevents learning I/O from delaying an entry. A clean
shutdown can flush the queue, but abrupt container loss can discard queued
observations. The Agent sees aggregate submitted, persisted, duplicate, pending,
drop and write-error counters. An `active` capture state means only that no
queue or storage fault has been detected since process start; it is not proof
that every market opportunity was observed. Missing evidence is never treated
as a pass.

## Evidence states

- `COLLECTING`: evidence is incomplete or only covers executed-trade subsets.
- `WAIT`: a registered test is still accumulating prospective evidence.
- `REJECT`: one or more performance or safety gates failed.
- `READY_FOR_LUKE_REVIEW`: reserved for a later protocol in which an
  independent, immutable registry and frozen referee verify every statistical,
  lineage and forward-shadow gate. This is still only a review request, never
  permission to change the Champion.

Current executed-trade subset analysis is always `COLLECTING`. It cannot prove
that a relaxed filter would create safe additional trades.

This v1 control plane also forces prospective reports to remain `COLLECTING`
with `promotion_protocol_not_implemented`. A report cannot self-attest its way
to READY: the independent registry/referee protocol must be built and reviewed
first.

## Promotion boundary

A future prospective Challenger can reach review only when all of these are
true:

- immutable registered definition and matching revision/config lineage;
- evidence collected after registration under a frozen referee;
- false-discovery control at 5% or stricter;
- at least 250 labelled outcomes, including at least 100 untouched validation
  outcomes across at least three pairs;
- positive after-cost expectancy and validation profit factor at least 1.10
  and no lower than the Champion;
- validation drawdown no worse than the allowed Champion bound;
- a separate forward-shadow pass with zero safety breaches.

Luke must approve the exact candidate, evidence bundle and reviewed commit
before any Champion change or deployment. Automatic promotion remains
disabled. The learning agent has no route for trading, configuration writes,
GitHub writes, deployment or promotion.

## Next implementation phase

Observation rows are the evidence foundation. The next Challenger must add a
future-only labeler using completed bid/ask candles, net R, MFE/MAE and
conservative same-bar stop/target handling, followed by purged chronological
walk-forward evaluation and an untouched holdout. Until those pieces exist,
the Agent must describe the prospective learner as collecting data, not as a
validated self-improving strategy.
