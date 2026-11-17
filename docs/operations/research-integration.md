# Shared research evidence

The practice-history monitor refreshes a private research digest after each
successful hourly/startup accounting sync. It joins the same eligible journal
cohort used by the learner to the verified after-cost broker ledger. Original
journal P/L, strategy rules, risk settings and the existing learner remain
unchanged. No broker or GitHub writes are available in the digest module.

## Evidence and limits

- Same-account fingerprint, AUD currency, verified history, source age <=75
  minutes, and a consistent read-only SQLite snapshot are required.
- Trade ID, instrument, side, entry/exit chronology, run tag, gross P/L and net
  cost arithmetic must match. Excluded matches are counted, never silently
  treated as losses, wins or an empty complete sample.
- The full matched cohort, recent 80, and a 70/30 chronological partition are
  described in AUD after costs. Spread already exists in fill-based P/L and is
  never subtracted again. Commission allocations retain their estimate flags.
- Existing shadow-filter definitions are reused for historical comparisons.
  Missing RSI, trend or session features are flagged. These are executed-trade
  subsets, not a replay of untraded opportunities, prospective validation or
  proof of a superior strategy. No automatic recommendation or promotion occurs.
- Quote status is a dated snapshot, never an executable signal or a completed
  gap audit. Read fresh PRACTICE_QUOTES logs separately for current coverage.
- The imported proposal catalogue must match its committed hash. Its six
  proposals still have zero validated experiments. Oil/crypto and external
  dollar/yield inputs are not created by this integration.

Private files: `memory/research/digest.json` and `digest-status.json` on the worker
disk. A failed refresh preserves the previous report but writes an unavailable
status. Always inspect the newest status/log before using an older successful
report. A failure of this optional module cannot interrupt accounting or trading.

Read `[RESEARCH_DIGEST]` for the aggregate join, cost results and blockers.
Read `[RESEARCH_CANDIDATE]` for the twelve historical filter comparisons, matched
to the digest generation time. These private log summaries contain no account
IDs, credentials, individual trade IDs or balances. They are not published to
the public monitoring bridge or GitHub.

## Task handoffs

The four existing companion tasks share a private work board:

1. News Paper Test keeps its frozen protocol and canonical study ledger. It adds
   a short handoff with its latest study result, sources and missing inputs.
2. Trade Watch records newly verified practice fills and material evidence
   blockers without treating those fills as registered news-study observations.
3. Friday Research checks the existing queue before adding a sourced, falsifiable
   hypothesis, required inputs and a proposed test. It does not claim an imported
   proposal has been run.
4. Monday Review combines fresh runtime health, the after-cost digest, paper-study
   outcomes and research handoffs, and selects the next evidence-supported test
   or repair for Luke. It does not silently deploy or change trading rules.

The work board is a coordination record, not a source of broker truth or an
executable instruction channel. Every task rereads source timestamps and keeps
proposals, implemented code, observed runs and validated outcomes distinct.
Updates preserve earlier entries with optimistic version checks; conflicting
writes are reread and merged by unique event ID, never overwritten blindly.
