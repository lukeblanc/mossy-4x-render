# Optional ten-entry A$10 practice experiment

Default: disabled. No render.yaml, strategy, learning, cash-risk, daily-limit or
position-limit setting is changed by this feature. The latest approved plan is
ten NEW A$10 AUD_USD positions, then restore the unchanged Champion sizing. There
is no A$2 stage or automatic restart of the quota.

MOSSY_PRACTICE_EXPERIMENT_ID opts a worker into a separately prepared ledger at
MOSSY_STATE_PATH/practice_experiment.sqlite. Merely deploying the code creates
or activates nothing. Configuring an ID without its matching active ledger
blocks entry. The ledger is bound to a hash of the practice account ID.

The broker applies an additional cap of exactly ten AUD base units, equivalent
to A$10 notional, only when the normal strategy/risk size permits at least ten
units. It never rounds a smaller permitted size up. Existing strategy filters,
learning reductions, technical stops, one-position and twenty-per-day gates
still apply. The experiment refuses non-demo/practice mode, other instruments,
or cash-risk/software-close limits above A$0.20.

Before every submission, a durable SQLite transaction reserves one slot.
It then verifies the exact AUD account, zero open trades/positions/pending
orders, a fresh empty open-trade list, exact closure of the experiment's last
trade, and account-specific AUD_USD minimum size and unit precision. Failures
before submission release only that known-unsent reservation. A returned
opening must have a trade ID newer than the preflight transaction watermark.

An exact new opening consumes a slot, even if later protection verification
requires emergency closure. Partial/wrong-size fills are closed and halted,
not accepted as approximate A$10 positions. On the tenth confirmed opening,
the ledger becomes complete and cannot be reactivated. Once the usual one-open-
position gate allows another entry, the broker resumes the unchanged Champion
sizing across restart and midnight. Exit/stop management continues for the
final experiment position. The broker also performs a fresh flat-account check
and requires exact positive closure evidence for that tenth position before it
allows the first or any later Champion submission through the completed plan.
History before this experiment does not consume its slots, but still counts
toward the normal daily cap. Eleven earlier entries leave at most nine new
entries that day; the tenth waits for a later permitted day.

No automatic recovery is provided for an uncertain submission, rejected order,
lost response, interrupted reservation or failed confirmation write. It remains
reserved and prevents another submission until a separately reviewed recovery.
Missing/corrupt/mismatched ledgers fail closed. Existing broker entry halts are
never cleared by prepare, activate, pause or normal polling.

## Approval and operation

The class src.practice_experiment.PracticeExperiment exposes prepare(),
activate(), status() and pause(). They are local operator operations, not HTTP
endpoints or startup hooks. prepare() creates an inactive plan and refuses to
overwrite the same ID; activate() accepts only an unused prepared plan;
pause() persistently blocks new entries. A paused/completed plan cannot be
rearmed by activate(). No actual operational ledger is included in this patch.

Before separately approved activation: preserve current configuration, risk
state, journal and entry-halt evidence; verify practice account identity and
AUD currency, current instrument metadata supporting ten units, zero exposure
and pending orders, durable writable storage, and the unchanged A$0.20 limits,
20/day and one concurrent position. Obtain explicit deployment/activation and
incident-recovery authorization. Prepare a unique plan on that persistent path,
verify its inactive status, configure the matching ID, and activate only at
the approved point. These steps do not bypass a retained broker halt.

After approved activation, use status() for count/state/pending observations
without broker calls, plus the existing fill/stop logs. At complete, review
the journal and outcomes; the immutable ledger does not create another plan and
normal Champion sizing becomes eligible automatically once the final position
is confirmed closed and normal gates pass. Pausing before completion remains
fail-closed; removing the experiment ID is not the approved way to pause it.

Rollback: pause the ledger first through a separately approved operation; let
existing protection/exit handling finish or use an explicitly approved safe
exposure-management procedure. Only once entries are otherwise disabled and
exposure is resolved should an approved rollback remove the feature or ID.
Retain the ledger and latest risk/journal state. Prefer the reviewed safety-only
revision, which retains persisted halts. The old a8c7667 startup can clear a
halt automatically and is unsafe as an unattended resume procedure.

No new paid service is needed; the ledger uses Python's SQLite. Broker spreads,
commission and financing terms have not been queried, so exact trade costs are
unverified. A$0.20 remains the stop-risk/software trigger, not a target loss or
guarantee of realised exit price. Small A$10 positions with normal technical
stops will generally have less planned exposure. No profit or completion time
is promised; normal gates may delay or stop the experiment.
