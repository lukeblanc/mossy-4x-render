# Private, on-demand demo trade handoff

The existing worker logs a bounded `[DEMO_TRADE_HANDOFF] ` JSON snapshot after
each heartbeat's fresh open-trade audit. It uses the same authenticated broker
reads that verify protective stops. No new broker endpoint, credentials, order
function, public HTTP route, or trading signal is added.

`Broker.demo_trade_handoff_evidence()` is a side-effect-free copy of the last
complete, identity-bound list audit. Each list read invalidates old evidence;
failed, partial or conflicting audits cannot expose a previous successful
snapshot. Captured fields are an allowlist of trade identity, current open
units, broker fill/opening time and linked stop/target order identity, state and
price. Broker account identifiers, credentials, balances and raw responses are
excluded. Successful flat is distinct from unavailable evidence.

The pure builder in `src/demo_trade_handoff.py` requires explicit demo/practice,
verified AUD currency, a worker revision, a source audit no more than 60 seconds
old, clear halt flags, the active demo floor and complete actual protection
evidence. It emits at most ten trades. A current stop may have trailed past the
original fill; destination quote checks determine whether those levels can
still form a valid draft. A missing target is accepted as absent only when
the exact broker detail and summary jointly verify its absence.

Snapshots stay in the existing private Render service logs. They never enter
`BOT_STATE`, the public status route or the aggregate-only MCP publisher. The
companion reads these logs through Luke's existing authenticated Render app,
scoped to the confirmed worker. It verifies deployment provenance and recalculates
freshness before using them. Log retention and periodic snapshots do not form a
durable per-trade event stream; an already closed trade is not a live handoff.

`handoff_ready` describes source evidence only. `destination_order_ready` is
always false. Before any requested FOREX.com draft, verify that broker's current
quote, account mode/currency, contract value, lot step/minimum, costs, margin,
stop distance and Luke's chosen maximum loss budget. Never copy OANDA units or
historical fill prices as destination sizing or an executable market entry.
One-click trading must be visibly off. The assistant may prepare only a new,
unsubmitted ticket; Luke personally reviews and clicks the final broker Trade
button. A chat confirmation never submits it.

The existing AUD 500 demo equity stop, live-trading block, all broker protection
checks and the read-only aggregate monitor remain in place. This change does
not loosen strategy filters or create a background copying service.
