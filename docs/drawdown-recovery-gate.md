# Demo drawdown recovery gate

PR #155 introduced a one-time demo drawdown recovery marker. This follow-up gates that migration against the persisted `max_drawdown_halt` state before application configuration loads.

- Active demo/practice halt: the marker is removed so the existing reset runs once.
- No active halt: the marker is ensured so a healthy peak-equity baseline is not moved.
- Live/non-practice modes: no action.
