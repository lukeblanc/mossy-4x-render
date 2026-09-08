# Trade recording and historical reconciliation

The worker records a new entry only from an OANDA `orderFillTransaction` with
a valid `tradeOpened` object. The journal uses its trade ID, execution price,
opened units and fill timestamp. Order-create IDs, protective-order IDs and
`lastTransactionID` are not substitutes. Cancelled orders are not entries.
An uncertain response stops further orders in that cycle; the next cycle reads
broker positions before considering another entry.

This follows the official OANDA [order response](https://developer.oanda.com/rest-live-v20/order-ep/)
and [fill definitions](https://developer.oanda.com/rest-live-v20/transaction-df/).

Close matching requires the exact trade ID and instrument. A missing exact
entry does not authorize writing onto another trade in the same instrument.
Legacy order IDs are resolved only through the broker's order and linked fill.
An order that only closed or reduced another trade cannot identify a new entry.
If a canonical broker-ID row already exists, its legacy alias is left for audit
so the same broker outcome cannot be counted twice.

Historical recovery requires a closed trade, finite realized P/L, a positive
closing price and a real close time at or after the recorded entry. Missing
evidence stays unresolved. Today's spread and account equity are not inserted
into historical closes. See OANDA's [trade fields](https://developer.oanda.com/rest-live-v20/trade-df/).

Historical lookup work defaults to three rows per cycle and is capped at five,
even if `JOURNAL_RECONCILE_ATTEMPTS_PER_CYCLE` requests more. Unsuccessful rows
wait 900 seconds before another attempt in the same process. This backoff is
separate from active-position protection, which continues every cycle.
Candidate rotation now reaches beyond the previous 500-row cutoff. Unbounded
transaction-history scans are removed; lookup status and recovery counts are
logged under `[JOURNAL][LOOKUP]` and `[JOURNAL][RECONCILE]`.

This release repairs recording and reduces repeated lookup work. It does not
validate a new entry strategy, reconcile every legacy row, or establish positive
expectancy. The existing practice-only guard, cash caps, daily/weekly/drawdown
controls, approved demo baseline and strategy/session settings remain intact.
