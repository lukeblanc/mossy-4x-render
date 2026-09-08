from copy import deepcopy
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from app.broker import Broker, read_trade_details
from src.journal_cleanup import run_cleanup
from src.journal_reconciler import JournalReconcilerProfitProtection
from src.trade_journal import TradeJournal
from src.weekly_ops_report import build_weekly_report
from test_journal_reconciler import _record_entry
from test_trade_detail_fallback import client_for


def evidence():
    opening = {"id": "101", "accountID": "account", "type": "ORDER_FILL", "instrument": "AUD_USD",
               "time": "2026-07-13T01:00:00Z", "tradeOpened": {"tradeID": "101", "units": "100", "price": "0.66"}}
    closing = {"id": "105", "accountID": "account", "type": "ORDER_FILL", "instrument": "AUD_USD",
               "time": "2026-07-13T01:01:00Z", "tradesClosed": [{"tradeID": "101", "units": "100",
                    "price": "0.659", "realizedPL": "-0.75"}], "pl": "9999"}
    return opening, closing


def responses(opening, transactions, latest="110"):
    return [(404, {}), (200, {"trades": []}),
            (200, {"transaction": opening, "lastTransactionID": latest}),
            (200, {"transactions": transactions})]


def test_exact_transactions_use_per_trade_pnl_and_a_bounded_range():
    opening, closing = evidence()
    requests = []
    with client_for(responses(opening, [opening, closing]), requests) as client:
        details = read_trade_details(client, "account", "101")
    assert details["realizedPL"] == "-0.75" and details["averageClosePrice"] == "0.659"
    assert details["_close_evidence"]["closing"] == closing
    assert len(requests) == 4 and all(r.method == "GET" for r in requests)
    assert dict(requests[-1].url.params) == {"from": "101", "to": "110", "type": "ORDER_FILL"}


def test_transaction_range_never_grows_beyond_1000_ids():
    opening, closing = evidence()
    requests = []
    with client_for(responses(opening, [opening, closing], latest="999999"), requests) as client:
        assert read_trade_details(client, "account", "101") is not None
    assert requests[-1].url.params["to"] == "1100"


@pytest.mark.parametrize("change", [
    "wrong_open_id", "wrong_open_account", "wrong_trade_link", "missing_open_price", "wrong_close_id",
    "wrong_close_instrument", "wrong_close_account", "before_entry", "missing_close_price",
    "nan_pnl", "wrong_units", "duplicate_close", "partial_reduction", "no_close", "range_overflow",
])
def test_insufficient_or_conflicting_transactions_cannot_create_a_close(change):
    opening, closing = evidence()
    transactions = [opening, closing]
    if change == "wrong_open_id": opening["id"] = "999"
    elif change == "wrong_open_account": opening["accountID"] = "different"
    elif change == "wrong_trade_link": opening["tradeOpened"]["tradeID"] = "999"
    elif change == "missing_open_price": opening["tradeOpened"].pop("price")
    elif change == "wrong_close_id": closing["tradesClosed"][0]["tradeID"] = "999"
    elif change == "wrong_close_instrument": closing["instrument"] = "GBP_USD"
    elif change == "wrong_close_account": closing["accountID"] = "different"
    elif change == "before_entry": closing["time"] = "2026-07-12T01:01:00Z"
    elif change == "missing_close_price": closing["tradesClosed"][0].pop("price")
    elif change == "nan_pnl": closing["tradesClosed"][0]["realizedPL"] = "NaN"
    elif change == "wrong_units": closing["tradesClosed"][0]["units"] = "50"
    elif change == "duplicate_close": transactions.append(deepcopy(closing))
    elif change == "partial_reduction": opening["tradeReduced"] = {"tradeID": "101", "units": "50"}
    elif change == "no_close": transactions = [opening]
    elif change == "range_overflow": closing["id"] = "999"
    with client_for(responses(opening, transactions), []) as client:
        assert read_trade_details(client, "account", "101") is None


def test_transaction_endpoint_failure_is_deferred():
    opening, _ = evidence()
    values = responses(opening, [])
    values[-1] = (503, {})
    with client_for(values, []) as client:
        with pytest.raises(RuntimeError):
            read_trade_details(client, "account", "101")


@pytest.mark.parametrize("maintenance", [False, True])
def test_verified_transaction_evidence_and_close_are_saved_once(tmp_path, maintenance):
    opening, closing = evidence()
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "101")
    broker = Broker.__new__(Broker)
    broker.account, broker.key, broker.mode = "account", "test-key", "demo"
    values = responses(opening, [opening, closing])
    if maintenance:
        values = [(404, {})] + values  # The cleanup first checks the order ID.
    broker._client = lambda: client_for(values, [])
    guard = JournalReconcilerProfitProtection(broker, arm_ccy=1, giveback_ccy=0.5, journal=journal)
    if maintenance:
        result = run_cleanup(journal, broker, guard, "test-tx", demo_mode=True,
                             oanda_env="practice", open_positions_count=0)
        assert result["counts"] == {"RECOVERED_CLOSE": 1}
    else:
        assert guard._reconcile_untracked_journal_rows([], now_utc=datetime(2026, 7, 14, tzinfo=timezone.utc)) == ["101"]
    report = build_weekly_report(journal.path, now_utc=datetime(2026, 7, 14, tzinfo=timezone.utc))
    assert report.open_trades == 0 and report.total.trades == 1 and report.total.net_pnl == -0.75
    with journal._connect() as conn:
        saved = json.loads(conn.execute("SELECT evidence_json FROM journal_close_evidence WHERE trade_id='101'").fetchone()[0])
        assert saved["opening"] == opening and saved["closing"] == closing
    assert guard._reconcile_untracked_journal_rows([], now_utc=datetime(2026, 7, 14, tzinfo=timezone.utc)) == []


def test_exit_write_failure_rolls_back_the_evidence_too(tmp_path):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "101")
    with journal._connect() as conn:
        conn.execute("CREATE TRIGGER reject_exit BEFORE UPDATE ON trades BEGIN SELECT RAISE(ABORT, 'unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError):
        journal.record_exit(trade_id="101", exit_timestamp_utc=datetime.now(timezone.utc), exit_price=.659,
            spread_at_exit=None, max_profit_ccy=None, realized_pnl_ccy=-.75, exit_reason="BROKER_CLOSED",
            duration_seconds=60, broker_confirmed=True, broker_evidence={"source": "test"})
    with journal._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM journal_close_evidence").fetchone()[0] == 0
        assert conn.execute("SELECT exit_timestamp_utc FROM trades WHERE trade_id='101'").fetchone()[0] is None
