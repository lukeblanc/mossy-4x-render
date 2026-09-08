from datetime import datetime, timezone
import sqlite3

import pytest

from src.journal_reconciler import JournalReconcilerProfitProtection
from src.learning_profit_protection import LearningProfitProtection
from src.trade_journal import TradeJournal
from test_journal_reconciler import ClosedTradeBroker, _record_entry, _read_exit


def closed_details(trade_id="42", **overrides):
    return {"id": trade_id, "instrument": "AUD_USD", "state": "CLOSED", "currentUnits": "0",
            "realizedPL": "-1.25", "averageClosePrice": "0.659",
            "closeTime": "2026-07-13T01:00:20Z", **overrides}


def guard_for(journal, details=None):
    return JournalReconcilerProfitProtection(ClosedTradeBroker(details or {}),
        arm_ccy=1, giveback_ccy=0.5, journal=journal)


@pytest.mark.parametrize("overrides", [
    {"realizedPL": None}, {"realizedPL": "NaN"}, {"realizedPL": "Infinity"},
    {"averageClosePrice": None}, {"averageClosePrice": "0"}, {"averageClosePrice": "NaN"},
    {"closeTime": None}, {"closeTime": "bad"}, {"state": "OPEN"},
    {"currentUnits": "10"}, {"currentUnits": "NaN"},
])
def test_missing_or_invalid_close_evidence_cannot_become_a_confirmed_result(tmp_path, overrides):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "42")
    guard = guard_for(journal, {"42": closed_details(**overrides)})
    assert guard.process_open_trades([]) == []
    assert _read_exit(journal, "42")[0] is None
    assert journal.count_trade_events() == 1


@pytest.mark.parametrize("overrides", [
    {"id": "999"}, {"instrument": "GBP_USD"}, {"closeTime": "2026-07-12T23:00:00Z"},
])
def test_wrong_identity_or_impossible_chronology_stays_unresolved(tmp_path, overrides):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "42")
    guard = guard_for(journal, {"42": closed_details(**overrides)})
    assert guard.process_open_trades([]) == []
    assert _read_exit(journal, "42")[0] is None


def test_close_only_or_reduce_only_orders_cannot_identify_a_new_trade():
    extract = JournalReconcilerProfitProtection._extract_trade_id
    assert extract({"tradesClosed": [{"tradeID": "42"}]}) is None
    assert extract({"tradeReduced": {"tradeID": "42"}}) is None
    assert extract({"tradeID": "42"}) is None
    assert extract({"tradeOpenedID": "42"}) == "42"
    assert extract({"tradeOpened": {"tradeID": "42"}}) == "42"


def test_no_same_instrument_fallback_or_orphan_exit(tmp_path):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "101")
    _record_entry(journal, "201")
    guard = LearningProfitProtection(ClosedTradeBroker({}), arm_ccy=1, giveback_ccy=0.5, journal=journal)
    assert guard._resolve_journal_trade_id("42", "AUD_USD") is None
    guard._close_fills["42"] = {"pnl": 1.25, "exit_price": 0.661,
        "closed_at": datetime(2026, 7, 13, 1, 5, tzinfo=timezone.utc)}
    guard._reconcile_closed("42", "AUD_USD", [], None, reason="BROKER_CLOSED",
                            closed_by="broker_confirmed", final_profit=1.25)
    assert _read_exit(journal, "101")[0] is None
    assert _read_exit(journal, "201")[0] is None
    assert _read_exit(journal, "42") is None
    assert journal.count_trade_events() == 2


def test_reconciliation_work_is_capped_and_failed_ids_back_off(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path / "journal.db")
    for i in range(10):
        _record_entry(journal, str(1000 + i))
    guard = guard_for(journal)
    clock = [1000.0]
    monkeypatch.setattr("src.journal_reconciler.time.monotonic", lambda: clock[0])
    monkeypatch.setenv("JOURNAL_RECONCILE_ATTEMPTS_PER_CYCLE", "100")
    calls = []
    def unavailable(trade_id):
        calls.append(trade_id)
        return trade_id, None
    monkeypatch.setattr(guard, "_details_for_journal_id", unavailable)
    guard.process_open_trades([])
    assert len(calls) == 5
    guard.process_open_trades([])
    assert len(calls) == 10 and len(set(calls)) == 10
    guard.process_open_trades([])
    assert len(calls) == 10
    clock[0] += 901
    guard.process_open_trades([])
    assert len(calls) == 15


def test_rows_beyond_old_500_row_limit_can_be_recovered(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path / "journal.db")
    with sqlite3.connect(journal.path) as conn:
        conn.executemany("INSERT INTO trades (trade_id,timestamp_utc,instrument,side,entry_price) VALUES (?, ?, ?, ?, ?)",
            [(str(1000 + i), "2026-07-13T01:00:00+00:00", "AUD_USD", "BUY", 0.66) for i in range(501)])
    guard = guard_for(journal, {"1500": closed_details("1500")})
    guard._reconcile_cursor = 500
    monkeypatch.setenv("JOURNAL_RECONCILE_ATTEMPTS_PER_CYCLE", "1")
    assert guard.process_open_trades([]) == ["1500"]
    assert _read_exit(journal, "1500")[2] == -1.25


def test_non_broker_local_ids_do_not_trigger_network_reads(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "local-1234")
    guard = guard_for(journal)
    monkeypatch.setattr(guard, "_broker_trade_details", lambda *a: pytest.fail("local ID queried at broker"))
    assert guard.process_open_trades([]) == []
    assert _read_exit(journal, "local-1234")[0] is None


def test_legacy_alias_cannot_count_the_same_broker_trade_twice(tmp_path):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "101")
    _record_entry(journal, "102")
    guard = guard_for(journal, {"102": closed_details("102")})
    assert guard.process_open_trades([]) == ["102"]
    assert _read_exit(journal, "101")[0] is None
    assert _read_exit(journal, "102")[2] == -1.25
    with sqlite3.connect(journal.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trades WHERE exit_timestamp_utc IS NOT NULL").fetchone()[0] == 1
        assert conn.execute("SELECT spread_at_exit FROM trades WHERE trade_id='102'").fetchone()[0] is None
        assert conn.execute("SELECT equity_after FROM trade_events WHERE reason='BROKER_CLOSED'").fetchone()[0] is None
