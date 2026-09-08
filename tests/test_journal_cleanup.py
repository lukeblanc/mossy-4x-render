from copy import deepcopy
from datetime import datetime, timezone
import json
import sqlite3
from types import SimpleNamespace

import pytest

from src.journal_cleanup import run_cleanup
from src.journal_reconciler import JournalReconcilerProfitProtection
from src.trade_journal import TradeJournal
from src.weekly_ops_report import build_weekly_report
from test_journal_reconciler import _record_entry, _read_exit


class ReadOnlyBroker:
    mode = "demo"
    account = "test-account"

    def __init__(self, resources):
        self.resources = resources
        self.calls = []
        self.connections = 0
        self.statuses = {}

    def _client(self):
        self.connections += 1
        broker = self
        class Client:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def get(self, path):
                key = path.split("/", 4)[-1]
                broker.calls.append(key)
                value = broker.resources.get(key)
                status = broker.statuses.get(key, 200 if value is not None else 404)
                return SimpleNamespace(status_code=status, json=lambda: deepcopy(value))
        return Client()


def cancellation(ticket="101"):
    cancel_id = str(int(ticket) + 1)
    return {
        f"orders/{ticket}": {"order": {"id": ticket, "instrument": "AUD_USD", "type": "MARKET",
            "state": "CANCELLED", "createTime": "2026-07-13T01:00:01Z", "cancellingTransactionID": cancel_id}},
        f"transactions/{cancel_id}": {"transaction": {"id": cancel_id, "type": "ORDER_CANCEL",
            "orderID": ticket, "time": "2026-07-13T01:00:02Z", "reason": "TAKE_PROFIT_ON_FILL_LOSS"}},
    }


def setup(tmp_path, resources=None):
    journal = TradeJournal(tmp_path / "journal.db")
    _record_entry(journal, "101")
    broker = ReadOnlyBroker(resources if resources is not None else cancellation())
    guard = JournalReconcilerProfitProtection(broker, arm_ccy=1, giveback_ccy=0.5, journal=journal)
    return journal, broker, guard


def run(journal, broker, guard, **kwargs):
    values = dict(demo_mode=True, oanda_env="practice", open_positions_count=0)
    values.update(kwargs)
    return run_cleanup(journal, broker, guard, "approved-test", **values)


def rows(path, table):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def test_cancelled_order_is_audited_without_deleting_history_or_creating_pnl(tmp_path):
    journal, broker, guard = setup(tmp_path)
    original_trades, original_events = rows(journal.path, "trades"), rows(journal.path, "trade_events")
    result = run(journal, broker, guard)
    assert result["status"] == "completed"
    assert result["counts"] == {"CANCELLED_ORDER": 1}
    assert rows(journal.path, "trades") == original_trades
    assert rows(journal.path, "trade_events") == original_events
    assert rows(result["backup"], "trades") == original_trades
    assert rows(result["backup"], "journal_entry_resolutions") == []
    from pathlib import Path
    standalone = tmp_path / "standalone-backup.sqlite"
    standalone.write_bytes(Path(result["backup"]).read_bytes())
    assert rows(standalone, "trades") == original_trades
    with sqlite3.connect(standalone) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    audit = rows(journal.path, "journal_entry_resolutions")[0]
    assert json.loads(audit[6])["cancellation"]["orderID"] == "101"
    assert json.loads(audit[5])["trade_id"] == "101"
    assert broker.connections == 1
    assert guard._unclosed_journal_rows() == []
    report = build_weekly_report(journal.path, now_utc=datetime(2026, 7, 14, tzinfo=timezone.utc))
    assert report.open_trades == 0 and report.cancelled_order_rows == 1
    assert report.total.trades == 0 and report.total.net_pnl == 0
    before_calls = list(broker.calls)
    assert run(journal, broker, guard)["status"] == "already-completed"
    assert broker.calls == before_calls


@pytest.mark.parametrize("target,field,value", [
    ("order", "instrument", "GBP_USD"), ("order", "id", "999"),
    ("order", "createTime", "2026-07-12T00:00:00Z"), ("order", "createTime", None),
    ("order", "state", "PENDING"), ("order", "type", "STOP_LOSS"),
    ("order", "fillingTransactionID", "100"), ("order", "tradeOpenedID", "100"),
    ("order", "cancellingTransactionID", None), ("transaction", "orderID", "999"),
    ("transaction", "type", "ORDER_FILL"), ("transaction", "time", None),
])
def test_incomplete_or_conflicting_evidence_stays_unresolved(tmp_path, target, field, value):
    resources = cancellation()
    key = "orders/101" if target == "order" else "transactions/102"
    resources[key][target][field] = value
    journal, broker, guard = setup(tmp_path, resources)
    result = run(journal, broker, guard)
    assert result["status"] == "completed"
    assert result["unresolved_ids"] == ["101"]
    assert rows(journal.path, "journal_entry_resolutions") == []
    assert _read_exit(journal, "101")[0] is None
    assert len(guard._unclosed_journal_rows()) == 1


def test_interrupted_cleanup_resumes_frozen_snapshot_and_is_idempotent(tmp_path):
    journal, broker, guard = setup(tmp_path)
    broker.statuses["orders/101"] = 503
    assert run(journal, broker, guard)["status"] == "incomplete"
    _record_entry(journal, "201")  # New rows are outside this approved snapshot.
    broker.resources.update(cancellation("201"))
    broker.statuses.clear()
    assert run(journal, broker, guard)["counts"] == {"CANCELLED_ORDER": 1}
    assert "orders/201" not in broker.calls
    assert len(guard._unclosed_journal_rows()) == 1
    assert len(rows(journal.path, "journal_entry_resolutions")) == 1


def test_backup_failure_prevents_cleanup(tmp_path, monkeypatch):
    journal, broker, guard = setup(tmp_path)
    def unavailable(*args):
        raise OSError("backup unavailable")
    monkeypatch.setattr("src.journal_cleanup._backup", unavailable)
    with pytest.raises(OSError):
        run(journal, broker, guard)
    assert not broker.calls
    assert rows(journal.path, "journal_cleanup_runs") == []
    assert rows(journal.path, "journal_entry_resolutions") == []


def test_audit_write_failure_does_not_hide_entry_or_consume_outcome(tmp_path):
    journal, broker, guard = setup(tmp_path)
    with journal._connect() as conn:
        conn.execute("CREATE TRIGGER fail_audit BEFORE INSERT ON journal_entry_resolutions "
                     "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    assert run(journal, broker, guard)["status"] == "incomplete"
    assert len(guard._unclosed_journal_rows()) == 1
    assert rows(journal.path, "journal_entry_resolutions") == []
    with journal._connect() as conn:
        conn.execute("DROP TRIGGER fail_audit")
    assert run(journal, broker, guard)["counts"] == {"CANCELLED_ORDER": 1}


@pytest.mark.parametrize("overrides", [
    {"demo_mode": False}, {"oanda_env": "live"}, {"open_positions_count": 1},
    {"open_positions_count": None}, {"open_positions_count": False},
])
def test_cleanup_requires_practice_and_verified_flat_account(tmp_path, overrides):
    journal, broker, guard = setup(tmp_path)
    assert run(journal, broker, guard, **overrides)["status"] != "completed"
    assert not broker.calls and not broker.connections
    assert rows(journal.path, "journal_cleanup_runs") == []


@pytest.mark.parametrize("with_canonical", [False, True])
def test_real_closes_and_duplicate_aliases_are_resolved_once(tmp_path, with_canonical):
    resources = {
        "orders/101": {"order": {"id": "101", "instrument": "AUD_USD", "state": "FILLED",
            "createTime": "2026-07-13T01:00:01Z", "tradeOpenedID": "102"}},
        "trades/102": {"trade": {"id": "102", "instrument": "AUD_USD", "state": "CLOSED",
            "currentUnits": "0", "realizedPL": "-1.25", "averageClosePrice": "0.659",
            "closeTime": "2026-07-13T01:00:20Z"}},
    }
    journal, broker, guard = setup(tmp_path, resources)
    if with_canonical:
        _record_entry(journal, "102")
    result = run(journal, broker, guard)
    assert result["status"] == "completed" and result["unresolved_ids"] == []
    report = build_weekly_report(journal.path, now_utc=datetime(2026, 7, 14, tzinfo=timezone.utc))
    assert report.open_trades == 0
    assert report.total.trades == 1 and report.total.net_pnl == -1.25
    assert report.duplicate_alias_rows == int(with_canonical)


def test_unconfirmed_results_do_not_inflate_verified_report(tmp_path):
    journal, _, _ = setup(tmp_path)
    with journal._connect() as conn:
        conn.execute("UPDATE trades SET exit_timestamp_utc='2026-07-13T02:00:00+00:00', "
                     "realized_pnl_ccy=9999, broker_confirmed=0 WHERE trade_id='101'")
    report = build_weekly_report(journal.path, now_utc=datetime(2026, 7, 14, tzinfo=timezone.utc))
    assert report.total.trades == 0 and report.total.net_pnl == 0
    assert report.unconfirmed_closed_rows == 1
    assert any("not broker-confirmed" in alert for alert in report.alerts)
