import copy
import hashlib
import json
import sqlite3
from decimal import Decimal

import httpx
import pytest

from src.account_reconciliation import EvidenceError
from src.practice_history import build_review, sync_history, start_history_monitor, read_context

ACCOUNT = "101-001-123-001"


def history():
    rows = []
    def tx(kind, **kwargs):
        n = len(rows) + 1
        rows.append(dict(id=str(n), type=kind, accountID=ACCOUNT,
                         time=f"2026-11-01T00:00:{n:02}Z", **kwargs))
    def fill(units, balance, **kwargs):
        fields = dict(instrument="AUD_USD", units=str(units), price="0.7000", fullVWAP="0.7000",
                      orderID="2", pl="0", financing="0", commission="0.20", guaranteedExecutionFee="0",
                      halfSpreadCost="99", reason="MARKET_ORDER", accountBalance=str(balance),
                      fullPrice={"timestamp": "2026-11-01T00:00:03Z", "bids": [{"price": "0.6999"}],
                                 "asks": [{"price": "0.7001"}]})
        fields.update(kwargs)
        tx("ORDER_FILL", **fields)
    tx("CREATE")
    tx("TRANSFER_FUNDS", amount="100", accountBalance="100")
    fill(10, "99.8", tradeOpened={"tradeID": "3", "units": "10", "price": "0.7000"})
    fill(30, "99.6", tradeOpened={"tradeID": "4", "units": "30", "price": "0.7000"})
    tx("STOP_LOSS_ORDER", tradeID="3", price="0.6900")
    tx("DAILY_FINANCING", financing="-0.4", accountBalance="99.2", positionFinancings=[{
        "instrument": "AUD_USD", "openTradeFinancings": [
            {"tradeID": "3", "financing": "-0.1"}, {"tradeID": "4", "financing": "-0.3"}]}])
    fill(-5, "99.5", pl="0.5", tradeReduced={"tradeID": "3", "units": "5", "realizedPL": "0.5"})
    fill(-35, "101.1", pl="2", financing="-0.2", tradesClosed=[
        {"tradeID": "3", "units": "5", "realizedPL": "1", "financing": "-0.1"},
        {"tradeID": "4", "units": "30", "realizedPL": "1", "financing": "-0.1"}])
    return rows


def review(rows=None, balance="101.1", journal_rows=()):
    rows = rows or history()
    return build_review(rows, account=ACCOUNT, cursor=len(rows), balance=balance, journal_rows=journal_rows)


def test_full_lifetime_costs_partials_and_allocation_conserve_money():
    report, trades, fills = review()
    assert report["ledger_status"] == "VERIFIED"
    assert Decimal(report["account_net_excluding_transfers_ccy"]) == Decimal("1.1")
    assert Decimal(report["closed"]["net_pl_ccy"]) == Decimal("1.1")
    assert Decimal(report["closed"]["gross_pl_ccy"]) == Decimal("2.5")
    assert report["closed"]["trades"] == 2  # partial close must not be an extra trade
    assert report["closed"]["allocated_commission_trades"] == 2
    assert trades[0]["partial_reductions"] == 1
    assert sum(t["commission"] for t in trades) == Decimal("0.8")
    assert sum(t["financing"] for t in trades) == Decimal("-0.6")
    assert trades[0]["protection_order_history"][0]["price"] == "0.6900"
    assert report["execution_by_instrument"]["AUD_USD"]["sample_count"] == 4
    assert fills[0]["execution_quality"]["half_spread_cost_ccy"] == 99  # NEVER debited again


def test_open_trade_charges_do_not_count_as_closed_loss():
    report, trades, _ = review(history()[:4], balance="99.6")
    assert report["closed"]["trades"] == 0
    assert report["open_trade_count"] == 2
    assert Decimal(report["open_trades_realized_charges_and_pl_ccy"]) == Decimal("-0.4")
    assert all(t["closed_net_pl"] is None for t in trades)


@pytest.mark.parametrize("mutation", [
    lambda t: t.pop(2),
    lambda t: t.append(copy.deepcopy(t[-1])),
    lambda t: t[2].update(accountID="different-account"),
    lambda t: t[-1].update(accountBalance="999"),
    lambda t: t[-1].update(financing="nan"),
    lambda t: t[-1]["tradesClosed"][0].update(financing="-1"),
    lambda t: t[5]["positionFinancings"][0]["openTradeFinancings"][0].update(financing="-1"),
    lambda t: t[-1].update(type="UNKNOWN_CHARGE"),
    lambda t: t[-1]["tradesClosed"][0].update(units="6"),
])
def test_bad_or_incomplete_evidence_cannot_produce_verified_report(mutation):
    rows = history()
    mutation(rows)
    with pytest.raises(EvidenceError):
        review(rows)


def test_journal_context_requires_exact_trade_identity_and_time():
    report, trades, _ = review(journal_rows=[dict(trade_id="3", instrument="AUD_USD",
        timestamp_utc="2026-11-01T00:00:03+00:00", run_tag="MINI_RUN",
        gating_flags={"runtime_revision": "known-revision"})])
    assert report["journal_context_matches"] == 1
    assert report["by_run_tag"]["MINI_RUN"]["trades"] == 1
    assert Decimal(report["by_run_tag"]["MINI_RUN"]["net_pl_ccy"]) == trades[0]["closed_net_pl"]
    assert trades[0]["journal_context"]["gating_flags"]["runtime_revision"] == "known-revision"
    report, trades, _ = review(journal_rows=[dict(trade_id="3", instrument="AUD_USD",
        timestamp_utc="2026-11-02T00:00:03+00:00", run_tag="MINI_RUN")])
    assert report["journal_context_matches"] == 0
    assert trades[0]["journal_context"] is None


def test_open_entry_context_does_not_require_exit_confirmation(tmp_path):
    path = tmp_path / "journal.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE trades (trade_id TEXT, instrument TEXT, timestamp_utc TEXT, "
                     "broker_confirmed INTEGER, gating_flags TEXT, indicators_snapshot TEXT)")
        conn.execute("CREATE TABLE journal_entry_resolutions (trade_id TEXT)")
        conn.executemany("INSERT INTO trades VALUES (?, 'AUD_USD', ?, NULL, ?, '{}')", [
            ("3", "2026-11-01T00:00:03Z", json.dumps({"runtime_revision": "entry-revision", "decision_reason": "trend"})),
            ("4", "2026-11-01T00:00:04Z", "{}")])
        conn.execute("INSERT INTO journal_entry_resolutions VALUES ('4')")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    context = read_context(path)
    assert [r["trade_id"] for r in context] == ["3"]
    report, trades, _ = review(history()[:4], balance="99.6", journal_rows=context)
    assert report["journal_context_matches"] == 1
    assert trades[0]["closed_net_pl"] is None
    assert trades[0]["journal_context"]["gating_flags"]["decision_reason"] == "trend"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("OANDA_ENV", "practice")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("OANDA_API_KEY", "unit-test-only")


def transport_for(rows, requests, bad_page=False, balance=None):
    def respond(request):
        requests.append((request.method, request.url.path, dict(request.url.params)))
        assert request.method == "GET"
        assert request.url.host == "api-fxpractice.oanda.com"
        if request.url.path.endswith("/summary"):
            return httpx.Response(200, json={"account": {"id": ACCOUNT, "currency": "AUD",
                "lastTransactionID": str(len(rows)), "balance": balance or rows[-1]["accountBalance"],
                "openTradeCount": 0 if len(rows) == 8 else 2}})
        assert request.url.path.endswith("/transactions/idrange")
        lower, upper = int(request.url.params["from"]), int(request.url.params["to"])
        batch = rows[lower - 1:upper]
        return httpx.Response(200, json={"transactions": batch[:-1] if bad_page else batch})
    return httpx.MockTransport(respond)


def test_incremental_restart_idempotency_and_no_journal_write(tmp_path, env):
    requests = []
    absent_journal = tmp_path / "journal-must-not-be-created.sqlite"
    report = sync_history(tmp_path, absent_journal, transport=transport_for(history()[:4], requests))
    assert report["transaction_count"] == 4
    report = sync_history(tmp_path, absent_journal, transport=transport_for(history(), requests))
    assert report["new_transactions"] == 4
    assert requests[-1][2] == {"from": "5", "to": "8"}
    previous_requests = len(requests)
    again = sync_history(tmp_path, absent_journal, transport=transport_for(history(), requests))
    assert again["new_transactions"] == 0
    assert len(requests) == previous_requests + 1  # summary only, no full re-download
    assert again["closed"] == report["closed"]
    assert not absent_journal.exists()
    folder = tmp_path / "memory/oanda-practice" / hashlib.sha256(ACCOUNT.encode()).hexdigest()[:16] / "enriched"
    with sqlite3.connect(folder / "history.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM transactions").fetchone()[0] == 8
        assert conn.execute("SELECT count(*) FROM trade_review").fetchone()[0] == 2
    assert (folder / "history.sqlite").stat().st_mode & 0o777 == 0o600
    assert (folder / "review.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["missing-page", "balance-mismatch"])
def test_failed_sync_preserves_previous_cursor_and_report(tmp_path, env, failure):
    requests = []
    journal = tmp_path / "journal"
    sync_history(tmp_path, journal, transport=transport_for(history()[:4], requests))
    with pytest.raises(EvidenceError):
        sync_history(tmp_path, journal, transport=transport_for(history(), requests,
                     bad_page=failure == "missing-page", balance="999" if failure == "balance-mismatch" else None))
    folder = tmp_path / "memory/oanda-practice" / hashlib.sha256(ACCOUNT.encode()).hexdigest()[:16] / "enriched"
    with sqlite3.connect(folder / "history.sqlite") as conn:
        assert conn.execute("SELECT value FROM state WHERE key='last_transaction_id'").fetchone()[0] == "4"
        assert conn.execute("SELECT count(*) FROM transactions").fetchone()[0] == 4
    assert json.loads((folder / "review.json").read_text())["last_transaction_id"] == "4"


def test_existing_archive_is_read_only_and_reused(tmp_path, env):
    folder = tmp_path / "memory/oanda-practice" / hashlib.sha256(ACCOUNT.encode()).hexdigest()[:16]
    seed = folder / "history-8/trade_history.sqlite"
    seed.parent.mkdir(parents=True)
    with sqlite3.connect(seed) as conn:
        conn.execute("CREATE TABLE broker_transactions(raw_json TEXT)")
        conn.executemany("INSERT INTO broker_transactions VALUES (?)", [(json.dumps(t),) for t in history()])
    digest = hashlib.sha256(seed.read_bytes()).hexdigest()
    requests = []
    report = sync_history(tmp_path, tmp_path / "journal", transport=transport_for(history(), requests))
    assert report["new_transactions"] == 0
    assert len(requests) == 1
    assert hashlib.sha256(seed.read_bytes()).hexdigest() == digest


def test_live_mode_and_opt_out_never_start_network(tmp_path, monkeypatch, env):
    monkeypatch.setenv("MODE", "live")
    with pytest.raises(EvidenceError, match="practice-demo-mode-required"):
        sync_history(tmp_path, tmp_path / "journal")
    assert start_history_monitor(tmp_path, tmp_path / "journal") is None
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("PRACTICE_HISTORY_ENABLED", "false")
    assert start_history_monitor(tmp_path, tmp_path / "journal") is None
