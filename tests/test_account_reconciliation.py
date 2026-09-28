from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from src import account_reconciliation as ar
from src import weekly_ops_report as weekly

START = datetime(2026, 9, 20, 11, 15, tzinfo=timezone.utc)
END = START + timedelta(days=7)
ACCOUNT = "101-001-123-001"


def tx(tid, kind, *, when=None, **fields):
    return {"id": str(tid), "accountID": ACCOUNT, "time": (when or START + timedelta(hours=tid)).isoformat(),
            "type": kind, **fields}


def evidence():
    rows = [
        tx(1, "TRANSFER_FUNDS", when=START - timedelta(days=1), amount="100", accountBalance="100"),
        tx(2, "ORDER_FILL", pl="2.0", financing="-0.10", commission="0.20", guaranteedExecutionFee="0.05",
           accountBalance="101.65", instrument="AUD_USD", halfSpreadCost="0.30",
           tradesClosed=[{"tradeID": "100", "realizedPL": "2.0", "financing": "-0.10"}]),
        tx(3, "DAILY_FINANCING", financing="-0.40", accountBalance="101.25",
           positionFinancings=[{"financing": "-0.40"}]),
        tx(4, "TRANSFER_FUNDS", amount="10", accountBalance="111.25"),
        tx(5, "ORDER_FILL", pl="-1.00", financing="0", accountBalance="110.25", instrument="GBP_USD",
           tradesClosed=[{"tradeID": "101", "realizedPL": "-1.00"}]),
        tx(6, "DIVIDEND_ADJUSTMENT", dividendAdjustment="0.25", accountBalance="110.50"),
        tx(7, "RESET_RESETTABLE_PL"),
    ]
    journal = [{"trade_id": key, "instrument": inst, "realized_pnl_ccy": pnl,
                "exit_timestamp_utc": rows[idx]["time"], "broker_confirmed": 1}
               for key, inst, pnl, idx in [("100", "AUD_USD", "2.00", 1), ("101", "GBP_USD", "-1.00", 4)]]
    return rows, journal


def run(rows=None, journal=None, **kwargs):
    base_rows, base_journal = evidence()
    rows = base_rows if rows is None else rows
    return ar.reconcile(transactions=rows, expected_count=len(rows), account_id=ACCOUNT, currency="AUD",
                        start=START, end=END, journal_rows=base_journal if journal is None else journal, **kwargs)


def test_exact_balance_bridge_separates_costs_and_transfers():
    report = run()
    assert report["status"] == "VERIFIED"
    assert report["components"] == {"realized_pl": "1.00", "financing": "-0.50", "commission_cost": "0.20",
                                    "guaranteed_fee_cost": "0.05", "dividend_adjustment": "0.25", "transfers": "10"}
    assert Decimal(report["account_net_excluding_transfers"]) == Decimal("0.50")
    assert Decimal(report["unexplained_balance_delta"]) == 0
    assert Decimal(report["journal_vs_broker_pl_delta"]) == 0
    assert ACCOUNT not in json.dumps(report)


def test_spread_and_nested_financing_are_not_deducted_twice():
    rows, journal = evidence()
    rows[1]["halfSpreadCost"] = "99999"
    rows[1]["tradesClosed"][0]["financing"] = "-99999"
    assert run(rows, journal)["status"] == "VERIFIED"


def test_positive_financing_and_negative_transfer_signs():
    rows = [tx(1, "TRANSFER_FUNDS", when=START, amount="100", accountBalance="100"),
            tx(2, "DAILY_FINANCING", financing="0.25", accountBalance="100.25"),
            tx(3, "TRANSFER_FUNDS", amount="-10", accountBalance="90.25")]
    report = run(rows, [])
    assert report["status"] == "VERIFIED"
    assert Decimal(report["account_net_excluding_transfers"]) == Decimal("0.25")


def test_unknown_adjustment_stays_unverified_instead_of_guessing():
    rows, journal = evidence()
    rows.append(tx(8, "UNRECOGNISED_FEE", accountBalance="101.48"))
    report = run(rows, journal)
    assert report["status"] == "UNVERIFIED"
    assert "unsupported-transaction-type" in report["reasons"]
    assert Decimal(report["unexplained_balance_delta"]) == Decimal("-9.02")
    assert report["account_net_excluding_transfers"] is None


def test_offsetting_intermediate_errors_do_not_pass_zero_final_delta():
    rows, journal = evidence()
    rows[1]["accountBalance"] = "102.65"
    report = run(rows, journal)
    assert Decimal(report["unexplained_balance_delta"]) == 0
    assert report["ledger_status"] == "UNVERIFIED"


@pytest.mark.parametrize("field,value", [("pl", "NaN"), ("pl", "Infinity"), ("pl", None),
                                         ("commission", "-1"), ("financing", "oops"), ("pl", True)])
def test_malformed_money_never_verifies(field, value):
    rows, journal = evidence()
    rows[1][field] = value
    assert run(rows, journal)["status"] == "UNVERIFIED"


def test_duplicate_transactions_never_double_count_or_verify():
    rows, journal = evidence()
    rows.append(copy.deepcopy(rows[1]))
    assert run(rows, journal)["status"] == "UNVERIFIED"


def test_missing_opening_balance_is_not_zero():
    rows, journal = evidence()
    report = run(rows[1:], journal)
    assert report["status"] == "UNVERIFIED"
    assert report["unexplained_balance_delta"] is None


def test_mixed_account_rejected():
    rows, journal = evidence()
    rows[3]["accountID"] = "different"
    assert run(rows, journal)["status"] == "UNVERIFIED"


def test_missing_journal_trade_detected_even_when_balance_reconciles():
    report = run(journal=evidence()[1][:1])
    assert report["ledger_status"] == "VERIFIED"
    assert report["journal_status"] == "UNVERIFIED"
    assert report["missing_journal_close_count"] == 1


def test_wrong_journal_ids_cannot_pass_by_matching_total_pnl():
    rows, journal = evidence()
    journal[0]["trade_id"] = "999"
    report = run(rows, journal)
    assert Decimal(report["journal_vs_broker_pl_delta"]) == 0
    assert report["status"] == "UNVERIFIED"


@pytest.mark.parametrize("field,value", [("instrument", "EUR_USD"), ("broker_confirmed", 0),
                                         ("realized_pnl_ccy", "1.90"),
                                         ("exit_timestamp_utc", (START + timedelta(hours=4)).isoformat())])
def test_individual_journal_close_details_must_match(field, value):
    rows, journal = evidence()
    journal[0][field] = value
    assert run(rows, journal)["status"] == "UNVERIFIED"


def test_partial_closes_require_explicit_allocation():
    rows, journal = evidence()
    rows[1]["tradeReduced"] = {"tradeID": "999", "realizedPL": "0"}
    assert "partial-close-needs-lifetime-allocation" in run(rows, journal)["reasons"]


def test_start_exclusive_end_inclusive_and_non_utc_timestamps():
    rows = [tx(1, "TRANSFER_FUNDS", when=START, amount="100", accountBalance="100"),
            tx(2, "DAILY_FINANCING", when=END, financing="-1", accountBalance="99")]
    rows[1]["time"] = END.astimezone(timezone(timedelta(hours=8))).isoformat()
    assert run(rows, [])["status"] == "VERIFIED"


def test_one_nanosecond_past_end_is_not_silently_included():
    rows = [tx(1, "TRANSFER_FUNDS", when=START, amount="100", accountBalance="100"),
            tx(2, "DAILY_FINANCING", when=END, financing="-1", accountBalance="99")]
    rows[1]["time"] = END.strftime("%Y-%m-%dT%H:%M:%S") + ".000000001Z"
    assert run(rows, [])["status"] == "UNVERIFIED"


def test_current_equity_is_not_used_as_historical_balance():
    report = run(checkpoint={"last_transaction_id": "999", "balance": "1", "nav": "-100"})
    assert report["status"] == "VERIFIED"
    assert report["closing_checkpoint"]["balance"] == "110.50"


def test_identical_cursor_summary_balance_must_match():
    assert run(checkpoint={"last_transaction_id": "7", "balance": "1"})["status"] == "UNVERIFIED"


def make_db(path):
    rows, journal = evidence()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE trades(trade_id TEXT, instrument TEXT, exit_timestamp_utc TEXT, realized_pnl_ccy REAL, broker_confirmed INTEGER)")
        conn.executemany("INSERT INTO trades VALUES(:trade_id,:instrument,:exit_timestamp_utc,:realized_pnl_ccy,:broker_confirmed)", journal)
    return rows


def test_journal_read_does_not_change_bytes_or_create_missing_db(tmp_path):
    db = tmp_path / "journal.db"
    make_db(db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert len(ar.read_journal(db, START, END)) == 2
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    missing = tmp_path / "missing.db"
    with pytest.raises(ar.EvidenceError):
        ar.read_journal(missing, START, END)
    assert not missing.exists()


@pytest.fixture
def demo_env(monkeypatch):
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("OANDA_ENV", "practice")
    monkeypatch.setenv("OANDA_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", ACCOUNT)


def mock_transport(rows, *, bad_page=None, wrong_count=False, status=200):
    calls = []
    def serve(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.url.host == "api-fxpractice.oanda.com"
        if status != 200:
            return httpx.Response(status, json={"errorMessage": "sensitive-info-do-not-log"})
        if request.url.path.endswith("/summary"):
            return httpx.Response(200, json={"lastTransactionID": "7", "account": {
                "id": ACCOUNT, "currency": "AUD", "balance": "110.50", "NAV": "113.50",
                "unrealizedPL": "3", "openTradeCount": 1}})
        if request.url.path.endswith("/transactions"):
            pages = [f"{ar.HOST}/v3/accounts/{ACCOUNT}/transactions/idrange?from=1&to=3",
                     f"{ar.HOST}/v3/accounts/{ACCOUNT}/transactions/idrange?from=4&to=7"]
            return httpx.Response(200, json={"count": len(rows) + int(wrong_count), "pages": [bad_page] if bad_page else pages})
        lower, upper = int(request.url.params["from"]), int(request.url.params["to"])
        return httpx.Response(200, json={"transactions": [r for r in rows if lower <= int(r["id"]) <= upper]})
    return httpx.MockTransport(serve), calls


def test_collection_get_only_follows_all_pages_and_uses_account_currency(tmp_path, demo_env):
    db = tmp_path / "journal.db"
    rows = make_db(db)
    transport, calls = mock_transport(rows)
    report = ar.collect_reconciliation(db, START.isoformat(), END.isoformat(), transport=transport)
    assert report["status"] == "VERIFIED"
    assert len(calls) == 4
    assert all(c.method == "GET" for c in calls)
    assert report["current_checkpoint"]["nav"] == "113.50"
    assert report["closing_checkpoint"]["balance"] == "110.50"


@pytest.mark.parametrize("url", ["https://evil.invalid/steal", f"https://api-fxtrade.oanda.com/v3/accounts/{ACCOUNT}/transactions/idrange?from=1&to=7",
    f"{ar.HOST}/v3/accounts/999/transactions/idrange?from=1&to=7",
    f"{ar.HOST}/v3/accounts/{ACCOUNT}/transactions/idrange?from=1&to=7&type=ORDER_FILL",
    f"{ar.HOST}/v3/accounts/{ACCOUNT}/transactions/idrange?from=1&from=2&to=7"])
def test_credentials_never_follow_untrusted_or_filtered_page_urls(tmp_path, demo_env, url):
    db = tmp_path / "journal.db"
    transport, calls = mock_transport(make_db(db), bad_page=url)
    report = ar.collect_reconciliation(db, START.isoformat(), END.isoformat(), transport=transport)
    assert report["status"] == "UNVERIFIED"
    assert len(calls) == 2


def test_incomplete_pagination_is_not_a_zero_pnl_report(tmp_path, demo_env):
    db = tmp_path / "journal.db"
    transport, _ = mock_transport(make_db(db), wrong_count=True)
    report = ar.collect_reconciliation(db, START.isoformat(), END.isoformat(), transport=transport)
    assert "incomplete-transaction-history" in report["reasons"]


@pytest.mark.parametrize("mode,env", [("live", "practice"), ("demo", "live"), ("simulation", "practice"), ("", "")])
def test_not_demo_practice_makes_no_request(monkeypatch, demo_env, mode, env, tmp_path):
    monkeypatch.setenv("MODE", mode)
    monkeypatch.setenv("OANDA_ENV", env)
    transport, calls = mock_transport([])
    report = ar.collect_reconciliation(tmp_path/"none.db", START.isoformat(), END.isoformat(), transport=transport)
    assert report["status"] == "UNVERIFIED" and not calls


@pytest.mark.parametrize("status", [401, 429, 500, 302])
def test_http_failures_are_redacted(tmp_path, demo_env, status):
    transport, _ = mock_transport([], status=status)
    report = ar.collect_reconciliation(tmp_path/"none.db", START.isoformat(), END.isoformat(), transport=transport)
    assert report["status"] == "UNVERIFIED"
    assert "sensitive-info" not in json.dumps(report)
    assert ACCOUNT not in json.dumps(report)


def test_report_attaches_account_verification_without_altering_trade_results(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    make_db(db)
    old = weekly.build_weekly_report(db, now_utc=END)
    monkeypatch.setattr(ar, "collect_reconciliation", lambda *a: run())
    new = weekly.attach_account_reconciliation(old)
    assert new.total == old.total and new.status == old.status and new.alerts == old.alerts
    assert new.account_reconciliation["status"] == "VERIFIED"
    markdown = weekly.render_markdown(new)
    assert "not total account return" in markdown
    assert "Performance verification: VERIFIED" in markdown
    assert "Financing (signed): -0.50" in markdown
    assert "accountID" not in markdown
    paths = weekly.save_report(new, tmp_path / "reports")
    assert json.loads(paths[1].read_text())["account_reconciliation"]["status"] == "VERIFIED"


def test_failed_audit_still_generates_a_truthful_report(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    make_db(db)
    old = weekly.build_weekly_report(db, now_utc=END)
    def fail(*a):
        raise RuntimeError("do-not-leak")
    monkeypatch.setattr(ar, "collect_reconciliation", fail)
    new = weekly.attach_account_reconciliation(old)
    assert new.total == old.total
    assert "Performance verification: UNVERIFIED" in weekly.render_markdown(new)
    assert "do-not-leak" not in weekly.render_markdown(new)


def test_weekly_report_with_no_account_read_is_explicitly_unverified(tmp_path):
    db = tmp_path / "journal.db"
    make_db(db)
    report = weekly.build_weekly_report(db, now_utc=END)
    assert "Performance verification: UNVERIFIED" in weekly.render_markdown(report)


def test_weekly_window_normalisation_excludes_start_and_includes_end(tmp_path):
    db = tmp_path / "journal.db"
    make_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM trades")
        conn.executemany("INSERT INTO trades VALUES(?, ?, ?, ?, 1)", [
            ("1", "AUD_USD", START.isoformat(), 100),
            ("2", "AUD_USD", END.astimezone(timezone(timedelta(hours=8))).isoformat(), 2),
        ])
    report = weekly.build_weekly_report(db, now_utc=END)
    assert report.total.trades == 1 and report.total.net_pnl == 2


def test_summary_cursor_before_period_cannot_verify():
    report = run(checkpoint={"last_transaction_id": "1", "balance": "100"})
    assert "summary-cursor-behind-report-period" in report["reasons"]


def test_budget_exhaustion_makes_no_request():
    transport, calls = mock_transport([])
    with httpx.Client(transport=transport) as client:
        reader = ar.PracticeReader(client, ACCOUNT)
        reader.deadline = -1
        with pytest.raises(ar.EvidenceError, match="time-budget"):
            reader.get(reader.root + "/summary")
    assert not calls


def test_too_many_pages_fails_before_download():
    calls = []
    def serve(request):
        calls.append(request)
        return httpx.Response(200, json={"count": 20001, "pages": []})
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        with pytest.raises(ar.EvidenceError, match="budget-exceeded"):
            ar.PracticeReader(client, ACCOUNT).history(END)
    assert len(calls) == 1


def test_account_evidence_is_private_versioned_and_does_not_overwrite_latest(tmp_path):
    latest = tmp_path / "LATEST_ALGO_REPORT.md"
    latest.write_text("leave weekly report intact")
    audit = run()
    path = weekly.save_account_evidence(audit, tmp_path, END.isoformat())
    changed = dict(audit, status="UNVERIFIED")
    second = weekly.save_account_evidence(changed, tmp_path, END.isoformat())
    assert path != second
    assert json.loads(path.read_text())["status"] == "VERIFIED"
    assert path.stat().st_mode & 0o777 == 0o600
    assert latest.read_text() == "leave weekly report intact"


def test_startup_audit_failure_is_isolated_from_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(weekly, "build_weekly_report", lambda: (_ for _ in ()).throw(RuntimeError("private")))
    weekly.startup_account_audit(tmp_path)
    assert "UNVERIFIED" in capsys.readouterr().out


def test_weekly_metrics_preserve_nanosecond_boundary_membership(tmp_path):
    db = tmp_path / "journal.db"
    make_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM trades")
        past_end = END.strftime("%Y-%m-%dT%H:%M:%S") + ".000000001Z"
        conn.execute("INSERT INTO trades VALUES('10', 'AUD_USD', ?, 200, 1)", (past_end,))
    assert weekly.build_weekly_report(db, now_utc=END).total.trades == 0
