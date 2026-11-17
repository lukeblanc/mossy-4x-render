import copy
import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.learning_cohort import CleanOutcome
from src.practice_history import build_review, _refresh_research_digest
from src.research_digest import compare_after_cost, publish_digest
from src.research_registry import publish_registry
from test_practice_history import ACCOUNT, history


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("OANDA_ENV", "practice")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("RENDER_GIT_COMMIT", "a" * 40)
    now = datetime.now(timezone.utc)
    indicators = {"rsi": 57, "ema_fast": 1.2, "ema_slow": 1.1, "ema50": 1.15, "ema200": 1.0}
    contexts = [{"trade_id": str(n), "instrument": "AUD_USD", "timestamp_utc": f"2026-11-01T00:00:0{n}Z",
                 "run_tag": "MINI_RUN", "session_id": "LONDON", "indicators_snapshot": indicators}
                for n in (3, 4)]
    review, rows, _ = build_review(history(), account=ACCOUNT, cursor=8, balance="101.1", journal_rows=contexts)
    review["generated_at_utc"] = now.isoformat()
    outcomes = [CleanOutcome(str(n), contexts[i]["timestamp_utc"], "2026-11-01T00:00:08Z", "AUD_USD",
                             "BUY", "LONDON", json.dumps(indicators), float(rows[i]["gross_pl"]), "MINI_RUN")
                for i, n in enumerate((3, 4))]
    journal = tmp_path / "journal.sqlite"
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE trades (trade_id TEXT, timestamp_utc TEXT, exit_timestamp_utc TEXT, "
                   "instrument TEXT, side TEXT, session_id TEXT, indicators_snapshot TEXT, "
                   "realized_pnl_ccy REAL, run_tag TEXT, broker_confirmed INT)")
        for o in outcomes:
            db.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,1)",
                       (o.trade_id, o.entry_timestamp_utc, o.exit_timestamp_utc, o.instrument, o.side,
                        o.session_id, o.indicators_snapshot, o.realized_pnl_ccy, o.run_tag))
    ledger = tmp_path / "memory/oanda-practice" / review["account_fingerprint"] / "enriched/history.sqlite"
    ledger.parent.mkdir(parents=True)
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("INSERT INTO state VALUES ('review',?)", (json.dumps(review),))
        db.execute("CREATE TABLE trade_review (trade_id TEXT PRIMARY KEY, record_json TEXT)")
        for row in rows:
            db.execute("INSERT INTO trade_review VALUES (?,?)", (row["trade_id"], json.dumps(row, default=str)))
    publish_registry(tmp_path)
    return tmp_path, journal, review, rows, outcomes, now, ledger


def test_costs_are_joined_without_subtracting_spread_twice(evidence):
    _, _, _, rows, outcomes, now, _ = evidence
    result = compare_after_cost(outcomes, rows, as_of=now)
    assert result["status"] == "COMPLETE_JOIN"
    assert result["full"]["trades"] == 2
    assert Decimal(result["full"]["net_pl_aud"]) == Decimal("1.1")
    assert Decimal(result["full"]["gross_pl_aud"]) == Decimal("2.5")
    assert Decimal(result["full"]["cost_drag_aud"]) == Decimal("1.4")
    assert result["full"]["allocated_cost_trades"] == 2
    assert result["early"]["trades"] == result["later"]["trades"] == 1
    assert all(not c["promotion_allowed"] for c in result["candidates"])


@pytest.mark.parametrize("field,value", [("instrument", "GBP_USD"), ("side", "SELL"),
    ("entry_timestamp_utc", "2026-11-02T00:00:03Z"), ("close_timestamp_utc", "2026-11-02T00:00:08Z"),
    ("gross_pl", "99"), ("closed_net_pl", "99"), ("closed_net_pl", "NaN"), ("journal_context", None)])
def test_identity_and_cost_mismatches_are_reported_not_silently_used(evidence, field, value):
    _, _, _, rows, outcomes, now, _ = evidence
    rows[0][field] = value
    result = compare_after_cost(outcomes, rows, as_of=now)
    assert result["status"] == "INCOMPLETE_JOIN"
    assert result["omitted_trades"] == 1 and result["matched_trades"] == 1


def test_missing_features_are_unknown_and_pair_filters_still_report(evidence):
    _, _, _, rows, outcomes, now, _ = evidence
    outcomes[0] = replace(outcomes[0], indicators_snapshot="{}", session_id="")
    result = compare_after_cost(outcomes, rows, as_of=now)
    candidates = {c["name"]: c for c in result["candidates"]}
    assert candidates["avoid_countertrend"]["status"] == "INCOMPLETE_FEATURES"
    assert candidates["momentum_50"]["missing_features"] == 1
    assert candidates["london_only"]["missing_features"] == 1
    assert candidates["aud_only"]["status"] == "HISTORICAL_ONLY"


def test_duplicate_ledger_identity_is_rejected(evidence):
    _, _, _, rows, outcomes, now, _ = evidence
    with pytest.raises(ValueError):
        compare_after_cost(outcomes, rows + [copy.deepcopy(rows[0])], as_of=now)


def test_report_is_private_preserves_sources_and_never_claims_validation(evidence, capsys):
    root, journal, review, _, _, _, ledger = evidence
    originals = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (journal, ledger)]
    report = publish_digest(root, journal, review)
    assert report["status"] == "RESEARCH_ONLY"
    assert report["comparison"]["matched_trades"] == 2
    assert report["proposal_count"] == 6 and report["validated_experiments"] == 0
    assert report["catalogue_import_verified"]
    assert not report["promotion_allowed"] and not report["execution_enabled"]
    assert report["quotes"]["state_at_digest"] == "unknown"
    assert not report["quotes"]["gap_audit_complete"]
    saved = root / "memory/research/digest.json"
    assert saved.stat().st_mode & 0o777 == 0o600
    assert originals == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (journal, ledger)]
    output = capsys.readouterr().out
    assert ACCOUNT not in output and '"trade_id"' not in output
    assert '"balance_ccy"' not in output and '"account_fingerprint"' not in output


@pytest.mark.parametrize("failure", ["missing", "stale", "future", "different-account", "db-change", "live"])
def test_failed_or_changed_history_never_reuses_old_success(evidence, monkeypatch, failure):
    root, journal, review, _, _, now, ledger = evidence
    assert publish_digest(root, journal, review)
    saved = root / "memory/research/digest.json"
    before = saved.read_bytes()
    if failure == "missing": review = None
    if failure == "stale": review["generated_at_utc"] = (now - timedelta(hours=2)).isoformat()
    if failure == "future": review["generated_at_utc"] = (now + timedelta(hours=2)).isoformat()
    if failure == "different-account": monkeypatch.setenv("OANDA_ACCOUNT_ID", "other-private-account")
    if failure == "live": monkeypatch.setenv("MODE", "live")
    if failure == "db-change":
        with sqlite3.connect(ledger) as db:
            db.execute("UPDATE state SET value='{}' WHERE key='review'")
    assert publish_digest(root, journal, review) is None
    assert saved.read_bytes() == before
    if failure != "live":
        assert json.loads((root / "memory/research/digest-status.json").read_text())["status"] == "UNAVAILABLE"


def test_failed_digest_does_not_interrupt_history_or_trading(evidence, monkeypatch, capsys):
    def broken(*args): raise RuntimeError("private-account private-token")
    monkeypatch.setattr("src.research_digest.publish_digest", broken)
    _refresh_research_digest(evidence[0], evidence[1], evidence[2])
    assert 'digest-refresh-failed' in capsys.readouterr().out


def test_unverified_catalogue_blocks_import_claim(evidence):
    root, journal, review, *_ = evidence
    registry = root / "memory/research/registry.json"
    data = json.loads(registry.read_text())
    data["catalogue_sha256"] = "wrong"
    registry.write_text(json.dumps(data))
    result = publish_digest(root, journal, review)
    assert not result["catalogue_import_verified"]
    assert "catalogue-import-unverified" in result["blockers"]


def test_quote_snapshot_has_age_and_does_not_prove_replay_coverage(evidence):
    root, journal, review, _, _, now, _ = evidence
    quotes = {"state": "fresh", "observed_at_utc": (now - timedelta(minutes=10)).isoformat(),
              "coverage": {"AUD_USD": {"rows": 5, "first_quote_utc": (now - timedelta(hours=1)).isoformat(),
                                        "last_quote_utc": (now - timedelta(minutes=10)).isoformat()}}}
    (root / "practice_quotes.status.json").write_text(json.dumps(quotes))
    result = publish_digest(root, journal, review)
    assert result["quotes"]["state_at_digest"] == "stale"
    assert result["quotes"]["coverage"]["AUD_USD"]["rows"] == 5
    assert not result["quotes"]["executable_snapshot"]
