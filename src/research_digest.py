"""Private research join: clean journal + verified costs + proposal/quote status.

This report has no broker-write, GitHub, deployment or strategy-setting API.
Historical filter comparisons are descriptive, never promotion evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.account_reconciliation import number, timestamp
from src.learning_cohort import (DEFAULT_LEARNING_COHORT_START_UTC,
                                 DEFAULT_LEARNING_RUN_TAG, load_clean_outcomes)
from src.research_registry import CATALOGUE
from src.shadow_learner import (_Trade, _candidate_specs, _deployed_revision,
                                _safe_float, _session_bucket, _trend_bucket)

SCHEMA = "mossy.research-digest.v1"
PAIRS = ("AUD_USD", "GBP_USD")
QUOTES = (*PAIRS, "EUR_USD", "XAU_USD")


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=".digest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(payload, out, sort_keys=True, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metrics(samples):
    net = [s["net"] for s in samples]
    total, peak, drawdown = Decimal(0), Decimal(0), Decimal(0)
    for value in net:
        total += value
        peak = max(peak, total)
        drawdown = max(drawdown, peak - total)
    gross = sum((s["gross"] for s in samples), Decimal(0))
    wins = sum((v for v in net if v > 0), Decimal(0))
    loss = -sum((v for v in net if v < 0), Decimal(0))
    return {"trades": len(net), "wins": sum(v > 0 for v in net), "losses": sum(v < 0 for v in net),
            "gross_pl_aud": str(gross), "net_pl_aud": str(total), "cost_drag_aud": str(gross - total),
            "expectancy_aud": str(total / len(net)) if net else None,
            "profit_factor": str(wins / loss) if loss else None,
            "closed_trade_drawdown_aud": str(drawdown) if net else None,
            "allocated_cost_trades": sum(s["allocated"] for s in samples)}


def compare_after_cost(outcomes, rows, *, as_of):
    """Match exact identities before replacing gross P/L for research only."""
    by_id = {}
    for row in rows:
        key = str(row["trade_id"])
        if key in by_id:
            raise ValueError("duplicate-ledger-trade")
        by_id[key] = row
    matched, omitted = [], 0
    for outcome in outcomes:
        row = by_id.get(outcome.trade_id)
        try:
            context = row.get("journal_context") if row else None
            if (not context or row["instrument"] != outcome.instrument or row["side"] != outcome.side
                    or context.get("run_tag") != outcome.run_tag
                    or abs((timestamp(row["entry_timestamp_utc"]) - timestamp(outcome.entry_timestamp_utc)).total_seconds()) > .001
                    or abs((timestamp(row["close_timestamp_utc"]) - timestamp(outcome.exit_timestamp_utc)).total_seconds()) > 5
                    or timestamp(row["close_timestamp_utc"]) > as_of
                    or abs(number(row["gross_pl"]) - number(outcome.realized_pnl_ccy)) > Decimal(".01")):
                raise ValueError("unmatched")
            gross, net = number(row["gross_pl"]), number(row["closed_net_pl"])
            expected = gross + number(row["financing"]) - number(row["commission"]) - number(row["guaranteed_fee"])
            if net != expected:
                raise ValueError("cost-mismatch")
            indicators = json.loads(outcome.indicators_snapshot or "{}")
            if not isinstance(indicators, dict):
                indicators = {}
            trend_known = all(_safe_float(indicators.get(k)) is not None
                              for k in ("ema_fast", "ema_slow", "ema50", "ema200"))
            feature = _Trade(timestamp(row["close_timestamp_utc"]), outcome.instrument, outcome.side,
                             _session_bucket(outcome.session_id), _safe_float(indicators.get("rsi")),
                             _trend_bucket(outcome.side, indicators) if trend_known else "UNKNOWN", float(net))
            matched.append({"feature": feature, "net": net, "gross": gross,
                            "allocated": row.get("commission_allocation") == "ALLOCATED_BY_UNITS",
                            "session_known": bool(outcome.session_id)})
        except (KeyError, ValueError, TypeError, AttributeError):
            omitted += 1
    matched.sort(key=lambda s: s["feature"].timestamp)
    split = int(len(matched) * .70)
    early, later = matched[:split], matched[split:]
    candidates = []
    for name, _, predicate in _candidate_specs()[1:]:
        def known(sample):
            if name in {"trend_full_only", "avoid_countertrend"}:
                return sample["feature"].trend != "UNKNOWN"
            if name.startswith("momentum_"):
                return sample["feature"].rsi is not None
            if name in {"london_only", "newyork_only"}:
                return sample["session_known"]
            return True
        missing = sum(not known(s) for s in matched)
        candidates.append({"name": name, "status": "INCOMPLETE_FEATURES" if missing else "HISTORICAL_ONLY",
                           "missing_features": missing,
                           "early": _metrics([s for s in early if known(s) and predicate(s["feature"])]),
                           "later": _metrics([s for s in later if known(s) and predicate(s["feature"])]),
                           "promotion_allowed": False})
    return {"status": "COMPLETE_JOIN" if matched and not omitted else "INCOMPLETE_JOIN",
            "eligible_trades": len(outcomes), "matched_trades": len(matched), "omitted_trades": omitted,
            "basis": "after-cost exact broker/journal join; spread already included",
            "full": _metrics(matched), "recent_80": _metrics(matched[-80:]),
            "early": _metrics(early), "later": _metrics(later), "candidates": candidates,
            "split": "70/30 chronological closed trades; retrospective, not untouched prospective validation"}


def _quote_status(data_dir, now):
    try:
        source = json.loads((Path(data_dir) / "practice_quotes.status.json").read_text())
        observed = timestamp(source["observed_at_utc"])
        age = (now - observed).total_seconds()
        state = source.get("state")
        if state not in {"fresh", "market_closed", "incomplete", "stale", "partial", "error", "storage_error"}:
            state = "unknown"
        if not 0 <= age <= 120:
            state = "stale"
        coverage = {}
        for pair in QUOTES:
            item = source.get("coverage", {}).get(pair, {})
            count = item.get("rows")
            if type(count) is int and count >= 0:
                coverage[pair] = {"rows": count,
                                  "first_quote_utc": timestamp(item["first_quote_utc"]).isoformat() if count else None,
                                  "last_quote_utc": timestamp(item["last_quote_utc"]).isoformat() if count else None}
        return {"state_at_digest": state, "observed_at_utc": observed.isoformat(),
                "coverage": coverage, "gap_audit_complete": False, "executable_snapshot": False}
    except (OSError, ValueError, TypeError, KeyError):
        return {"state_at_digest": "unknown", "coverage": {}, "gap_audit_complete": False,
                "executable_snapshot": False}


def publish_digest(data_dir, journal_path, history_review):
    """Use a successful same-account history snapshot; never reuse failed reads."""
    root = Path(data_dir) / "memory" / "research"
    now = datetime.now(timezone.utc)
    try:
        if os.getenv("MODE") != "demo" or os.getenv("OANDA_ENV") != "practice":
            raise ValueError("practice-required")
        account = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("ACCOUNT_ID")
        fingerprint = hashlib.sha256((account or "").encode()).hexdigest()[:16]
        if (not account or not isinstance(history_review, dict)
                or history_review.get("ledger_status") != "VERIFIED"
                or history_review.get("currency") != "AUD"
                or history_review.get("account_fingerprint") != fingerprint
                or not 0 <= (now - timestamp(history_review["generated_at_utc"])).total_seconds() <= 4500):
            raise ValueError("history-unverified")
        database = Path(data_dir) / "memory" / "oanda-practice" / fingerprint / "enriched" / "history.sqlite"
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.execute("BEGIN")
            stored = json.loads(db.execute("SELECT value FROM state WHERE key='review'").fetchone()[0])
            if stored != history_review:
                raise ValueError("history-snapshot-changed")
            rows = [json.loads(r[0]) for r in db.execute("SELECT record_json FROM trade_review ORDER BY CAST(trade_id AS INTEGER)")]
        tag = os.getenv("ADAPTIVE_RUN_TAG", "").strip() or DEFAULT_LEARNING_RUN_TAG
        start = os.getenv("ADAPTIVE_WINDOW_START_UTC", "").strip() or DEFAULT_LEARNING_COHORT_START_UTC
        outcomes = load_clean_outcomes(journal_path, instruments=PAIRS, run_tag=tag, entry_start_utc=start,
                                       as_of_utc=history_review["generated_at_utc"])
        comparison = compare_after_cost(outcomes, rows, as_of=timestamp(history_review["generated_at_utc"]))
        expected_hash = hashlib.sha256(CATALOGUE.read_bytes()).hexdigest()
        try:
            registry = json.loads((root / "registry.json").read_text())
            imported = (registry.get("catalogue_sha256") == expected_hash
                        and registry.get("status") == "IMPORTED_PROPOSALS"
                        and registry.get("execution_enabled") is False and registry.get("promotion_enabled") is False)
        except (OSError, ValueError, TypeError):
            imported = False
        quote_status = _quote_status(data_dir, now)
        report = {"schema": SCHEMA, "status": "RESEARCH_ONLY", "generated_at_utc": now.isoformat(),
                  "history_generated_at_utc": history_review["generated_at_utc"],
                  "worker_revision": _deployed_revision(), "account_currency": "AUD",
                  "archive_closed_trades": history_review["closed"]["trades"],
                  "comparison": comparison, "quotes": quote_status,
                  "catalogue_sha256": expected_hash, "catalogue_import_verified": imported,
                  "proposal_count": len(json.loads(CATALOGUE.read_bytes())["experiments"]) if imported else None,
                  "validated_experiments": 0, "promotion_allowed": False, "execution_enabled": False,
                  "blockers": ["prospective-labels-and-gap-audit-required", "forward-test-required",
                               "news-study-remains-separate-protocol", "independent-promotion-review-required"],
                  "next_work": ["resolve-quote-gaps-and-price-evidence", "review-after-cost-filter-comparisons",
                                "freeze-one-prospective-test-before-collecting-outcomes"],
                  "limitations": ["Historical executed-trade filters omit skipped opportunities and cannot prove new entries.",
                                  "Individual results with allocated commission are estimates; totals retain broker costs.",
                                  "The cash-risk learner and trading strategy are unchanged."]}
        if comparison["status"] != "COMPLETE_JOIN":
            report["blockers"].insert(0, "incomplete-clean-cohort-cost-join")
        if not imported:
            report["blockers"].insert(0, "catalogue-import-unverified")
        _write(root / "digest.json", report)
        _write(root / "digest-status.json", {"status": "AVAILABLE", "generated_at_utc": now.isoformat()})
        summary = {k: report[k] for k in ("schema", "status", "generated_at_utc", "history_generated_at_utc",
                   "worker_revision", "account_currency", "archive_closed_trades", "catalogue_import_verified",
                   "proposal_count", "validated_experiments", "promotion_allowed", "execution_enabled", "blockers", "next_work")}
        summary["after_cost"] = {k: comparison[k] for k in ("status", "eligible_trades", "matched_trades", "omitted_trades",
                                                            "full", "recent_80", "early", "later", "split")}
        summary["quote_status_at_digest"] = quote_status["state_at_digest"]
        print("[RESEARCH_DIGEST] " + json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        for candidate in comparison["candidates"]:
            print("[RESEARCH_CANDIDATE] " + json.dumps({"generated_at_utc": report["generated_at_utc"],
                  "basis": "after-cost historical executed-trade subset", **candidate}, sort_keys=True, allow_nan=False), flush=True)
        return report
    except Exception:
        status = {"schema": SCHEMA, "status": "UNAVAILABLE", "generated_at_utc": now.isoformat(),
                  "reason": "history-or-research-evidence-unavailable", "promotion_allowed": False}
        try:
            if os.getenv("MODE") == "demo" and os.getenv("OANDA_ENV") == "practice":
                _write(root / "digest-status.json", status)
        except OSError:
            pass
        print("[RESEARCH_DIGEST] " + json.dumps(status, sort_keys=True), flush=True)
        return None
