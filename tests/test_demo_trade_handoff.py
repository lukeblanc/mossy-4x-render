"""Offline tests for private source evidence; no destination broker or orders."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from src.demo_trade_handoff import build_demo_trade_handoff


NOW = datetime(2026, 11, 10, 7, 30, tzinfo=timezone.utc)


def evidence():
    summary = {"id": "42", "instrument": "AUD_USD", "stopLossOrderID": "43",
               "takeProfitOrderID": "44", "currentUnits": "100", "price": "0.70261",
               "openTime": "2026-11-10T07:29:00Z"}
    detail = {
        "id": "42", "instrument": "AUD_USD", "state": "OPEN",
        "currentUnits": "100", "price": "0.70261", "openTime": summary["openTime"],
        "stopLossOrder": {"id": "43", "tradeID": "42", "type": "STOP_LOSS",
                          "state": "PENDING", "price": "0.70161"},
        "takeProfitOrder": {"id": "44", "tradeID": "42", "type": "TAKE_PROFIT",
                            "state": "PENDING", "price": "0.70461"},
    }
    return {
        "open_trades": [summary], "trade_details": {"42": detail},
        "protection_audited": True, "mode": "demo", "oanda_environment": "practice",
        "account_currency": "AUD", "revision": "a" * 40,
        "demo_equity_floor_aud": 500.0,
        "snapshot_observed_at": NOW - timedelta(seconds=2), "observed_at": NOW,
        "broker_entry_halted": False, "demo_equity_floor_halted": False,
    }


def test_fresh_exact_demo_trade_is_source_evidence_never_a_destination_order():
    inputs = evidence()
    original = deepcopy(inputs)
    result = build_demo_trade_handoff(**inputs)
    assert inputs == original
    assert result["schema_version"] == "mossy.demo-trade-handoff.v1"
    assert result["read_status"] == "open" and result["handoff_ready"] is True
    assert result["destination_order_ready"] is False and result["blockers"] == []
    assert result["snapshot_age_seconds"] == 2.0 and result["freshness"] == "fresh"
    assert result["source"]["account_currency"] == "AUD"
    assert result["source"]["protection_audit_verified"] is True
    assert result["safety"] == {"demo_equity_floor_aud": 500.0,
                                "broker_entry_halted": False,
                                "demo_equity_floor_halted": False}
    trade = result["trades"][0]
    assert trade["source_trade_id"] == "42" and trade["direction"] == "BUY"
    assert trade["signed_units"] == "100" and trade["entry_fill_price"] == "0.70261"
    assert trade["opened_at_utc"] == "2026-11-10T07:29:00Z"
    assert trade["open_state_verified"] is True
    assert trade["protective_stop_verified"] is True
    assert trade["stop_loss"]["price"] == "0.70161"
    assert trade["take_profit"]["price"] == "0.70461"
    assert trade["take_profit_state"] == "pending" and trade["take_profit_verified"] is True
    assert trade["take_profit_absence_verified"] is False
    json.dumps(result, allow_nan=False)


def test_both_current_positions_are_retained_with_their_actual_signed_units():
    inputs = evidence()
    second = deepcopy(inputs["trade_details"]["42"])
    second.update(id="52", instrument="GBP_USD", currentUnits="-90", price="1.34")
    second["stopLossOrder"].update(id="53", tradeID="52", price="1.35")
    second["takeProfitOrder"].update(id="54", tradeID="52", price="1.32")
    inputs["open_trades"].append({"id": "52", "instrument": "GBP_USD",
                                  "stopLossOrderID": "53", "takeProfitOrderID": "54"})
    inputs["trade_details"]["52"] = second
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is True and result["open_trade_count"] == 2
    assert [(t["source_trade_id"], t["direction"], t["signed_units"])
            for t in result["trades"]] == [("42", "BUY", "100"), ("52", "SELL", "-90")]


def test_exact_open_detail_can_verify_absence_of_a_take_profit():
    inputs = evidence()
    inputs["open_trades"][0].pop("takeProfitOrderID")
    inputs["trade_details"]["42"].pop("takeProfitOrder")
    result = build_demo_trade_handoff(**inputs)
    trade = result["trades"][0]
    assert result["handoff_ready"] is True
    assert trade["take_profit"] is None and trade["take_profit_state"] == "absent"
    assert trade["take_profit_absence_verified"] is True


@pytest.mark.parametrize("keep_id,present_null", [(True, False), (False, True)])
def test_missing_or_null_target_evidence_is_not_invented(keep_id, present_null):
    inputs = evidence()
    if not keep_id:
        inputs["open_trades"][0].pop("takeProfitOrderID")
    inputs["trade_details"]["42"].pop("takeProfitOrder")
    if present_null:
        inputs["trade_details"]["42"]["takeProfitOrder"] = None
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False
    assert result["trades"][0]["take_profit_state"] == "unknown"
    assert result["trades"][0]["take_profit_absence_verified"] is False


def test_guaranteed_stop_is_preserved_as_its_actual_order_type():
    inputs = evidence()
    detail = inputs["trade_details"]["42"]
    stop = detail.pop("stopLossOrder")
    stop["type"] = "GUARANTEED_STOP_LOSS"
    detail["guaranteedStopLossOrder"] = stop
    summary = inputs["open_trades"][0]
    summary["guaranteedStopLossOrderID"] = summary.pop("stopLossOrderID")
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is True
    assert result["trades"][0]["stop_loss"]["type"] == "GUARANTEED_STOP_LOSS"


def test_actual_trailing_stop_and_target_need_not_bracket_original_fill():
    inputs = evidence()
    inputs["trade_details"]["42"]["stopLossOrder"]["price"] = "0.70300"
    inputs["trade_details"]["42"]["takeProfitOrder"]["price"] = "0.70200"
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is True
    assert result["trades"][0]["stop_loss"]["price"] == "0.703"
    assert result["trades"][0]["take_profit"]["price"] == "0.702"
    assert result["destination_order_ready"] is False


@pytest.mark.parametrize("field,value", [
    ("id", "different"), ("tradeID", "999"), ("state", "CANCELLED"),
    ("type", "TAKE_PROFIT"), ("price", "NaN"), ("price", "0"), ("price", "Infinity"),
])
def test_pending_stop_identity_type_and_price_must_be_verified(field, value):
    inputs = evidence()
    inputs["trade_details"]["42"]["stopLossOrder"][field] = value
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False
    assert result["trades"][0]["protective_stop_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("tradeID", "999"), ("state", "FILLED"), ("type", "STOP_LOSS"), ("price", "bad"),
])
def test_target_fields_are_actual_pending_broker_order_fields(field, value):
    inputs = evidence()
    inputs["trade_details"]["42"]["takeProfitOrder"][field] = value
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False
    assert result["trades"][0]["take_profit_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("id", "999"), ("instrument", "GBP_USD"), ("state", "CLOSED"),
    ("currentUnits", "0"), ("currentUnits", "NaN"), ("price", "bad"),
    ("openTime", "2026-11-10T07:29:00"), ("openTime", "invalid"),
    ("openTime", "2026-11-10T07:35:00Z"),
])
def test_current_trade_identity_and_actual_fill_evidence_are_required(field, value):
    inputs = evidence()
    inputs["trade_details"]["42"][field] = value
    result = build_demo_trade_handoff(**inputs)
    assert result["read_status"] == "incomplete" and result["handoff_ready"] is False


@pytest.mark.parametrize("field,value", [
    ("price", "0.70262"), ("currentUnits", "101"),
    ("openTime", "2026-11-10T07:28:00Z"), ("stopLossOrderID", "999"),
])
def test_summary_and_exact_audited_detail_cannot_conflict(field, value):
    inputs = evidence()
    inputs["open_trades"][0][field] = value
    assert build_demo_trade_handoff(**inputs)["handoff_ready"] is False


def test_summaries_alone_cannot_supply_protection_or_prices():
    inputs = evidence()
    inputs["trade_details"] = None
    result = build_demo_trade_handoff(**inputs)
    assert result["read_status"] == "incomplete" and result["handoff_ready"] is False
    assert result["trades"][0]["entry_fill_price"] is None
    assert result["trades"][0]["stop_loss"] is None


def test_audited_detail_id_set_must_match_the_same_current_snapshot():
    inputs = evidence()
    inputs["trade_details"]["99"] = deepcopy(inputs["trade_details"]["42"])
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False
    assert "audited-detail-set-mismatch" in result["blockers"]
    inputs["open_trades"] = []
    result = build_demo_trade_handoff(**inputs)
    assert result["read_status"] == "incomplete"
    assert result["open_trade_count"] == 0 and result["handoff_ready"] is False


@pytest.mark.parametrize("field,value,code", [
    ("mode", "live", "source-not-demo-practice"),
    ("oanda_environment", "live", "source-not-demo-practice"),
    ("account_currency", "USD", "source-currency-unverified"),
    ("account_currency", None, "source-currency-unverified"),
    ("account_currency", {}, "source-currency-unverified"),
    ("revision", "unknown", "worker-revision-unverified"),
    ("protection_audited", False, "protection-audit-unverified"),
    ("protection_audited", "true", "protection-audit-unverified"),
    ("broker_entry_halted", True, "broker-entry-halted"),
    ("broker_entry_halted", None, "broker-entry-halt-unknown"),
    ("demo_equity_floor_halted", True, "demo-equity-floor-halted"),
    ("demo_equity_floor_halted", 0, "demo-equity-floor-halt-unknown"),
    ("demo_equity_floor_aud", None, "demo-equity-floor-unverified"),
    ("demo_equity_floor_aud", 499.0, "demo-equity-floor-unverified"),
    ("demo_equity_floor_aud", True, "demo-equity-floor-unverified"),
    ("demo_equity_floor_aud", "500", "demo-equity-floor-unverified"),
    ("demo_equity_floor_aud", float("nan"), "demo-equity-floor-unverified"),
])
def test_unverified_source_or_safety_state_never_readies_a_handoff(field, value, code):
    inputs = evidence()
    inputs[field] = value
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False and code in result["blockers"]
    assert result["destination_order_ready"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("age,freshness", [(61, "stale"), (-6, "future")])
def test_old_or_future_cached_evidence_is_never_retimestamped_as_current(age, freshness):
    inputs = evidence()
    inputs["snapshot_observed_at"] = NOW - timedelta(seconds=age)
    result = build_demo_trade_handoff(**inputs)
    assert result["freshness"] == freshness and result["handoff_ready"] is False
    assert result["trades"] == []
    assert result["broker_observed_at_utc"] != result["observed_at_utc"]


def test_missing_snapshot_and_verified_flat_are_different():
    inputs = evidence()
    inputs["open_trades"] = None
    unavailable = build_demo_trade_handoff(**inputs)
    assert unavailable["read_status"] == "unavailable"
    assert unavailable["open_trade_count"] is None
    inputs["open_trades"] = []
    inputs["trade_details"] = {}
    flat = build_demo_trade_handoff(**inputs)
    assert flat["read_status"] == "flat" and flat["open_trade_count"] == 0
    assert flat["trades"] == [] and flat["blockers"] == []
    assert flat["handoff_ready"] is False


@pytest.mark.parametrize("count", [2, 11])
def test_duplicate_ids_and_over_limit_snapshots_do_not_hide_exposure(count):
    inputs = evidence()
    inputs["open_trades"] *= count
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False and result["trades"] == []
    assert result["read_status"] != "flat" and result["open_trade_count"] == count


def test_private_snapshot_is_allowlisted_and_does_not_echo_sensitive_or_unknown_inputs():
    inputs = evidence()
    inputs["open_trades"][0].update(accountID="private-account", balance="99999")
    inputs["trade_details"]["42"].update(accountID="private-account", unrealizedPL="99",
                                            clientExtensions={"comment": "private-note"})
    inputs["trade_details"]["42"]["stopLossOrder"]["secret"] = "private-token"
    rendered = json.dumps(build_demo_trade_handoff(**inputs), allow_nan=False)
    for secret in ("private-account", "99999", "private-note", "private-token",
                   "unrealizedPL", "accountID", "clientExtensions"):
        assert secret not in rendered
    inputs["revision"] = "private-revision-token"
    assert "private-revision-token" not in json.dumps(build_demo_trade_handoff(**inputs))


def test_snapshot_id_is_deterministic_and_changes_with_source_observation():
    inputs = evidence()
    first = build_demo_trade_handoff(**inputs)
    assert build_demo_trade_handoff(**deepcopy(inputs)) == first
    inputs["observed_at"] = NOW + timedelta(seconds=1)
    assert build_demo_trade_handoff(**inputs)["snapshot_id"] != first["snapshot_id"]


def test_extreme_numeric_input_is_rejected_without_logging_or_overflow():
    inputs = evidence()
    inputs["demo_equity_floor_aud"] = 10 ** 5_000
    inputs["trade_details"]["42"]["currentUnits"] = 10 ** 5_000
    result = build_demo_trade_handoff(**inputs)
    assert result["handoff_ready"] is False
    assert result["safety"]["demo_equity_floor_aud"] is None
    assert result["trades"][0]["signed_units"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("maximum_age", [0, -1, 61, True, "60", float("nan")])
def test_freshness_limit_cannot_be_disabled_or_extended(maximum_age):
    with pytest.raises(ValueError, match="maximum age"):
        build_demo_trade_handoff(**evidence(), max_age_seconds=maximum_age)
