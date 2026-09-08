import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.broker import Broker, opened_trade_fill
from app.config import settings
from order_fakes import confirmed_order_result
from src import main
from src.decision_engine import Evaluation
from test_broker_read_recovery import runtime


def test_order_and_protective_transaction_ids_never_replace_trade_id():
    payload = confirmed_order_result(trade_id="42", price=0.72123)["response"]
    payload["orderFillTransaction"]["id"] = "40"
    fill = opened_trade_fill(payload)
    assert fill == {"trade_id": "42", "price": 0.72123, "units": 100,
                    "timestamp": datetime(2026, 9, 8, 6, tzinfo=timezone.utc)}
    assert main._order_ticket({"response": payload}) == "42"


@pytest.mark.parametrize("path,value", [
    (("tradeOpened",), None), (("tradeOpened", "tradeID"), "local-1"),
    (("tradeOpened", "tradeID"), "0"), (("tradeOpened", "price"), "NaN"),
    (("tradeOpened", "price"), "Infinity"), (("tradeOpened", "price"), "0"),
    (("tradeOpened", "units"), "NaN"), (("tradeOpened", "units"), "0"),
    (("time",), "bad-date"), (("time",), "2026-09-08T06:00:00"),
])
def test_incomplete_or_invalid_opening_fill_is_not_confirmed(path, value):
    payload = confirmed_order_result()["response"]
    target = payload["orderFillTransaction"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert opened_trade_fill(payload) is None


@pytest.mark.parametrize("payload,expected", [
    ({"orderCreateTransaction": {"id": "1"}}, "UNKNOWN"),
    ({"orderCancelTransaction": {"id": "2"}}, "CANCELLED"),
    ({"orderRejectTransaction": {"id": "2"}}, "REJECTED"),
    ({"orderFillTransaction": {"id": "2", "tradesClosed": [{"tradeID": "10"}]}}, "UNKNOWN"),
    ({"orderFillTransaction": {"id": "2", "tradeReduced": {"tradeID": "10"}}}, "UNKNOWN"),
    ([], "UNKNOWN"),
])
def test_http_success_does_not_mean_a_new_trade(monkeypatch, payload, expected):
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "OANDA_API_KEY", "test-only")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "test-only")
    from contextlib import nullcontext
    client = SimpleNamespace(post=lambda *args, **kwargs:
                             SimpleNamespace(status_code=201, json=lambda: deepcopy(payload)))
    monkeypatch.setattr(Broker, "_client", lambda self: nullcontext(client))
    assert Broker().place_order("EUR_USD", "BUY", 100, sl_distance=0.001)["status"] == expected


def prepare_decision(runtime, monkeypatch):
    broker, risk, guard, engine = runtime
    risk.should_open.return_value = (True, "ok")
    risk.risk_per_trade_pct = 0.0025
    risk.demo_mode = True
    risk.sl_distance_from_atr.return_value = 0.001
    risk.tp_distance_from_atr.return_value = 0.002
    engine.evaluate_all.return_value = [Evaluation(
        instrument=instrument, signal="BUY", diagnostics={
            "atr": 0.001, "close": 1.2, "ema_trend_fast": 1.21, "ema_trend_slow": 1.19},
        reason="test", market_active=True, candles=[],
    ) for instrument in ("EUR_USD", "GBP_USD")]
    monkeypatch.setattr(main, "_utc_now", lambda: datetime(2026, 9, 8, 6, tzinfo=timezone.utc))
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *a, **k: SimpleNamespace(
        allowed=True, session=None, in_session=True, risk_scale=1.0, mode="SOFT", reason=None))
    monkeypatch.setattr(main, "_macd_confirms", lambda *a: (True, 0, 0, 0))
    monkeypatch.setattr(main, "_orb_filter", lambda *a: (True, None, {}))
    monkeypatch.setattr(main.position_sizer, "units_for_risk", lambda *a, **k: (100, {}))
    return broker, risk, guard, engine


def test_entry_journal_uses_actual_fill_price_size_and_time(runtime, monkeypatch):
    broker, risk, _, engine = prepare_decision(runtime, monkeypatch)
    engine.evaluate_all.return_value = engine.evaluate_all.return_value[:1]
    broker.place_order.return_value = confirmed_order_result(units=90, price=1.20123, trade_id="42")
    asyncio.run(main.decision_cycle())
    record = main.journal.record_entry.call_args.kwargs
    assert record["trade_id"] == "42"
    assert record["units"] == 90
    assert record["entry_price"] == 1.20123
    assert record["timestamp_utc"] == datetime(2026, 9, 8, 6, tzinfo=timezone.utc)
    assert record["stop_loss_price"] == pytest.approx(1.20023)
    assert record["gating_flags"]["signal_price"] == 1.2
    risk.register_entry.assert_called_once()


@pytest.mark.parametrize("result", [{"status": "UNKNOWN"}, {"status": "SENT"}])
def test_uncertain_order_stops_additional_entries_and_does_not_invent_journal_rows(runtime, monkeypatch, result):
    broker, risk, _, engine = prepare_decision(runtime, monkeypatch)
    broker.place_order.return_value = result
    asyncio.run(main.decision_cycle())
    broker.place_order.assert_called_once()
    main.journal.record_entry.assert_not_called()
    risk.register_entry.assert_not_called()
    engine.mark_trade.assert_not_called()
    assert not main._ACTIVE_CYCLE_TICKS
