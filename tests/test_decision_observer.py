from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from order_fakes import confirmed_order_result
from src import main
from src.decision_engine import DecisionEngine, Evaluation
from src.decision_observer import (
    DecisionObservationDraft,
    DecisionObservationLedger,
    DecisionObservationSink,
    configuration_fingerprint,
)
from test_broker_read_recovery import runtime


BAR_OPEN = datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)


def _draft(
    *,
    config_hash: str | None = None,
    observed_at: datetime | None = None,
) -> DecisionObservationDraft:
    return DecisionObservationDraft(
        observed_at_utc=observed_at or BAR_OPEN + timedelta(minutes=6),
        instrument="AUD_USD",
        timeframe="M5",
        runtime_signal="BUY",
        runtime_signal_reason="bullish",
        market_active=True,
        completed_bar_open_utc=BAR_OPEN,
        completed_bar_close_utc=BAR_OPEN + timedelta(minutes=5),
        observer_signal="BUY",
        observer_signal_reason="bullish",
        completed_diagnostics={"close": 0.66, "atr": 0.001, "rsi": 62.0},
        runtime_revision="abc123",
        config_fingerprint=config_hash or configuration_fingerprint({"timeframe": "M5"}),
        reference_price=0.66,
        stop_distance=0.0012,
        target_distance=0.001,
    )


def _record(
    *,
    observed_at: datetime | None = None,
    outcome: str = "blocked",
    stage: str = "risk",
    reason: str = "daily-loss-cap",
):
    draft = _draft(observed_at=observed_at)
    draft.set_session(mode="SOFT", name="london")
    draft.set_spread(0.7)
    record = draft.finalize(
        outcome=outcome,
        gate_stage=stage,
        gate_reason=reason,
    )
    assert record is not None
    return record


def _evaluation(
    signal: str = "BUY",
    reason: str = "bullish",
    instrument: str = "AUD_USD",
) -> Evaluation:
    return Evaluation(
        instrument=instrument,
        signal=signal,
        diagnostics={
            "atr": 0.001,
            "atr_baseline_50": 0.001,
            "close": 0.66,
            "ema_trend_fast": 0.661,
            "ema_trend_slow": 0.659,
        },
        reason=reason,
        market_active=True,
        candles=[{"o": 0.659, "h": 0.661, "l": 0.658, "c": 0.66}],
        completed_bar_open_utc=BAR_OPEN,
        completed_bar_close_utc=BAR_OPEN + timedelta(minutes=5),
        completed_signal=signal,
        completed_reason=reason,
        completed_diagnostics={
            "atr": 0.001,
            "atr_baseline_50": 0.001,
            "close": 0.66,
            "ema_trend_fast": 0.661,
            "ema_trend_slow": 0.659,
        },
    )


class CaptureSink:
    def __init__(self, *, fail: bool = False, accept: bool = True) -> None:
        self.records = []
        self.fail = fail
        self.accept = accept

    def submit(self, observation):
        if self.fail:
            raise RuntimeError("queue unavailable")
        self.records.append(observation)
        return self.accept


def test_completed_observer_excludes_forming_candle_without_changing_runtime_input():
    config = {
        "instruments": ["AUD_USD"],
        "candles_to_fetch": 5,
        "timeframe": "M5",
        "ema_fast": 2,
        "ema_slow": 3,
        "rsi_length": 2,
        "atr_length": 2,
        "min_atr": 0.00001,
    }
    candles = [
        {"time": "2026-09-08T05:45:00Z", "complete": True, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0},
        {"time": "2026-09-08T05:50:00Z", "complete": True, "o": 1.0, "h": 1.2, "l": 1.0, "c": 1.1},
        {"time": "2026-09-08T05:55:00Z", "complete": True, "o": 1.1, "h": 1.3, "l": 1.1, "c": 1.2},
        {"time": "2026-09-08T06:00:00Z", "complete": False, "o": 1.2, "h": 9.0, "l": 0.1, "c": 9.0},
    ]
    engine = DecisionEngine(config, candle_fetcher=lambda *args, **kwargs: candles)

    evaluation = engine.evaluate_all()[0]

    assert evaluation.diagnostics["close"] == 9.0
    assert evaluation.completed_diagnostics["close"] == 1.2
    assert evaluation.completed_bar_open_utc == datetime(2026, 9, 8, 5, 55, tzinfo=timezone.utc)
    assert evaluation.completed_bar_close_utc == datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)


def test_completed_observer_uses_stable_lookback_across_new_forming_candle():
    config = {
        "instruments": ["AUD_USD"],
        "candles_to_fetch": 4,
        "timeframe": "M5",
        "ema_fast": 2,
        "ema_slow": 3,
        "rsi_length": 2,
        "atr_length": 2,
        "min_atr": 0.00001,
    }
    engine = DecisionEngine(config, candle_fetcher=lambda *args, **kwargs: [])

    completed = [
        {"time": "2026-09-08T05:45:00Z", "complete": True, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0},
        {"time": "2026-09-08T05:50:00Z", "complete": True, "o": 1.0, "h": 1.2, "l": 1.0, "c": 1.1},
        {"time": "2026-09-08T05:55:00Z", "complete": True, "o": 1.1, "h": 1.3, "l": 1.1, "c": 1.2},
        {"time": "2026-09-08T06:00:00Z", "complete": True, "o": 1.2, "h": 1.4, "l": 1.2, "c": 1.3},
    ]
    forming = {
        "time": "2026-09-08T06:05:00Z",
        "complete": False,
        "o": 1.3,
        "h": 9.0,
        "l": 0.1,
        "c": 9.0,
    }

    boundary = engine._completed_bar_observation(
        completed,
        granularity="M5",
        completed_lookback=3,
    )
    after_boundary = engine._completed_bar_observation(
        completed[1:] + [forming],
        granularity="M5",
        completed_lookback=3,
    )

    assert boundary["completed_bar_open_utc"] == after_boundary["completed_bar_open_utc"]
    assert boundary["completed_bar_close_utc"] == after_boundary["completed_bar_close_utc"]
    assert boundary["completed_signal"] == after_boundary["completed_signal"]
    assert boundary["completed_reason"] == after_boundary["completed_reason"]
    assert boundary["completed_diagnostics"] == after_boundary["completed_diagnostics"]


def test_configuration_fingerprint_is_canonical_and_sensitive():
    first = configuration_fingerprint({"timeframe": "M5", "risk": {"cap": 0.01}})
    reordered = configuration_fingerprint({"risk": {"cap": 0.01}, "timeframe": "M5"})
    changed = configuration_fingerprint({"timeframe": "M5", "risk": {"cap": 0.02}})

    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_completed_bar_must_be_ordered_and_not_future_dated():
    future_close = _draft(observed_at=BAR_OPEN + timedelta(minutes=4, seconds=58))
    assert future_close.finalize(
        outcome="blocked",
        gate_stage="risk",
        gate_reason="test",
    ) is None

    non_positive_interval = _draft()
    non_positive_interval.completed_bar_close_utc = BAR_OPEN
    assert non_positive_interval.finalize(
        outcome="blocked",
        gate_stage="risk",
        gate_reason="test",
    ) is None


def test_overflowing_optional_numbers_are_dropped():
    class OverflowingNumber:
        def __float__(self):
            raise OverflowError("too large")

    draft = _draft()
    draft.set_spread(OverflowingNumber())
    draft.reference_price = OverflowingNumber()
    draft.stop_distance = OverflowingNumber()
    draft.target_distance = OverflowingNumber()
    record = draft.finalize(
        outcome="blocked",
        gate_stage="risk",
        gate_reason="test",
    )

    assert record is not None
    assert record.spread_pips is None
    assert record.reference_price is None
    assert record.stop_distance is None
    assert record.target_distance is None


def test_ledger_separates_same_minute_events_from_stable_opportunity(tmp_path):
    path = tmp_path / "learning_observations.db"
    ledger = DecisionObservationLedger(path)
    first = _record(observed_at=BAR_OPEN + timedelta(minutes=6, seconds=1))
    later = _record(
        observed_at=BAR_OPEN + timedelta(minutes=6, seconds=45),
        outcome="executed",
        stage="executed",
        reason="broker-fill-verified",
    )

    assert first.opportunity_id == later.opportunity_id
    assert first.decision_event_id != later.decision_event_id
    assert ledger.append(first) is True
    assert ledger.append(first) is False
    assert ledger.append(later) is True

    with sqlite3.connect(path) as connection:
        opportunity = connection.execute(
            "SELECT COUNT(*), reference_price, stop_distance, target_distance "
            "FROM learning_opportunities"
        ).fetchone()
        events = connection.execute(
            "SELECT final_outcome, gate_stage, evaluation_tick_utc "
            "FROM runtime_decision_events ORDER BY observed_at_utc"
        ).fetchall()
        assert opportunity == (1, 0.66, 0.0012, 0.001)
        assert events == [
            ("blocked", "risk", first.evaluation_tick_utc),
            ("executed", "executed", first.evaluation_tick_utc),
        ]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE learning_opportunities SET observer_signal='SELL' WHERE opportunity_id=?",
                (first.opportunity_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM runtime_decision_events WHERE decision_event_id=?",
                (first.decision_event_id,),
            )


def test_divergent_payload_for_existing_opportunity_is_rejected(tmp_path):
    path = tmp_path / "learning_observations.db"
    ledger = DecisionObservationLedger(path)
    original = _record()
    divergent = replace(
        original,
        decision_event_id="f" * 64,
        features_json='{"close":999.0}',
    )

    assert ledger.append(original) is True
    with pytest.raises(
        sqlite3.IntegrityError,
        match="immutable learning opportunity payload conflict",
    ):
        ledger.append(divergent)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT features_json FROM learning_opportunities WHERE opportunity_id=?",
            (original.opportunity_id,),
        ).fetchone() == (original.features_json,)
        assert connection.execute(
            "SELECT COUNT(*) FROM runtime_decision_events",
        ).fetchone() == (1,)


def test_runtime_drafts_keep_path_specific_distances_out_of_stable_opportunity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main, "_runtime_revision", lambda: "abc123")
    first_draft = main._new_decision_observation_draft(
        _evaluation(),
        BAR_OPEN + timedelta(minutes=6, seconds=1),
    )
    later_draft = main._new_decision_observation_draft(
        _evaluation(),
        BAR_OPEN + timedelta(minutes=6, seconds=45),
    )
    assert first_draft is not None
    assert later_draft is not None
    first = first_draft.finalize(
        outcome="blocked",
        gate_stage="risk",
        gate_reason="daily-loss-cap",
    )
    later = later_draft.finalize(
        outcome="executed",
        gate_stage="executed",
        gate_reason="broker-fill-verified",
    )
    assert first is not None
    assert later is not None
    assert first.opportunity_id == later.opportunity_id
    assert (first.stop_distance, first.target_distance) == (None, None)
    assert (later.stop_distance, later.target_distance) == (None, None)

    ledger = DecisionObservationLedger(tmp_path / "learning_observations.db")
    assert ledger.append(first) is True
    assert ledger.append(later) is True
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*), stop_distance, target_distance "
            "FROM learning_opportunities"
        ).fetchone() == (1, None, None)
        assert connection.execute(
            "SELECT COUNT(*) FROM runtime_decision_events"
        ).fetchone() == (2,)


def test_opportunity_and_event_insert_roll_back_together(tmp_path):
    path = tmp_path / "learning_observations.db"
    ledger = DecisionObservationLedger(path)
    assert ledger.append(_record()) is True
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER force_event_failure BEFORE INSERT ON runtime_decision_events "
            "WHEN NEW.gate_reason='force-fail' BEGIN SELECT RAISE(ABORT, 'forced'); END;"
        )

    draft = _draft(config_hash="f" * 64, observed_at=BAR_OPEN + timedelta(minutes=8))
    failed = draft.finalize(
        outcome="blocked",
        gate_stage="risk",
        gate_reason="force-fail",
    )
    assert failed is not None
    with pytest.raises(sqlite3.IntegrityError):
        ledger.append(failed)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM learning_opportunities WHERE config_fingerprint=?",
            ("f" * 64,),
        ).fetchone()[0] == 0


def test_sink_queue_full_and_database_failure_are_fail_open():
    record = _record()
    held = DecisionObservationSink(
        DecisionObservationLedger("unused.db"),
        max_queue_size=1,
        start_worker=False,
    )
    assert held.counters()["pending_events"] == 0
    assert held.submit(record) is True
    assert held.submit(record) is False
    assert held.counters() == {
        "submitted": 1,
        "persisted_events": 0,
        "duplicate_events": 0,
        "queue_drops": 1,
        "write_errors": 0,
        "pending_events": 1,
    }

    attempted = threading.Event()

    class BrokenLedger:
        def append(self, observation):
            attempted.set()
            raise sqlite3.OperationalError("disk offline")

    broken = DecisionObservationSink(BrokenLedger())
    assert broken.submit(record) is True
    assert attempted.wait(1.0)
    assert broken.flush(1.0) is True
    assert broken.counters()["write_errors"] == 1
    assert broken.counters()["pending_events"] == 0


def test_shutdown_flush_is_bounded_and_fail_open(monkeypatch):
    class CapturingSink:
        def __init__(self):
            self.timeouts = []

        def flush(self, timeout_seconds):
            self.timeouts.append(timeout_seconds)
            return True

        def counters(self):
            return {"pending_events": 0}

    capturing = CapturingSink()
    monkeypatch.setattr(main, "decision_observation_sink", capturing)
    assert main._flush_decision_observations(99.0) is True
    assert capturing.timeouts == [2.0]

    class BrokenSink(CapturingSink):
        def flush(self, timeout_seconds):
            raise RuntimeError("shutdown storage failure")

    monkeypatch.setattr(main, "decision_observation_sink", BrokenSink())
    assert main._flush_decision_observations() is False


def test_hold_and_session_block_capture_terminal_stage(runtime, monkeypatch):
    broker, _, _, engine = runtime
    capture = CaptureSink()
    monkeypatch.setattr(main, "decision_observation_sink", capture)
    monkeypatch.setattr(
        main.session_filter,
        "session_decision",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            session=None,
            in_session=False,
            risk_scale=1.0,
            mode="SOFT",
            reason=None,
        ),
    )
    engine.evaluate_all.return_value = [_evaluation("HOLD", "neutral")]

    asyncio.run(main.decision_cycle())

    assert broker.place_order.call_count == 0
    assert len(capture.records) == 1
    assert capture.records[0].final_outcome == "hold"
    assert capture.records[0].gate_stage == "signal"
    assert capture.records[0].gate_reason == "neutral"
    assert capture.records[0].spread_pips is None

    capture.records.clear()
    engine.evaluate_all.return_value = [_evaluation("BUY", "bullish")]
    monkeypatch.setattr(
        main.session_filter,
        "session_decision",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=False,
            session=None,
            in_session=False,
            risk_scale=1.0,
            mode="STRICT",
            reason="strict-off-session",
        ),
    )

    asyncio.run(main.decision_cycle())

    assert broker.place_order.call_count == 0
    assert len(capture.records) == 1
    assert capture.records[0].final_outcome == "blocked"
    assert capture.records[0].gate_stage == "session"
    assert capture.records[0].gate_reason == "strict-off-session"
    assert capture.records[0].spread_pips is None


def test_uncertain_order_marks_untouched_pair_not_evaluated(runtime, monkeypatch):
    broker, risk, _, engine = runtime
    engine.evaluate_all.return_value = [
        _evaluation("BUY", "bullish", "AUD_USD"),
        _evaluation("BUY", "bullish", "GBP_USD"),
    ]
    risk.risk_per_trade_pct = 0.0025
    risk.demo_mode = False
    risk.should_open.return_value = (True, "ok")
    risk.sl_distance_from_atr.return_value = 0.001
    risk.tp_distance_from_atr.return_value = 0.002
    broker.place_order.return_value = {"status": "UNKNOWN"}
    capture = CaptureSink()
    monkeypatch.setattr(main, "decision_observation_sink", capture)
    monkeypatch.setattr(
        main.session_filter,
        "session_decision",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            session=None,
            in_session=True,
            risk_scale=1.0,
            mode="SOFT",
            reason=None,
        ),
    )
    monkeypatch.setattr(main, "_macd_confirms", lambda *args: (True, 0.1, 0.0, 0.1))
    monkeypatch.setattr(main, "_orb_filter", lambda *args: (True, None, None))
    monkeypatch.setattr(main.position_sizer, "units_for_risk", lambda *args, **kwargs: (100, {}))

    asyncio.run(main.decision_cycle())

    broker.place_order.assert_called_once()
    assert [(record.instrument, record.final_outcome, record.gate_stage, record.gate_reason)
            for record in capture.records] == [
        ("AUD_USD", "order-failed", "order", "order-unknown"),
        ("GBP_USD", "not_evaluated", "cycle", "prior-order-state-uncertain"),
    ]


@pytest.mark.parametrize(
    "failure_mode",
    [
        "none",
        "queue-error",
        "database-error",
        "set-session-error",
        "set-spread-error",
    ],
)
def test_executed_broker_call_is_unchanged_when_observer_fails(
    runtime,
    monkeypatch,
    failure_mode,
):
    broker, risk, _, engine = runtime
    engine.evaluate_all.return_value = [_evaluation("BUY", "bullish")]
    risk.risk_per_trade_pct = 0.0025
    risk.demo_mode = False
    risk.should_open.return_value = (True, "ok")
    risk.sl_distance_from_atr.return_value = 0.001
    risk.tp_distance_from_atr.return_value = 0.002
    broker.place_order.return_value = confirmed_order_result(
        instrument="AUD_USD",
        signal="BUY",
        units=100,
        price=0.66,
        trade_id="42",
    )
    monkeypatch.setattr(
        main.session_filter,
        "session_decision",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            session=None,
            in_session=True,
            risk_scale=1.0,
            mode="SOFT",
            reason=None,
        ),
    )
    monkeypatch.setattr(main, "_macd_confirms", lambda *args: (True, 0.1, 0.0, 0.1))
    monkeypatch.setattr(main, "_orb_filter", lambda *args: (True, None, None))
    monkeypatch.setattr(main.position_sizer, "units_for_risk", lambda *args, **kwargs: (100, {}))

    if failure_mode.startswith("set-"):
        method_name = failure_mode.removeprefix("set-").removesuffix("-error")
        method_name = f"set_{method_name.replace('-', '_')}"

        def fail_mutator(*args, **kwargs):
            raise RuntimeError("observer mutation unavailable")

        monkeypatch.setattr(DecisionObservationDraft, method_name, fail_mutator)

    if failure_mode in {
        "none",
        "set-session-error",
        "set-spread-error",
    }:
        sink = CaptureSink()
    elif failure_mode == "queue-error":
        sink = CaptureSink(fail=True)
    else:
        class BrokenLedger:
            def append(self, observation):
                raise sqlite3.OperationalError("disk offline")

        sink = DecisionObservationSink(BrokenLedger())
    monkeypatch.setattr(main, "decision_observation_sink", sink)

    asyncio.run(main.decision_cycle())

    broker.place_order.assert_called_once_with(
        "AUD_USD",
        "BUY",
        100,
        sl_distance=0.001,
        tp_distance=0.002,
        entry_price=0.66,
    )
    risk.register_entry.assert_called_once()
    risk.sl_distance_from_atr.assert_called_once_with(0.001, instrument="AUD_USD")
    risk.tp_distance_from_atr.assert_called_once_with(0.001, instrument="AUD_USD")
    if isinstance(sink, CaptureSink) and not sink.fail:
        assert len(sink.records) == 1
        assert sink.records[0].final_outcome == "executed"
        assert sink.records[0].gate_stage == "executed"
        assert sink.records[0].gate_reason == "broker-fill-verified"
        if failure_mode == "none":
            assert sink.records[0].stop_distance is None
            assert sink.records[0].target_distance is None
    elif failure_mode == "database-error":
        assert sink.wait_until_empty(1.0) is True
