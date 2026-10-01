from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import src.main as main_mod


def test_cycle_stale_detection_with_controlled_time():
    tracker = main_mod.CycleHealthTracker(gap_warn_seconds=90, summary_interval_seconds=900)
    t0 = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
    tracker.record_cycle_complete(0.42, now_utc=t0)

    assert tracker.is_cycle_stale(t0 + timedelta(seconds=89)) is False
    assert tracker.is_cycle_stale(t0 + timedelta(seconds=91)) is True


def test_cycle_percentiles_and_summary_interval_with_controlled_time():
    tracker = main_mod.CycleHealthTracker(gap_warn_seconds=90, summary_interval_seconds=900)
    t0 = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)

    tracker.record_cycle_complete(0.20, now_utc=t0)
    tracker.record_cycle_complete(0.50, now_utc=t0 + timedelta(minutes=1))
    tracker.record_cycle_complete(1.50, now_utc=t0 + timedelta(minutes=2))

    p50, p95 = tracker.duration_percentiles()
    assert p50 == 0.50
    assert p95 > 1.0

    assert tracker.should_emit_summary(t0) is True
    assert tracker.should_emit_summary(t0 + timedelta(minutes=10)) is False
    assert tracker.should_emit_summary(t0 + timedelta(minutes=16)) is True


def test_health_status_uses_cached_open_trades_snapshot(monkeypatch):
    class DummyBroker:
        def __init__(self) -> None:
            self.calls = 0

        def list_open_trades(self):
            self.calls += 1
            return [{"instrument": "EUR_USD"}]

    broker = DummyBroker()
    monkeypatch.setattr(main_mod, "broker", broker)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_SNAPSHOT", [])

    first = main_mod._health_status()
    second = main_mod._health_status()

    assert first["open_trades_count"] == 1
    assert second["open_trades_count"] == 1
    assert first["broker_entry_halted"] is False
    assert first["broker_entry_halt_reason"] is None
    assert broker.calls == 1


@pytest.mark.parametrize("reason", [None, "protection-unconfirmed", "account-currency-mismatch"])
@pytest.mark.parametrize("open_trades", [[], None])
def test_health_status_reports_halt_separately_from_scheduler_and_positions(
    monkeypatch, reason, open_trades
):
    broker = SimpleNamespace(
        entry_halt_reason=reason,
        list_open_trades=lambda: open_trades,
    )
    monkeypatch.setattr(main_mod, "broker", broker)
    monkeypatch.setattr(main_mod, "_SCHEDULER_REF", SimpleNamespace(running=True))
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_BROKER_SYNC_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_SNAPSHOT", [])

    health = main_mod._health_status()

    assert health["scheduler_alive"] is True
    assert health["broker_entry_halted"] is (reason is not None)
    assert health["broker_entry_halt_reason"] == reason
    assert health["open_trades_count"] == (0 if open_trades == [] else None)
    assert broker.entry_halt_reason == reason


def test_health_status_reads_halt_after_broker_snapshot(monkeypatch):
    class BrokerLatchingOnRead:
        entry_halt_reason = None

        def list_open_trades(self):
            self.entry_halt_reason = "protection-unconfirmed"
            return []

    broker = BrokerLatchingOnRead()
    monkeypatch.setattr(main_mod, "broker", broker)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_BROKER_SYNC_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_SNAPSHOT", [])

    health = main_mod._health_status()

    assert health["broker_entry_halted"] is True
    assert health["broker_entry_halt_reason"] == "protection-unconfirmed"


def test_status_endpoint_reports_current_halt_even_with_cached_positions(monkeypatch):
    broker = SimpleNamespace(entry_halt_reason="protection-unconfirmed")
    monkeypatch.setattr(main_mod, "broker", broker)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_TS", main_mod._utc_now())
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_SNAPSHOT", [])
    monkeypatch.setattr(main_mod, "BOT_STATE", {
        "status": "running",
        "broker_entry_halted": False,
        "broker_entry_halt_reason": None,
    })
    captured = {}

    def fake_serve(app, **kwargs):
        captured["app"] = app

    monkeypatch.setattr(main_mod, "serve", fake_serve)
    main_mod.start_status_server()
    client = captured["app"].test_client()

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "running"
    assert payload["open_trades_count"] == 0
    assert payload["broker_entry_halted"] is True
    assert payload["broker_entry_halt_reason"] == "protection-unconfirmed"

    broker.entry_halt_reason = None
    payload = client.get("/status").get_json()
    assert payload["broker_entry_halted"] is False
    assert payload["broker_entry_halt_reason"] is None


def test_heartbeat_includes_broker_halt_in_state_and_health_log(
    monkeypatch, tmp_path, capsys
):
    reason = "protection-unconfirmed"
    broker = SimpleNamespace(
        entry_halt_reason=reason,
        account_equity=lambda: 10000.0,
        list_open_trades=lambda: [],
    )
    monkeypatch.setattr(main_mod, "broker", broker)
    monkeypatch.setattr(main_mod, "BOT_STATE", {})
    monkeypatch.setattr(main_mod, "_SCHEDULER_REF", SimpleNamespace(running=True))
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_BROKER_SYNC_TS", None)
    monkeypatch.setattr(main_mod, "_LAST_OPEN_TRADES_SNAPSHOT", [])
    monkeypatch.setattr(main_mod, "_safe_adaptive_snapshot", lambda context: None)
    monkeypatch.setattr(main_mod, "_runtime_revision", lambda: "test-revision")
    monkeypatch.setattr(main_mod, "journal", SimpleNamespace(
        path=tmp_path / "journal.sqlite",
        count_trade_events=lambda: 0,
        latest_verified_entry_timestamp=lambda: None,
    ))
    monkeypatch.delenv("MOSSY_MCP_STATUS_KEY", raising=False)

    published = []

    async def fake_publish(payload, **kwargs):
        published.append(payload)
        return False, "disabled"

    monkeypatch.setattr(main_mod, "publish_runtime_heartbeat", fake_publish)

    asyncio.run(main_mod.heartbeat())

    assert main_mod.BOT_STATE["status"] == "running"
    assert main_mod.BOT_STATE["scheduler_alive"] is True
    assert main_mod.BOT_STATE["broker_entry_halted"] is True
    assert main_mod.BOT_STATE["broker_entry_halt_reason"] == reason
    assert published[0]["broker_entry_halted"] is True
    assert published[0]["broker_entry_halt_reason"] == "other"
    health_line = next(
        line for line in capsys.readouterr().out.splitlines()
        if "HEALTH_STATUS" in line
    )
    assert "broker_entry_halted=True" in health_line
    assert f"broker_entry_halt_reason={reason}" in health_line
