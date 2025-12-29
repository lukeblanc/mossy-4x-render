from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import asyncio

from src import main
from src.decision_engine import Evaluation


class Monday(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2024, 2, 5, 13, 0, tzinfo=timezone.utc)


class EarlyMonday(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2024, 2, 5, 2, 0, tzinfo=timezone.utc)


def _reset_clock(original):
    main.datetime = original  # type: ignore[assignment]


def test_xau_falling_knife_block(monkeypatch, capsys):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def __init__(self) -> None:
            self.entries: List[datetime] = []

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr):
            return 0.5

        def tp_distance_from_atr(self, atr):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="XAU_USD",
                    signal="SELL",
                    diagnostics={
                        "atr": 1.2,
                        "atr_baseline_50": 1.0,
                        "rsi": 15.0,
                        "close": 1900.0,
                        "ema_trend_fast": 1890.0,
                        "ema_trend_slow": 1900.0,
                    },
                    reason="bearish",
                    market_active=True,
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(
            self,
            instrument: str,
            signal: str,
            units: int,
            *args,
            **kwargs,
        ) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )
    original_datetime = main.datetime
    monkeypatch.setattr(main, "datetime", Monday)

    before = Monday.now(timezone.utc) - timedelta(hours=1)
    from app.health import watchdog

    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        captured = capsys.readouterr().out
        assert "[FILTER] XAU_USD blocked SELL" in captured
        assert dummy_engine.marked == []
        assert dummy_broker.calls == []
        assert dummy_risk.entries == []
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        _reset_clock(original_datetime)


def test_off_session_blocks_entries_but_trailing_runs(monkeypatch, capsys):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def __init__(self) -> None:
            self.entries: List[datetime] = []
            self.demo_mode = True

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr):
            return 0.5

        def tp_distance_from_atr(self, atr):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={
                        "atr": 0.01,
                        "atr_baseline_50": 0.01,
                        "rsi": 55.0,
                        "close": 1.2345,
                        "ema_trend_fast": 1.25,
                        "ema_trend_slow": 1.2,
                    },
                    reason="trend",
                    market_active=True,
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(
            self,
            instrument: str,
            signal: str,
            units: int,
            *args,
            **kwargs,
        ) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    class DummyGuard:
        def __init__(self) -> None:
            self.calls: List[List[Dict[str, object]]] = []

        def process_open_trades(self, trades):
            self.calls.append(list(trades))
            return []

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    dummy_guard = DummyGuard()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", dummy_guard)
    monkeypatch.setattr(main, "_open_trades_state", lambda: [{"instrument": "EUR_USD"}])
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )
    original_datetime = main.datetime
    monkeypatch.setattr(main, "datetime", EarlyMonday)

    before = EarlyMonday.now(timezone.utc) - timedelta(hours=1)
    from app.health import watchdog

    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        captured = capsys.readouterr().out
        assert captured.count("[FILTER] Entries paused (off-session)") == 1
        assert dummy_guard.calls  # trailing executed
        assert dummy_engine.marked == []
        assert dummy_broker.calls == []
        assert dummy_risk.entries == []
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        _reset_clock(original_datetime)
