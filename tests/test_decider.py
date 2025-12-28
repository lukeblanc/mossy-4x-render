import sys
from pathlib import Path
from typing import Dict, List

import asyncio
import httpx
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.health import watchdog

from src.decision_engine import (
    DEFAULT_INSTRUMENTS,
    PRACTICE_BASE_URL,
    DecisionEngine,
)
from src.decision_engine import Evaluation, _default_fetcher
from src import main


@pytest.fixture()
def sample_config() -> Dict:
    return {
        "instruments": ["EUR_USD", "AUD_USD", "XAU_USD"],
        "cooldown_minutes": 0,
        "risk_per_trade": 0.02,
        "account_balance": 10000,
        "candles_to_fetch": 5,
        "timeframe": "M1",
        "ema_fast": 2,
        "ema_slow": 3,
        "rsi_length": 2,
        "rsi_buy": 60,
        "rsi_sell": 40,
        "atr_length": 2,
        "min_atr": 0.0001,
    }


def test_scans_all_instruments(capfd, sample_config):
    prices: Dict[str, List[Dict[str, float]]] = {
        "EUR_USD": [
            {"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0},
            {"o": 1.0, "h": 1.2, "l": 1.0, "c": 1.2},
            {"o": 1.2, "h": 1.3, "l": 1.1, "c": 1.3},
            {"o": 1.3, "h": 1.4, "l": 1.2, "c": 1.4},
        ],
        "AUD_USD": [
            {"o": 0.75, "h": 0.76, "l": 0.74, "c": 0.75},
            {"o": 0.75, "h": 0.75, "l": 0.73, "c": 0.74},
            {"o": 0.74, "h": 0.74, "l": 0.72, "c": 0.73},
            {"o": 0.73, "h": 0.73, "l": 0.71, "c": 0.72},
        ],
        "XAU_USD": [
            {"o": 1950.0, "h": 1951.0, "l": 1949.0, "c": 1950.5},
            {"o": 1950.5, "h": 1951.5, "l": 1949.5, "c": 1950.5},
        ],
    }

    def fetcher(instrument: str, **kwargs):
        return prices[instrument]

    engine = DecisionEngine(sample_config, candle_fetcher=fetcher, now_fn=lambda: datetime.now(timezone.utc))
    evaluations = engine.evaluate_all()

    assert [ev.instrument for ev in evaluations] == sample_config["instruments"]
    signals = {ev.instrument: ev.signal for ev in evaluations}
    assert signals["EUR_USD"] == "BUY"
    assert signals["AUD_USD"] == "SELL"
    assert signals["XAU_USD"] == "HOLD"

    captured = capfd.readouterr()
    signal_lines = [line for line in captured.out.splitlines() if line.startswith("[SIGNAL]")]
    assert any("[SIGNAL] EUR_USD signal=BUY" in line for line in signal_lines)
    assert any("[SIGNAL] AUD_USD signal=SELL" in line for line in signal_lines)
    assert any("[SIGNAL] XAU_USD signal=HOLD" in line for line in signal_lines)


def test_skips_inactive_markets(capfd, sample_config):
    def fetcher(instrument: str, **kwargs):
        return [{"o": 1.0, "h": 1.0, "l": 1.0, "c": None}]

    engine = DecisionEngine(sample_config, candle_fetcher=fetcher, now_fn=lambda: datetime.now(timezone.utc))
    evaluations = engine.evaluate_all()

    assert all(not ev.market_active for ev in evaluations)
    assert all(ev.signal == "HOLD" for ev in evaluations)

    captured = capfd.readouterr()
    signal_lines = [line for line in captured.out.splitlines() if line.startswith("[SIGNAL]")]
    assert len(signal_lines) == len(sample_config["instruments"])
    assert all("signal=HOLD" in line for line in signal_lines)
    assert all("rsi=n/a" in line for line in signal_lines)
    assert all("atr=n/a" in line for line in signal_lines)


def test_fetches_each_instrument_individually(capfd, sample_config):
    instruments = ["EUR_USD", "AUD_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
    sample_config = {**sample_config, "instruments": instruments}

    requested: List[str] = []

    def fetcher(instrument: str, **kwargs):
        requested.append(instrument)
        assert "," not in instrument
        return [
            {"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0},
            {"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0},
        ]

    engine = DecisionEngine(sample_config, candle_fetcher=fetcher, now_fn=lambda: datetime.now(timezone.utc))
    evaluations = engine.evaluate_all()

    assert len(evaluations) == len(instruments)
    assert requested == instruments

    captured = capfd.readouterr()
    out_lines = captured.out.splitlines()
    fetch_logs = [line for line in out_lines if line.startswith("[SCAN]")]
    assert len(fetch_logs) == len(instruments)
    for instrument in instruments:
        assert any(
            line.startswith(f"[SCAN] {instrument} OK (") for line in fetch_logs
        ), f"missing scan OK log for {instrument}"
    assert "400 Bad Request" not in captured.out
    assert "✅ Multi-Pair Candle Fetch Verified" in captured.out


def test_decision_cycle_updates_watchdog_on_success(monkeypatch):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def __init__(self) -> None:
            self.entries: List[datetime] = []

        def enforce_equity_floor(self, *args, **kwargs) -> None:
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr):
            if atr is None:
                return 0.0
            return atr * 1.5

        def tp_distance_from_atr(self, atr):
            if atr is None:
                return 0.0
            return atr * 3.0

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, realized_pl: float) -> None:
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []
            self.evaluations: int = 0

        def evaluate_all(self) -> List[Evaluation]:
            self.evaluations += 1
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={"atr": 0.01, "close": 1.2345},
                    reason="trend",
                    market_active=True,
                )
            ]

        def position_size(self, instrument: str, diagnostics: Dict) -> int:
            return 1

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
            *,
            sl_distance: float | None = None,
            tp_distance: float | None = None,
            entry_price: float | None = None,
        ) -> Dict[str, str]:
            self.calls.append(
                {
                    "instrument": instrument,
                    "signal": signal,
                    "units": units,
                    "sl_distance": sl_distance,
                    "tp_distance": tp_distance,
                    "entry_price": entry_price,
                }
            )
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    class Monday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 2, 5, 12, 0, tzinfo=timezone.utc)

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "datetime", Monday)
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )

    before = Monday.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        assert dummy_engine.marked == ["EUR_USD"]
        expected_sl = dummy_risk.sl_distance_from_atr(0.01)
        assert dummy_broker.calls == [
            {
                "instrument": "EUR_USD",
                "signal": "BUY",
                "units": 100,
                "sl_distance": expected_sl,
                "tp_distance": dummy_risk.tp_distance_from_atr(0.01),
                "entry_price": 1.2345,
            }
        ]
        assert dummy_risk.entries
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        monkeypatch.setattr(main, "datetime", datetime)


def test_decision_cycle_updates_watchdog_on_error(monkeypatch):
    class FailingEngine:
        def evaluate_all(self) -> List[Evaluation]:
            raise RuntimeError("boom")

    events: Dict[str, bool] = {"error": False}

    def record_error() -> None:
        events["error"] = True

    class DummyRisk:
        risk_per_trade_pct = 0.001

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr):
            return 0.01

        def register_entry(self, *args, **kwargs):
            pass

        def register_exit(self, *args, **kwargs):
            pass

    failing_engine = FailingEngine()
    monkeypatch.setattr(main, "engine", failing_engine)
    monkeypatch.setattr(main.watchdog, "record_error", record_error)
    monkeypatch.setattr(main, "risk", DummyRisk())

    before = datetime.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        assert events["error"] is True
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts


def test_decision_cycle_blocks_entries_outside_session(monkeypatch, capsys):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def __init__(self) -> None:
            self.entries: List[datetime] = []

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr):
            return 0.01

        def tp_distance_from_atr(self, atr):
            return 0.02

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []
            self.evaluations: int = 0

        def evaluate_all(self) -> List[Evaluation]:
            self.evaluations += 1
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={"atr": 0.01, "close": 1.2345},
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

    class Monday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 2, 5, 12, 0, tzinfo=timezone.utc)

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    calls = {"should_open": 0}
    dummy_risk.demo_mode = True
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(
        dummy_risk,
        "should_open",
        lambda *args, **kwargs: calls.__setitem__("should_open", calls["should_open"] + 1) or (True, "ok"),
    )
    monkeypatch.setattr(main, "datetime", Monday)
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: False)
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )

    before = Monday.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        captured = capsys.readouterr().out
        assert "[SESSION] Entry blocked – outside trading session" in captured
        assert dummy_engine.evaluations == 1
        assert dummy_engine.marked == []
        assert dummy_broker.calls == []
        assert dummy_risk.entries == []
        assert calls["should_open"] == 1
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        monkeypatch.setattr(main, "datetime", datetime)


def test_decision_cycle_blocks_entries_on_weekend(monkeypatch, capsys):
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
            return 0.01

        def tp_distance_from_atr(self, atr):
            return 0.02

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []
            self.evaluations: int = 0

        def evaluate_all(self) -> List[Evaluation]:
            self.evaluations += 1
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={"atr": 0.01, "close": 1.2345},
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

    class Saturday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 2, 3, 12, 0, tzinfo=timezone.utc)

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    calls = {"should_open": 0}
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main, "datetime", Saturday)
    monkeypatch.setattr(
        dummy_risk,
        "should_open",
        lambda *args, **kwargs: calls.__setitem__("should_open", calls["should_open"] + 1) or (True, "ok"),
    )
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )

    before = Saturday.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        captured = capsys.readouterr().out
        assert "[WEEKEND] Entry blocked - weekend lock active (UTC Saturday/Sunday)" in captured
        assert dummy_engine.evaluations == 1
        assert dummy_engine.marked == []
        assert dummy_broker.calls == []
        assert dummy_risk.entries == []
        assert calls["should_open"] == 1
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        monkeypatch.setattr(main, "datetime", datetime)


def test_decision_cycle_allows_entries_inside_session(monkeypatch):
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
            return 0.01

        def tp_distance_from_atr(self, atr):
            return 0.02

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []
            self.evaluations: int = 0

        def evaluate_all(self) -> List[Evaluation]:
            self.evaluations += 1
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={"atr": 0.01, "close": 1.2345},
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

    class Monday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 2, 5, 12, 0, tzinfo=timezone.utc)

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "datetime", Monday)
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )

    before = Monday.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        assert dummy_engine.evaluations == 1
        assert dummy_engine.marked == ["EUR_USD"]
        assert dummy_broker.calls == [
            {"instrument": "EUR_USD", "signal": "BUY", "units": 100}
        ]
        assert dummy_risk.entries
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        monkeypatch.setattr(main, "datetime", datetime)


def test_live_mode_ignores_weekend_lock(monkeypatch, capsys):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def __init__(self) -> None:
            self.entries: List[datetime] = []
            self.demo_mode = False

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr):
            return 0.01

        def tp_distance_from_atr(self, atr):
            return 0.02

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(now_utc)

        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []
            self.evaluations: int = 0

        def evaluate_all(self) -> List[Evaluation]:
            self.evaluations += 1
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={"atr": 0.01, "close": 1.2345},
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

    class Sunday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 2, 4, 12, 0, tzinfo=timezone.utc)

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "mode_env", "live")
    monkeypatch.setattr(main, "datetime", Sunday)
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )

    before = Sunday.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        captured = capsys.readouterr().out
        assert "[WEEKEND] Entry blocked" not in captured
        assert dummy_engine.evaluations == 1
        assert dummy_engine.marked == ["EUR_USD"]
        assert dummy_broker.calls == [
            {"instrument": "EUR_USD", "signal": "BUY", "units": 100}
        ]
        assert dummy_risk.entries
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
        monkeypatch.setattr(main, "datetime", datetime)


def test_default_fetcher_uses_instrument_path(monkeypatch):
    requests: List[Dict[str, object]] = []

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            self.base_url = kwargs.get("base_url")
            self.headers = dict(kwargs.get("headers") or {})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path: str, params: Dict[str, str]):
            url = f"{self.base_url}{path}"
            requests.append({"url": url, "params": params, "headers": dict(self.headers)})
            request = httpx.Request("GET", url, headers=self.headers)
            return httpx.Response(
                200,
                request=request,
                json={
                    "candles": [
                        {"mid": {"o": "1.0", "h": "1.0", "l": "1.0", "c": "1.0"}}
                    ]
                },
            )

    monkeypatch.setattr(httpx, "Client", DummyClient)

    candles = _default_fetcher("EUR_USD", count=10, granularity="M1", api_key="token")

    assert len(candles) == 1
    assert requests
    request_info = requests[0]
    assert request_info["url"] == f"{PRACTICE_BASE_URL}/instruments/EUR_USD/candles"
    assert request_info["params"] == {"count": "10", "granularity": "M1", "price": "M"}
    assert request_info["headers"].get("Authorization") == "Bearer token"


def test_fetch_candles_retries_on_transient_error(capfd, sample_config):
    attempts = {"count": 0}

    def fetcher(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            response = httpx.Response(
                500,
                request=httpx.Request("GET", "https://retry.test"),
            )
            raise httpx.HTTPStatusError("server error", request=response.request, response=response)
        return [
            {"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0},
        ]

    config = {
        **sample_config,
        "fetch_retry_attempts": 3,
        "fetch_retry_backoff": 0,
    }

    engine = DecisionEngine(config, candle_fetcher=lambda instrument, **kw: fetcher(**kw))
    candles = engine._fetch_candles("EUR_USD", candle_count=2, granularity="M1")

    assert attempts["count"] == 3
    assert len(candles) == 1

    captured = capfd.readouterr().out
    warn_lines = [line for line in captured.splitlines() if line.startswith("[WARN]")]
    assert any("attempt 1 failed 500" in line for line in warn_lines)
    assert "[SCAN] EUR_USD OK" in captured


def test_fetch_candles_does_not_retry_on_client_error(capfd, sample_config):
    attempts = {"count": 0}

    def fetcher(**kwargs):
        attempts["count"] += 1
        response = httpx.Response(
            400,
            request=httpx.Request("GET", "https://client-error.test"),
        )
        raise httpx.HTTPStatusError("bad request", request=response.request, response=response)

    config = {
        **sample_config,
        "fetch_retry_attempts": 3,
        "fetch_retry_backoff": 0,
    }

    engine = DecisionEngine(config, candle_fetcher=lambda instrument, **kw: fetcher(**kw))
    candles = engine._fetch_candles("EUR_USD", candle_count=2, granularity="M1")

    assert attempts["count"] == 1
    assert candles == []

    captured = capfd.readouterr().out
    assert "[WARN] EUR_USD fetch failed 400 – skipping" in captured


def test_resolve_instruments_normalizes_input(sample_config):
    config = {
        **sample_config,
        "instruments": " eur_usd, gbp_usd usd_jpy  eur_usd ",
        "merge_default_instruments": False,
    }

    engine = DecisionEngine(config, candle_fetcher=lambda *args, **kwargs: [])

    assert engine.instruments == ["EUR_USD", "GBP_USD", "USD_JPY"]


def test_resolve_instruments_from_set():
    config = {"instruments": {"eur_usd", "aud_usd"}}
    engine = DecisionEngine(config)
    assert engine.instruments == ["EUR_USD", "AUD_USD"]


def test_resolve_instruments_with_whitespace_and_duplicates():
    config = {"instruments": [" eur_usd ", "", "GBP_USD", "eur_usd"]}
    engine = DecisionEngine(config)
    assert engine.instruments == ["EUR_USD", "GBP_USD"]


def test_missing_instruments_defaults_when_merge_enabled():
    config = {"merge_default_instruments": True}
    engine = DecisionEngine(config)
    assert engine.instruments == DEFAULT_INSTRUMENTS


def test_empty_instruments_defaults_when_merge_enabled():
    config = {"instruments": [], "merge_default_instruments": True}
    engine = DecisionEngine(config)
    assert engine.instruments == DEFAULT_INSTRUMENTS


def test_empty_instruments_without_merge_returns_empty():
    config = {"instruments": [], "merge_default_instruments": False}
    engine = DecisionEngine(config)
    assert engine.instruments == []
