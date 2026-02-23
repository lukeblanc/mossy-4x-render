from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime, timezone

import src.session_filter as session_filter


def _reload_main(monkeypatch, **env):
    for key in [
        "AGGRESSIVE_TEST_MODE",
        "SESSION_MODE",
        "AGGRESSIVE_RISK_PCT",
        "AGG_MAX_TOTAL_OPEN_RISK",
        "DAILY_MAX_DRAWDOWN",
        "WEEKLY_MAX_DRAWDOWN",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    fake_waitress = types.ModuleType("waitress")
    fake_waitress.serve = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "waitress", fake_waitress)

    sys.modules.pop("src.main", None)
    import src.main as main_mod

    return importlib.reload(main_mod)


def test_beast_mode_forces_always_session(monkeypatch):
    main_mod = _reload_main(
        monkeypatch,
        AGGRESSIVE_TEST_MODE="true",
        AGGRESSIVE_RISK_PCT="0.004",
        AGG_MAX_TOTAL_OPEN_RISK="0.02",
        DAILY_MAX_DRAWDOWN="0.03",
        WEEKLY_MAX_DRAWDOWN="0.05",
    )

    assert main_mod.config["aggressive_test_mode"] is True
    assert main_mod.config["session_mode"] == "ALWAYS"
    assert main_mod.config["risk"]["risk_per_trade_pct"] == 0.004

    decision = session_filter.session_decision(
        datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc),
        mode=main_mod.config["session_mode"],
        atr=1.0,
        atr_baseline=1.0,
        trend_aligned=False,
    )
    assert decision.allowed is True


def test_normal_session_gating_unchanged_without_beast_mode(monkeypatch):
    main_mod = _reload_main(monkeypatch, SESSION_MODE="STRICT", AGGRESSIVE_TEST_MODE="false")

    assert main_mod.config["aggressive_test_mode"] is False
    assert main_mod.config["session_mode"] == "STRICT"

    decision = session_filter.session_decision(
        datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc),
        mode=main_mod.config["session_mode"],
    )
    assert decision.allowed is False


def test_status_server_uses_waitress(monkeypatch):
    main_mod = _reload_main(monkeypatch)

    called = {}

    def fake_serve(app, host, port):
        called["app"] = app
        called["host"] = host
        called["port"] = port

    monkeypatch.setenv("PORT", "12345")
    monkeypatch.setattr(main_mod, "serve", fake_serve)

    main_mod.start_status_server()

    assert called["host"] == "0.0.0.0"
    assert called["port"] == 12345
    assert hasattr(called["app"], "route")
