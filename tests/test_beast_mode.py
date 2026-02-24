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


def test_beast_mode_applies_risk_overrides_only(monkeypatch, capsys):
    main_mod = _reload_main(monkeypatch, AGGRESSIVE_TEST_MODE="true")
    captured = capsys.readouterr()

    assert main_mod.config["aggressive_test_mode"] is True
    assert main_mod.config["risk"]["risk_per_trade_pct"] == 0.025
    assert main_mod.config["risk"]["daily_profit_target_usd"] == 0.0
    assert main_mod.config["risk"]["max_trades_per_day"] == 100
    assert "[CONFIG] AGGRESSIVE_TEST_MODE=True" in captured.out
    assert "[CONFIG] Daily profit cap DISABLED (aggressive demo mode)" in captured.out
    assert "[CONFIG] Risk per trade set to 2.5%." in captured.out
    assert "[CONFIG] Max trades per day set to 100 (aggressive demo mode)" in captured.out

    # AGGRESSIVE_TEST_MODE should not force session behavior; only risk knobs change.
    assert main_mod.config["session_mode"] == "SOFT"


def test_normal_session_gating_unchanged_without_beast_mode(monkeypatch, capsys):
    main_mod = _reload_main(monkeypatch, SESSION_MODE="STRICT", AGGRESSIVE_TEST_MODE="false")
    captured = capsys.readouterr()

    assert main_mod.config["aggressive_test_mode"] is False
    assert main_mod.config["session_mode"] == "STRICT"
    assert "[CONFIG] AGGRESSIVE_TEST_MODE=False" in captured.out
    assert "Daily profit cap DISABLED" not in captured.out

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
