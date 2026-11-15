import json

from src.research_registry import publish_registry, CATALOGUE


def test_catalogue_is_copied_as_proposals_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("OANDA_ENV", "practice")
    report = publish_registry(tmp_path)
    path = tmp_path / "memory/research/registry.json"
    assert json.loads(path.read_text()) == report
    assert path.stat().st_mode & 0o777 == 0o600
    assert len(report["catalogue"]["experiments"]) == 6
    assert not report["execution_enabled"] and not report["promotion_enabled"]
    assert all(e["results"] is None for e in report["catalogue"]["experiments"])
    assert not (tmp_path / "trade_journal.db").exists()


def test_invalid_proposal_cannot_overwrite_existing_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("OANDA_ENV", "practice")
    publish_registry(tmp_path)
    path = tmp_path / "memory/research/registry.json"
    before = path.read_bytes()
    data = json.loads(CATALOGUE.read_text())
    data["experiments"][0]["status"] = "VALIDATED"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    assert publish_registry(tmp_path, bad) is None
    assert path.read_bytes() == before


def test_live_mode_cannot_import(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("OANDA_ENV", "practice")
    assert publish_registry(tmp_path) is None
    assert not (tmp_path / "memory").exists()
