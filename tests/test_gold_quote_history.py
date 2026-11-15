import json
from datetime import datetime, timezone, timedelta

import httpx
import pytest

from src import gold_quote_history as quotes


def payload(now=None, **changes):
    now = now or datetime.now(timezone.utc).isoformat()
    price = {"instrument": "XAU_USD", "time": now, "tradeable": True,
             "bids": [{"price": "2500.10"}], "asks": [{"price": "2500.30"}]}
    price.update(changes)
    return {"prices": [price]}


class StopAfter:
    def __init__(self, count):
        self.count, self.waits = count, []

    def is_set(self):
        return len(self.waits) >= self.count

    def wait(self, seconds):
        self.waits.append(seconds)


def test_records_fresh_quotes_deduplicates_and_reports_coverage(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    now = datetime.now(timezone.utc).isoformat()
    assert quotes.record(connection, payload(now), now, ["XAU_USD"])["new_rows"] == 1
    assert quotes.record(connection, payload(now), now, ["XAU_USD"])["new_rows"] == 0
    assert quotes.coverage(connection)["XAU_USD"]["rows"] == 1
    assert quotes.coverage(connection)["AUD_USD"]["rows"] == 0


def test_off_session_is_successful_response_without_executable_quotes(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    data = payload("2026-11-13T22:00:00Z", tradeable=False, status="non-tradeable")
    result = quotes.record(connection, data, "2026-11-15T00:00:00Z", ["XAU_USD"])
    assert result["state"] == "market_closed"
    assert quotes.coverage(connection)["XAU_USD"]["rows"] == 0


def test_legacy_status_and_missing_fields_are_distinguished(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    now = datetime.now(timezone.utc).isoformat()
    data = payload(now, status="tradeable")
    del data["prices"][0]["tradeable"]
    assert quotes.record(connection, data, now, ["XAU_USD"])["state"] == "fresh"
    del data["prices"][0]["status"]
    with pytest.raises(ValueError, match="tradability"):
        quotes.record(connection, data, now)


@pytest.mark.parametrize("change", [
    {"bids": [{"price": "nan"}]}, {"asks": [{"price": "2499"}]},
    {"time": "2999-01-01T00:00:00Z"}, {"tradeable": "true"},
])
def test_malformed_quotes_never_create_rows(tmp_path, change):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    with pytest.raises(ValueError):
        quotes.record(connection, payload(**change), datetime.now(timezone.utc).isoformat())
    assert quotes.coverage(connection)["XAU_USD"]["rows"] == 0


def test_stale_and_missing_quotes_are_not_fresh(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    result = quotes.record(connection, payload("2026-11-14T00:00:00Z"), "2026-11-14T00:01:00Z", ["XAU_USD"])
    assert result["state"] == "stale"
    assert result["new_rows"] == 0
    assert quotes.record(connection, {"prices": []}, "2026-11-14T00:01:00Z")["state"] == "incomplete"


def test_malformed_snapshot_cannot_partially_commit(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    now = datetime.now(timezone.utc).isoformat()
    data = payload(now)
    data["prices"].append(payload(now, instrument="AUD_USD", bids=[{"price": "nan"}])["prices"][0])
    with pytest.raises(ValueError):
        quotes.record(connection, data, now)
    assert quotes.coverage(connection)["XAU_USD"]["rows"] == 0


def test_replay_excludes_legacy_stale_quotes(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    connection.execute("INSERT INTO practice_quotes VALUES (?, ?, ?, ?, ?)",
        ("XAU_USD", "2026-11-14T00:00:02+00:00", "2026-11-14T00:02:00+00:00", 2501, 2502))
    result = quotes.delayed_entry(connection, side="BUY", fill=2500, costs=0,
        opened="2026-11-14T00:00:01Z", closed="2026-11-14T00:03:00Z", target=2505)
    assert result["status"] == "no_quotes"


@pytest.mark.parametrize("failure", ["timeout", "read_error", 500, 429, "malformed"])
def test_collector_recovers_and_sanitizes_failures(tmp_path, capsys, failure):
    requests = []
    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.host == "api-fxpractice.oanda.com"
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("secret-token private-account", request=request)
            if failure == "read_error":
                raise httpx.ReadError("secret-token private-account", request=request)
            if failure == "malformed":
                return httpx.Response(200, text="secret-token private-account")
            return httpx.Response(failure, headers={"retry-after": "20"}, text="secret-token private-account")
        return httpx.Response(200, json=payload())
    stop = StopAfter(2)
    path = tmp_path / "status.json"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        quotes.collect(quotes.database(tmp_path / "quotes.sqlite"), 5, ["XAU_USD"],
                       account="private-account", key="secret-token", client=client,
                       stop_event=stop, status_path=path)
    assert len(requests) == 2
    assert stop.waits[0] >= 10 and stop.waits[1] == 5
    result = json.loads(path.read_text())
    assert result["state"] == "fresh"
    assert result["consecutive_failures"] == 0
    assert result["coverage"]["XAU_USD"]["rows"] == 1
    assert path.stat().st_mode & 0o777 == 0o600
    output = capsys.readouterr().out
    assert "secret-token" not in output and "private-account" not in output


def test_redirect_never_sends_credentials_to_another_host(tmp_path):
    requested = []
    def respond(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://different.invalid/"})
    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        quotes.collect(quotes.database(tmp_path / "quotes.sqlite"), 5, ["XAU_USD"],
                       account="test", key="test", client=client, stop_event=StopAfter(1))
    assert len(requested) == 1
    assert requested[0].startswith(quotes.PRACTICE_HOST)


def test_auth_failure_backs_off_without_changing_credentials(tmp_path):
    stop = StopAfter(1)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401))) as client:
        quotes.collect(quotes.database(tmp_path / "quotes.sqlite"), 5, ["XAU_USD"],
                       account="test", key="test", client=client, stop_event=stop)
    assert stop.waits == [300]


def test_worker_collection_requires_practice_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("OANDA_ENV", "practice")
    with pytest.raises(ValueError, match="practice-demo"):
        quotes.collect_for_worker(tmp_path / "never-created", "test", "test")
    assert not (tmp_path / "never-created").exists()


def test_replay_never_uses_quote_received_after_close(tmp_path):
    connection = quotes.database(tmp_path / "quotes.sqlite")
    quotes.record(connection, payload("2026-11-14T00:00:02Z"), "2026-11-14T00:00:03Z", ["XAU_USD"])
    result = quotes.delayed_entry(connection, side="BUY", fill=2499, costs=0,
                                 opened="2026-11-14T00:00:01Z", closed="2026-11-14T00:00:03Z", target=2502)
    assert result["status"] == "no_quotes"
