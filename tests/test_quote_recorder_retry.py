"""Offline regressions for transient practice pricing response failures."""
from contextlib import contextmanager
import io
import json
import sqlite3
from urllib.parse import urlsplit

import pytest

from src import gold_quote_history as recorder


class StopCollection(Exception):
    """End the otherwise unbounded collector after its scheduled waits."""


def quote_payload():
    return {"prices": [{
        "instrument": "XAU_USD", "tradeable": True,
        "time": "2026-11-14T10:00:00+00:00",
        "bids": [{"price": "3000.0"}], "asks": [{"price": "3000.5"}],
    }]}


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("timeout_stage", ["open", "body"])
def test_timeouts_retry_at_normal_interval_and_record_next_quote(
    tmp_path, monkeypatch, capsys, worker, timeout_stage
):
    path = tmp_path / "quotes.sqlite"
    requests, waits, responses = [], [], []
    secret = "private-account-and-token-must-not-appear"

    class TimedOutBody(io.BytesIO):
        def read(self, *args, **kwargs):
            raise TimeoutError(secret)

    @contextmanager
    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 10
        assert request.get_method() == "GET"
        assert urlsplit(request.full_url).hostname == "api-fxpractice.oanda.com"
        assert urlsplit(request.full_url).path.endswith("/pricing")
        if len(requests) <= 2 and timeout_stage == "open":
            raise TimeoutError(secret)
        response = (TimedOutBody() if len(requests) <= 2
                    else io.BytesIO(json.dumps(quote_payload()).encode()))
        responses.append(response)
        with response:
            yield response

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            raise StopCollection

    monkeypatch.setattr(recorder, "urlopen", fake_urlopen)
    monkeypatch.setattr(recorder.time, "sleep", sleep)
    with pytest.raises(StopCollection):
        if worker:
            recorder.collect_for_worker(path, secret, secret)
        else:
            with recorder.database(path) as connection:
                recorder.collect(connection, 5, recorder.DEFAULT_INSTRUMENTS,
                                 account=secret, key=secret)

    assert len(requests) == 3
    assert waits == [5, 5, 5]
    assert all(response.closed for response in responses)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT instrument, bid, ask FROM practice_quotes"
        ).fetchall() == [("XAU_USD", 3000.0, 3000.5)]
    output = capsys.readouterr().out
    assert output.count("Practice quote unavailable (TimeoutError)") == 2
    assert "[PRACTICE_QUOTES] stored=XAU_USD" in output
    assert "collector stopped" not in output
    assert secret not in output


def test_worker_storage_failure_still_stops_instead_of_retrying_forever(
    tmp_path, monkeypatch, capsys
):
    requests, waits = [], []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(quote_payload()).encode())

    def failing_record(*args, **kwargs):
        raise sqlite3.OperationalError("private-storage-error")

    monkeypatch.setattr(recorder, "urlopen", fake_urlopen)
    monkeypatch.setattr(recorder, "record", failing_record)
    monkeypatch.setattr(recorder.time, "sleep", waits.append)
    recorder.collect_for_worker(tmp_path / "quotes.sqlite", "fake-account", "fake-key")
    assert len(requests) == 1 and waits == []
    output = capsys.readouterr().out
    assert "collector stopped (OperationalError)" in output
    assert "private-storage-error" not in output
