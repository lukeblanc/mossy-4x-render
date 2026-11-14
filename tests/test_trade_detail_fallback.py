import httpx
import pytest

from app.broker import read_trade_details


def client_for(responses, requests):
    def handler(request):
        requests.append(request)
        status, payload = responses[len(requests) - 1]
        return httpx.Response(status, json=payload)
    return httpx.Client(base_url="https://broker.invalid", transport=httpx.MockTransport(handler))


def test_detail_404_recovers_only_requested_closed_trade_without_history_scan():
    requests = []
    closed = {"id": "6515", "state": "CLOSED", "instrument": "AUD_USD"}
    with client_for([(404, {}), (200, {"trades": [closed]})], requests) as client:
        assert read_trade_details(client, "account", "6515") == closed
    assert [r.method for r in requests] == ["GET", "GET"]
    assert requests[0].url.path == "/v3/accounts/account/trades/6515"
    assert requests[1].url.path == "/v3/accounts/account/trades"
    assert dict(requests[1].url.params) == {"ids": "6515", "state": "ALL", "count": "1"}


@pytest.mark.parametrize("payload", [{}, {"trades": None}, {"trades": []},
    {"trades": [{"id": "999"}]}, {"trades": [{"id": "6515"}, {"id": "6515"}]},
    {"trades": [None]}])
def test_list_fallback_rejects_missing_conflicting_or_ambiguous_identity(payload):
    requests = []
    with client_for([(404, {}), (200, payload), (404, {})], requests) as client:
        assert read_trade_details(client, "account", "6515") is None


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_detail_read_failures_do_not_trigger_more_requests(status):
    requests = []
    with client_for([(status, {})], requests) as client:
        with pytest.raises(RuntimeError):
            read_trade_details(client, "account", "6515")
    assert len(requests) == 1


def test_list_outage_is_not_an_empty_account():
    requests = []
    with client_for([(404, {}), (503, {})], requests) as client:
        with pytest.raises(RuntimeError):
            read_trade_details(client, "account", "6515")


def test_successful_details_need_no_fallback_and_must_match_id():
    requests = []
    trade = {"id": "6515", "state": "OPEN"}
    with client_for([(200, {"trade": trade})], requests) as client:
        assert read_trade_details(client, "account", "6515") == trade
    assert len(requests) == 1
    with client_for([(200, {"trade": {"id": "999"}})], []) as client:
        assert read_trade_details(client, "account", "6515") is None


def test_read_diagnostics_preserve_original_transport_exception(capsys):
    error = httpx.ReadTimeout('private-request-url-and-key')
    def handler(request):
        raise error
    with httpx.Client(base_url='https://broker.invalid',
                      transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.ReadTimeout) as raised:
            read_trade_details(client, 'private-account', '6515')
    assert raised.value is error
    output = capsys.readouterr().out
    assert 'private-request-url-and-key' not in output
    assert 'private-account' not in output


def test_successful_read_does_not_emit_failure_event(capsys):
    with client_for([(200, {'trade': {'id': '6515', 'state': 'OPEN'}})], []) as client:
        assert read_trade_details(client, 'account', '6515')['id'] == '6515'
    assert 'TRADE-READ-FAILURE' not in capsys.readouterr().out
