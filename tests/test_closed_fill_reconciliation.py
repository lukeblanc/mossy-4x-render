"""A confirmed opening still counts when it closes before stop verification."""
from copy import deepcopy

import pytest

from app.broker import Broker
from test_broker import DummyClient, DummyResponse, _configure_settings


class ClosedFillClient(DummyClient):
    def __init__(self):
        super().__init__({})
        self.reads = []
        self.posts = 0
        self.closes = []
        self.detail = {'id': '2', 'instrument': 'AUD_USD', 'state': 'CLOSED', 'currentUnits': '0'}
        self.account = {'id': 'acct-123', 'currency': 'AUD', 'openTradeCount': 0,
                        'openPositionCount': 0, 'pendingOrderCount': 0}
        self.watermark = '4'
        self.open_trades = []
        self.summary_failure = False

    def post(self, path, json):
        self.posts += 1
        return super().post(path, json)

    def get(self, path, params=None):
        self.reads.append(path)
        if path.endswith('/summary'):
            return DummyResponse(503 if self.summary_failure else 200, {
                'account': deepcopy(self.account), 'lastTransactionID': self.watermark})
        if path.endswith('/openTrades'):
            return DummyResponse(200, {'trades': deepcopy(self.open_trades)})
        if path.endswith('/trades/2'):
            return DummyResponse(200, {'trade': deepcopy(self.detail)})
        return super().get(path, params)

    def put(self, path, json):
        self.closes.append(path)
        return super().put(path, json)


@pytest.fixture
def closed_fill(monkeypatch, tmp_path):
    _configure_settings(monkeypatch)
    monkeypatch.setenv('MOSSY_STATE_PATH', str(tmp_path))
    monkeypatch.delenv('MOSSY_PRACTICE_EXPERIMENT_ID', raising=False)
    monkeypatch.setenv('MAX_RISK_PER_TRADE_CCY', '0.20')
    monkeypatch.setenv('HARD_MAX_LOSS_CCY', '0.20')
    client = ClosedFillClient()
    monkeypatch.setattr(Broker, '_client', lambda self: client)
    monkeypatch.setattr(Broker, 'conversion_rate', lambda *args: 1.0)
    return Broker(), client


def test_verified_closed_fill_is_recordable_without_emergency_close(closed_fill):
    broker, client = closed_fill
    result = broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)
    assert result['status'] == 'SENT'
    assert result['trade_state'] == 'CLOSED'
    assert result['response']['orderFillTransaction']['tradeOpened']['tradeID'] == '2'
    assert client.posts == 1 and client.closes == []
    assert any(path.endswith('/summary') for path in client.reads)
    assert any(path.endswith('/openTrades') for path in client.reads)
    assert broker.entry_halt_reason is None


@pytest.mark.parametrize('damage', ['summary-unavailable', 'wrong-account', 'wrong-currency',
                                  'open-trade-count', 'open-position-count', 'pending-order',
                                  'missing-count', 'bool-count', 'stale-cursor', 'bad-cursor',
                                  'open-list-unavailable', 'contradictory-open-list'])
def test_closed_fill_requires_verified_fresh_flat_account(closed_fill, monkeypatch, damage):
    broker, client = closed_fill
    if damage == 'summary-unavailable': client.summary_failure = True
    elif damage == 'wrong-account': client.account['id'] = 'other-account'
    elif damage == 'wrong-currency': client.account['currency'] = 'USD'
    elif damage == 'open-trade-count': client.account['openTradeCount'] = 1
    elif damage == 'open-position-count': client.account['openPositionCount'] = 1
    elif damage == 'pending-order': client.account['pendingOrderCount'] = 1
    elif damage == 'missing-count': del client.account['pendingOrderCount']
    elif damage == 'bool-count': client.account['openTradeCount'] = False
    elif damage == 'stale-cursor': client.watermark = '2'
    elif damage == 'bad-cursor': client.watermark = 'bad'
    elif damage == 'open-list-unavailable': client.open_trades = None
    else: client.open_trades = [{'id': '2', 'instrument': 'AUD_USD'}]
    result = broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)
    assert result['status'] == 'UNKNOWN'
    assert broker.entry_halt_reason == 'closed-fill-reconciliation-unavailable'
    assert Broker().entry_halt_reason == broker.entry_halt_reason
    assert client.posts == 1 and client.closes == []
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.posts == 1


@pytest.mark.parametrize('change', [{'id': '9'}, {'instrument': 'GBP_USD'}, {'currentUnits': '1'},
                                   {'currentUnits': 'NaN'}, {'currentUnits': None},
                                   {'state': 'CLOSE_PENDING'}, {'state': None}])
def test_inconsistent_or_unknown_initial_details_retain_emergency_halt(closed_fill, change):
    broker, client = closed_fill
    client.detail.update(change)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'UNKNOWN'
    assert broker.entry_halt_reason == 'protective-stop-not-confirmed'
    assert client.posts == 1 and len(client.closes) == 1


@pytest.mark.parametrize('second', [None, {'id': '2', 'state': 'OPEN'},
                                    {'id': '2', 'state': 'CLOSED', 'currentUnits': '1'},
                                    {'id': '2', 'state': 'CLOSED', 'instrument': 'GBP_USD'}])
def test_closure_must_remain_confirmed_after_account_and_open_list_reads(closed_fill, monkeypatch, second):
    broker, client = closed_fill
    original = client.get
    detail_reads = 0
    def get(path, params=None):
        nonlocal detail_reads
        if path.endswith('/trades/2'):
            detail_reads += 1
            if detail_reads > 1: return DummyResponse(200, {'trade': second})
        return original(path, params)
    monkeypatch.setattr(client, 'get', get)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'UNKNOWN'
    assert broker.entry_halt_reason in {'protective-stop-audit-unavailable', 'unprotected-open-trade'}
    assert client.closes == ['/v3/accounts/acct-123/trades/2/close']
    assert detail_reads >= 3


@pytest.mark.parametrize('open_list', [None, [], [{'id': '4', 'instrument': 'AUD_USD'}]])
def test_failed_flat_checkpoint_protects_known_fill_and_other_exposure(closed_fill, monkeypatch, open_list):
    broker, client = closed_fill
    client.summary_failure = True
    client.open_trades = open_list
    original = client.get
    detail_reads = 0
    def get(path, params=None):
        nonlocal detail_reads
        if path.endswith('/trades/2'):
            detail_reads += 1
            if detail_reads > 1:
                return DummyResponse(200, {'trade': {'id': '2', 'instrument': 'AUD_USD', 'state': 'OPEN'}})
        if path.endswith('/trades/4'):
            return DummyResponse(503, {})
        return original(path, params)
    monkeypatch.setattr(client, 'get', get)
    result = broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)
    assert result['status'] == 'UNKNOWN'
    assert broker.entry_halt_reason
    assert '/v3/accounts/acct-123/trades/2/close' in client.closes
    if open_list:
        assert '/v3/accounts/acct-123/trades/4/close' in client.closes
    assert client.posts == 1


@pytest.mark.parametrize('protected', [True, False])
def test_remaining_exposure_is_audited_without_reclosing_the_verified_closed_fill(closed_fill, monkeypatch, protected):
    from test_closed_trade_audit import protected as protected_trade
    broker, client = closed_fill
    client.open_trades = [{'id': '4', 'instrument': 'AUD_USD'}]
    other = protected_trade('4')
    if not protected: other.pop('stopLossOrder')
    original_get, original_put = client.get, client.put
    def get(path, params=None):
        if path.endswith('/trades/4'): return DummyResponse(200, {'trade': deepcopy(other)})
        return original_get(path, params)
    def put(path, json):
        if path.endswith('/trades/4/close'):
            client.closes.append(path)
            other.update(state='CLOSED', currentUnits='0')
            return DummyResponse(200, {'orderFillTransaction': {'tradesClosed': [{'tradeID': '4'}]}})
        return original_put(path, json)
    monkeypatch.setattr(client, 'get', get)
    monkeypatch.setattr(client, 'put', put)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'UNKNOWN'
    assert client.closes == ([] if protected else ['/v3/accounts/acct-123/trades/4/close'])
    assert broker.entry_halt_reason == ('closed-fill-reconciliation-unavailable' if protected
                                        else 'unprotected-open-trade')


def test_existing_halt_still_prevents_submission(closed_fill):
    broker, client = closed_fill
    broker._latch_entry_halt('existing-incident')
    assert Broker().place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.posts == 0 and client.closes == []
    assert broker._entry_halt_path.read_text().strip() == 'existing-incident'


def test_new_halt_during_reconciliation_is_never_cleared(closed_fill, monkeypatch):
    broker, client = closed_fill
    original = client.get
    def get(path, params=None):
        if path.endswith('/summary'): broker._latch_entry_halt('new-incident')
        return original(path, params)
    monkeypatch.setattr(client, 'get', get)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert broker.entry_halt_reason == Broker().entry_halt_reason == 'new-incident'
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.posts == 1


def test_closed_fill_diagnostics_do_not_log_raw_detail_fields(closed_fill, capsys):
    broker, client = closed_fill
    client.detail.update(accountID='PRIVATE_ACCOUNT', Authorization='PRIVATE_TOKEN')
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    output = capsys.readouterr().out
    assert 'closed-before-stop-verification' in output
    assert 'PRIVATE_' not in output


def test_periodic_unavailable_detail_still_closes_and_halts(closed_fill, monkeypatch):
    """Reproduce the logged 7610 failure class without assuming earlier closure."""
    broker, client = closed_fill
    client.recorder['trade_id'] = '2'
    client.open_trades = [{'id': '2', 'instrument': 'AUD_USD'}]
    original = client.get
    def get(path, params=None):
        if path.endswith('/trades/2') and not client.closes:
            return DummyResponse(503, {'private_error': 'PRIVATE'})
        return original(path, params)
    monkeypatch.setattr(client, 'get', get)
    assert broker.list_open_trades() is None
    assert client.closes == ['/v3/accounts/acct-123/trades/2/close']
    assert broker.entry_halt_reason == 'protective-stop-audit-unavailable'
