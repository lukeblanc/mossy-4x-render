"""Offline list/detail races: refresh exposure without weakening the entry halt."""
from copy import deepcopy

import pytest

from app.broker import Broker
from test_broker import DummyClient, DummyResponse, _configure_settings


def summary(ticket):
    return {'id': ticket, 'instrument': 'AUD_USD'}


def protected(ticket):
    return {
        **summary(ticket), 'state': 'OPEN', 'currentUnits': '100', 'price': '0.70',
        'stopLossOrder': {'id': '99', 'state': 'PENDING', 'type': 'STOP_LOSS',
                          'tradeID': ticket, 'price': '0.699'},
    }


def closed(ticket):
    return {'id': ticket, 'state': 'CLOSED', 'currentUnits': '0'}


class SnapshotClient(DummyClient):
    def __init__(self):
        super().__init__({})
        self.snapshots = [[summary('2')], []]
        self.details = {'2': closed('2')}
        self.reads = []
        self.closes = []

    def get(self, path, params=None):
        self.reads.append(path.rsplit('/', 1)[-1])
        if path.endswith('/openTrades'):
            # An unexpected third refresh fails the test rather than looping.
            assert self.snapshots, 'Unbounded open-trade refresh'
            snapshot = self.snapshots.pop(0)
            if isinstance(snapshot, Exception):
                raise snapshot
            if isinstance(snapshot, DummyResponse):
                return snapshot
            return DummyResponse(status_code=200, payload={'trades': deepcopy(snapshot)})
        if '/trades/' in path:
            value = self.details[path.rsplit('/', 1)[-1]]
            if isinstance(value, Exception):
                raise value
            return DummyResponse(status_code=200, payload={'trade': deepcopy(value)})
        return super().get(path, params=params)

    def put(self, path, json):
        assert path.endswith('/close')
        assert json == {'units': 'ALL'}
        ticket = path.split('/')[-2]
        self.closes.append(ticket)
        self.details[ticket] = closed(ticket)
        return DummyResponse(status_code=200, payload={
            'orderFillTransaction': {'tradesClosed': [{'tradeID': ticket}]}})

    def post(self, *args, **kwargs):
        pytest.fail('An audit must never place an entry order')


@pytest.fixture
def race(monkeypatch, tmp_path):
    _configure_settings(monkeypatch)
    monkeypatch.setenv('MAX_RISK_PER_TRADE_CCY', '0.20')
    monkeypatch.setenv('MOSSY_STATE_PATH', str(tmp_path))
    client = SnapshotClient()
    monkeypatch.setattr(Broker, '_client', lambda self: client)
    monkeypatch.setattr(Broker, 'conversion_rate', lambda *args: 1.0)
    return Broker(), client


def test_closure_between_list_and_detail_returns_fresh_empty_snapshot(race, capsys):
    broker, client = race
    client.details['2'].update({'Authorization': 'private-token',
                                'accountID': 'private-account'})
    assert broker.list_open_trades() == []
    assert client.reads == ['openTrades', '2', 'openTrades']
    assert client.closes == []
    assert broker.entry_halt_reason is None
    assert not broker._entry_halt_path.exists()
    output = capsys.readouterr().out
    assert 'trade-closed-during-audit' in output
    assert 'emergency_close' not in output
    assert 'private-' not in output


def test_refresh_audits_existing_and_new_exposure_and_returns_only_fresh_list(race):
    broker, client = race
    client.snapshots = [[summary('2'), summary('4')], [summary('4'), summary('6')]]
    client.details.update({'4': protected('4'), '6': protected('6')})
    assert broker.list_open_trades() == [summary('4'), summary('6')]
    assert client.reads == ['openTrades', '2', '4', 'openTrades', '4', '6']
    assert client.closes == []
    assert broker.entry_halt_reason is None


def test_refresh_rechecks_a_stop_that_was_safe_on_the_original_list(race, monkeypatch):
    broker, client = race
    client.snapshots = [[summary('4'), summary('2')], [summary('4')]]
    client.details['4'] = protected('4')
    original_get = client.get

    def change_stop_on_refresh(path, params=None):
        if path.endswith('/openTrades') and client.reads.count('openTrades') == 1:
            client.details['4'].pop('stopLossOrder')
        return original_get(path, params=params)

    monkeypatch.setattr(client, 'get', change_stop_on_refresh)
    assert broker.list_open_trades() is None
    assert client.reads == ['openTrades', '4', '2', 'openTrades', '4', '4']
    assert client.closes == ['4']
    assert broker.entry_halt_reason == 'unprotected-open-trade'


@pytest.mark.parametrize('initially_present', [True, False])
@pytest.mark.parametrize('failure', ['missing-stop', 'above-twenty-cents', 'unknown'])
def test_closure_never_hides_other_unsafe_or_unknown_exposure(race, initially_present, failure):
    broker, client = race
    trade = protected('4')
    if failure == 'missing-stop':
        trade.pop('stopLossOrder')
    elif failure == 'above-twenty-cents':
        trade['currentUnits'] = '201'  # A$0.201; the A$0.20 audit remains strict.
    else:
        trade = None
    client.details['4'] = trade
    client.snapshots = ([[summary('2'), summary('4')], []] if initially_present
                        else [[summary('2')], [summary('4')]])
    assert broker.list_open_trades() is None
    assert client.closes == ['4']
    assert client.reads.count('openTrades') == 2
    assert broker.entry_halt_reason == ('protective-stop-audit-unavailable'
                                        if failure == 'unknown' else 'unprotected-open-trade')
    assert broker._entry_halt_path.exists()


@pytest.mark.parametrize('snapshot', [
    RuntimeError('private-error'),
    DummyResponse(status_code=503, payload={'secret': 'private-payload'}),
    DummyResponse(status_code=200, payload={'wrong': 'private-payload'}),
    None, {}, [None], [{'id': 'bad', 'instrument': 'AUD_USD'}], [{'id': '4'}],
])
def test_failed_or_malformed_refresh_cannot_return_stale_exposure(race, capsys, snapshot):
    broker, client = race
    client.snapshots[1] = snapshot
    assert broker.list_open_trades() is None
    assert client.closes == []
    assert broker.entry_halt_reason == 'protective-stop-audit-unavailable'
    assert broker._entry_halt_path.exists()
    output = capsys.readouterr().out
    assert 'open-trades-refresh-unavailable' in output
    assert 'private-' not in output


def test_closed_id_still_in_refreshed_list_blocks_but_audits_other_exposure(race, capsys):
    broker, client = race
    client.snapshots[1] = [summary('2'), summary('4')]
    client.details['4'] = protected('4')
    client.details['4'].pop('stopLossOrder')
    assert broker.list_open_trades() is None
    assert client.reads.count('2') == 1
    assert client.reads.count('openTrades') == 2
    assert client.closes == ['4']
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert 'open-trades-refresh-inconsistent' in capsys.readouterr().out


def test_second_closure_is_bounded_and_never_returns_an_unaudited_snapshot(race, capsys):
    broker, client = race
    client.snapshots[1] = [summary('4'), summary('6')]
    client.details.update({'4': closed('4'), '6': protected('6')})
    assert broker.list_open_trades() is None
    assert client.reads == ['openTrades', '2', 'openTrades', '4', '6']
    assert client.closes == []
    assert broker.entry_halt_reason == 'protective-stop-audit-unavailable'
    assert 'open-trades-refresh-inconsistent' in capsys.readouterr().out


@pytest.mark.parametrize('detail', [
    None, [], {}, {'id': '9', 'state': 'CLOSED'}, {'id': '2'},
    {'id': '2', 'state': None}, {'id': '2', 'state': ['CLOSED']},
    {'id': '2', 'state': 'CLOSE_PENDING'}, {'id': '2', 'state': 'private-state'},
    {'id': '2', 'state': 'CLOSED', 'currentUnits': '1'},
    {'id': '2', 'state': 'CLOSED', 'currentUnits': '-1'},
    {'id': '2', 'state': 'CLOSED', 'currentUnits': 'NaN'},
    {'id': '2', 'state': 'CLOSED', 'currentUnits': None},
    {'id': '2', 'state': 'CLOSED', 'currentUnits': []},
    {'id': '2', 'state': 'CLOSED', 'instrument': 'private-instrument'},
    {'id': '2', 'state': 'CLOSED', 'instrument': None},
    {'id': '2', 'state': 'CLOSED', 'instrument': []},
    RuntimeError('private-error'),
])
def test_unknown_or_malformed_detail_still_fails_closed(race, capsys, detail):
    broker, client = race
    client.details['2'] = detail
    assert broker.list_open_trades() is None
    assert client.reads.count('openTrades') == 1
    assert client.closes == ['2']
    assert broker.entry_halt_reason == 'protective-stop-audit-unavailable'
    assert broker._entry_halt_path.exists()
    output = capsys.readouterr().out
    assert 'trade-closed-during-audit' not in output
    assert 'private-' not in output


@pytest.mark.parametrize('startup', [False, True])
def test_verified_closure_never_clears_existing_persisted_halt(race, startup):
    broker, client = race
    broker._latch_entry_halt('unprotected-open-trade')
    broker = Broker()  # Load the actual test-local persisted marker.
    if startup:
        assert broker.connectivity_check()['ok'] is True
    else:
        assert broker.list_open_trades() == []
    assert client.closes == []
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert broker._entry_halt_path.read_text() == 'unprotected-open-trade\n'
    assert Broker().entry_halt_reason == broker.entry_halt_reason
    assert broker.place_order('AUD_USD', 'BUY', 1, sl_distance=0.001)['status'] == 'BLOCKED'


def test_failed_refresh_retains_original_persisted_halt(race):
    broker, client = race
    broker._latch_entry_halt('order-transport-state-uncertain')
    client.snapshots[1] = None
    assert broker.list_open_trades() is None
    assert client.closes == []
    assert broker.entry_halt_reason == 'order-transport-state-uncertain'
    assert broker._entry_halt_path.read_text() == 'order-transport-state-uncertain\n'
