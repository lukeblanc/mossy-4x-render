"""Offline regression cases for protection diagnostics and retained entry halts."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from app.broker import Broker
from test_broker import DummyClient, DummyResponse, _configure_settings


class AuditClient(DummyClient):
    def get(self, path, params=None):
        if path.endswith('/openTrades'):
            trades = [] if self.recorder.get('closed') else [
                {'id': '2', 'instrument': 'AUD_USD', 'stopLossOrderID': '3'}]
            return DummyResponse(status_code=200, payload={'trades': trades})
        if path.endswith('/trades/2'):
            if self.recorder.get('closed'):
                return DummyResponse(status_code=200, payload={'trade': {'id': '2', 'state': 'CLOSED'}})
            return DummyResponse(status_code=200, payload={'trade': self.recorder['trade']})
        return super().get(path, params=params)


@pytest.fixture
def audited_broker(monkeypatch, tmp_path):
    _configure_settings(monkeypatch)
    monkeypatch.setenv('MOSSY_STATE_PATH', str(tmp_path))
    trade = {'id': '2', 'state': 'OPEN', 'instrument': 'AUD_USD',
             'price': '0.70261', 'currentUnits': '828',
             'stopLossOrder': {'id': '3', 'state': 'PENDING', 'type': 'STOP_LOSS',
                               'tradeID': '2', 'price': '0.70219'}}
    recorded = {'trade': deepcopy(trade), 'closed': False, 'trade_id': '2'}
    monkeypatch.setattr(Broker, '_client', lambda self: AuditClient(recorded))
    monkeypatch.setattr(Broker, 'conversion_rate', lambda self, *args: 1.4375)
    return Broker(), recorded


def audit_record(capsys):
    lines = capsys.readouterr().out.splitlines()
    return json.loads(next(line.split('] ', 1)[1] for line in lines
                           if line.startswith('[BROKER][PROTECTION-AUDIT] ')))


def test_conversion_drift_has_specific_reason_and_keeps_cash_cap(audited_broker, monkeypatch, capsys):
    broker, recorded = audited_broker
    assert broker.list_open_trades() is not None
    assert broker.entry_halt_reason is None
    # Only the conversion changes; the valid pending stop remains in place.
    monkeypatch.setattr(broker, 'conversion_rate', lambda *args: 1.438)
    assert broker.list_open_trades() is None
    event = audit_record(capsys)
    assert event['audit_reason'] == 'stop-cash-risk-exceeded'
    assert event['stop_order_id'] == '3'
    assert event['loss_conversion'] == '1.438'
    assert float(event['planned_stop_risk']) > 0.50
    assert event['cash_limit'] == '0.5'
    assert recorded['closed'] is True
    assert broker.entry_halt_reason == 'unprotected-open-trade'


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('stopLossOrder', None, 'stop-order-missing'),
    ('instrument', 'GBP_USD', 'trade-instrument-mismatch'),
    ('currentUnits', 'NaN', 'stop-risk-input-invalid'),
    ('price', 'bad', 'stop-risk-input-invalid'),
])
def test_trade_failures_keep_distinct_diagnostics(audited_broker, capsys, field, value, reason):
    broker, recorded = audited_broker
    recorded['trade'][field] = value
    assert broker.list_open_trades() is None
    assert audit_record(capsys)['audit_reason'] == reason
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert recorded['closed'] is True


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('id', 'bad', 'stop-id-invalid'),
    ('id', '4', 'stop-id-mismatch'),
    ('state', 'CANCELLED', 'stop-not-pending'),
    ('type', 'TAKE_PROFIT', 'stop-type-invalid'),
    ('tradeID', '9', 'stop-trade-id-mismatch'),
])
def test_stop_failures_keep_distinct_diagnostics(audited_broker, capsys, field, value, reason):
    broker, recorded = audited_broker
    recorded['trade']['stopLossOrder'][field] = value
    assert broker.list_open_trades() is None
    assert audit_record(capsys)['audit_reason'] == reason
    assert broker.entry_halt_reason == 'unprotected-open-trade'


@pytest.mark.parametrize('conversion', [None, float('nan'), float('inf'), 0.0, -1.0])
def test_unavailable_conversion_fails_closed(audited_broker, monkeypatch, capsys, conversion):
    broker, recorded = audited_broker
    monkeypatch.setattr(broker, 'conversion_rate', lambda *args: conversion)
    assert broker.list_open_trades() is None
    event = audit_record(capsys)
    assert event['audit_reason'] == 'loss-conversion-unavailable'
    assert event['status'] == 'unknown'
    assert recorded['closed'] is True
    assert broker.entry_halt_reason == 'protective-stop-audit-unavailable'


def test_emergency_close_retries_transient_broker_failure(audited_broker, monkeypatch, capsys):
    broker, recorded = audited_broker
    recorded['trade']['stopLossOrder'] = None
    original_put = AuditClient.put
    attempts = {'count': 0}

    def flaky_put(self, path, json):
        attempts['count'] += 1
        if attempts['count'] < 3:
            return DummyResponse(status_code=503)
        return original_put(self, path, json)

    monkeypatch.setattr(AuditClient, 'put', flaky_put)
    monkeypatch.setattr('app.broker.time.sleep', lambda *_: None)

    assert broker.list_open_trades() is None
    assert attempts['count'] == 3
    assert recorded['closed'] is True
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    output = capsys.readouterr().out
    assert '"attempt": 1' in output and '"retrying": true' in output
    assert '"attempt": 3' in output and '"result": "close-fill-confirmed"' in output


def test_emergency_close_does_not_retry_non_transient_rejection(audited_broker, monkeypatch):
    broker, recorded = audited_broker
    recorded['trade']['stopLossOrder'] = None
    attempts = {'count': 0}

    def rejected_put(self, path, json):
        attempts['count'] += 1
        return DummyResponse(status_code=400)

    monkeypatch.setattr(AuditClient, 'put', rejected_put)
    monkeypatch.setattr('app.broker.time.sleep', lambda *_: pytest.fail('must not sleep'))

    assert broker.list_open_trades() is None
    assert attempts['count'] == 1
    assert recorded['closed'] is False
    assert broker.entry_halt_reason == 'unprotected-open-trade'


def test_close_fill_remains_positive_evidence_when_detail_reread_fails(
    audited_broker, monkeypatch, capsys
):
    broker, recorded = audited_broker
    recorded['trade']['stopLossOrder'] = None
    original_get = AuditClient.get
    close_seen = {'value': False}
    original_put = AuditClient.put

    def put(self, path, json):
        response = original_put(self, path, json)
        close_seen['value'] = True
        return response

    def get(self, path, params=None):
        if close_seen['value'] and path.endswith('/trades/2'):
            return DummyResponse(status_code=503)
        return original_get(self, path, params=params)

    monkeypatch.setattr(AuditClient, 'put', put)
    monkeypatch.setattr(AuditClient, 'get', get)

    assert broker.list_open_trades() is None
    assert recorded['closed'] is True
    output = capsys.readouterr().out
    assert '"result": "close-fill-confirmed"' in output
    assert '"detail_state": "unavailable"' in output


def test_closed_trade_poll_does_not_clear_persisted_halt(audited_broker):
    broker, recorded = audited_broker
    recorded['trade']['stopLossOrder'] = None
    assert broker.list_open_trades() is None
    assert recorded['closed'] is True
    assert broker.list_open_trades() == []
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert broker._entry_halt_path.read_text().strip() == broker.entry_halt_reason
    assert Broker().entry_halt_reason == broker.entry_halt_reason
    assert broker.place_order('AUD_USD', 'BUY', 1, sl_distance=0.00042)['status'] == 'BLOCKED'


def test_halt_property_is_side_effect_free(audited_broker, monkeypatch):
    broker, recorded = audited_broker
    broker._latch_entry_halt('test-retained')
    def no_client(*args):
        pytest.fail('Observing health must not call the broker')
    monkeypatch.setattr(broker, '_client', no_client)
    assert broker.entry_halt_reason == 'test-retained'
    assert broker._entry_halt_path.read_text().strip() == 'test-retained'
    assert recorded['closed'] is False


def test_failed_close_remains_halted_and_is_not_confirmed(audited_broker, monkeypatch, capsys):
    broker, recorded = audited_broker
    recorded['trade']['stopLossOrder'] = None
    monkeypatch.setattr(AuditClient, 'put', lambda *args, **kwargs: DummyResponse(status_code=503))
    assert broker.list_open_trades() is None
    output = capsys.readouterr().out
    assert 'emergency_close=False' in output
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert recorded['closed'] is False


@pytest.mark.parametrize('closed', [True, False])
def test_clean_startup_audit_retains_existing_latch(audited_broker, capsys, closed):
    broker, recorded = audited_broker
    broker._latch_entry_halt('unprotected-open-trade')
    recorded['closed'] = closed
    assert broker.connectivity_check()['ok'] is True
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert broker._entry_halt_path.read_text().strip() == broker.entry_halt_reason
    assert Broker().entry_halt_reason == broker.entry_halt_reason
    assert broker.place_order('AUD_USD', 'BUY', 1, sl_distance=0.00042)['status'] == 'BLOCKED'
    assert '[BROKER][RECOVERY]' not in capsys.readouterr().out


def test_audit_logs_exclude_raw_response_secrets(audited_broker, capsys):
    broker, recorded = audited_broker
    recorded['trade'].update({'accountID': 'private-account-123',
                              'Authorization': 'Bearer secret-value',
                              'private_metadata': 'private-sentinel'})
    recorded['trade']['stopLossOrder'] = None
    assert broker.list_open_trades() is None
    output = capsys.readouterr().out
    for secret in ('private-account-123', 'secret-value', 'private-sentinel'):
        assert secret not in output
    assert 'stop-order-missing' in output


def test_startup_unavailable_open_trade_snapshot_retains_halt(audited_broker, monkeypatch, capsys):
    broker, recorded = audited_broker
    broker._latch_entry_halt('unprotected-open-trade')
    original_get = AuditClient.get
    def unavailable(self, path, params=None):
        if path.endswith('/openTrades'):
            return DummyResponse(status_code=503)
        return original_get(self, path, params=params)
    monkeypatch.setattr(AuditClient, 'get', unavailable)
    broker.connectivity_check()
    assert broker.entry_halt_reason == 'unprotected-open-trade'
    assert broker._entry_halt_path.exists()
    assert recorded['closed'] is False
    assert '[BROKER][RECOVERY]' not in capsys.readouterr().out


def test_startup_unavailable_exact_trade_evidence_closes_but_never_clears_halt(audited_broker, monkeypatch, capsys):
    broker, recorded = audited_broker
    broker._latch_entry_halt('unprotected-open-trade')
    original_get = AuditClient.get
    def unavailable(self, path, params=None):
        if path.endswith('/trades/2'):
            return DummyResponse(status_code=503)
        return original_get(self, path, params=params)
    monkeypatch.setattr(AuditClient, 'get', unavailable)
    broker.connectivity_check()
    assert broker.entry_halt_reason == 'protective-stop-audit-unavailable'
    assert broker._entry_halt_path.exists()
    # The authenticated close fill can positively confirm this exact trade was
    # closed even when the immediate detail read remains unavailable. The
    # persisted incident must still remain latched until explicit recovery.
    output = capsys.readouterr().out
    assert 'emergency_close=True' in output
    assert '"result": "close-fill-confirmed"' in output
    assert '"detail_state": "unavailable"' in output
    assert '[BROKER][RECOVERY]' not in output
