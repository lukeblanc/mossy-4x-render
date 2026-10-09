"""All orders/metadata are fakes; ledgers live only in pytest temporary paths."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pytest

from app.broker import Broker
from app.config import settings
from src.practice_experiment import ExperimentBlocked, PracticeExperiment
from src.risk_manager import RiskManager
from test_broker import DummyClient, DummyResponse, _configure_settings


@pytest.fixture
def ledger(tmp_path):
    plan = PracticeExperiment(tmp_path, 'approved-test-only', 'acct-123')
    plan.prepare()
    return plan


def confirm(plan, ticket):
    token, units = plan.reserve()
    plan.confirm_opening(token, str(ticket))
    return units


def test_plan_requires_explicit_activation_and_cannot_reinitialize(ledger):
    with pytest.raises(ExperimentBlocked, match='not-active'):
        ledger.reserve()
    with pytest.raises(sqlite3.IntegrityError):
        ledger.prepare()
    ledger.activate()
    with pytest.raises(ExperimentBlocked, match='cannot-reactivate'):
        ledger.activate()


def test_ten_new_fills_then_pause_persists_across_instances(ledger):
    ledger.activate()
    for number in range(10):
        fresh = PracticeExperiment(ledger.path.parent, ledger.id, 'acct-123')
        assert confirm(fresh, 1000 + number) == 10
    assert ledger.status() == {'state': 'complete', 'filled_count': 10, 'pending': 0, 'units': 10}
    with pytest.raises(ExperimentBlocked, match='not-active'):
        PracticeExperiment(ledger.path.parent, ledger.id, 'acct-123').reserve()
    with pytest.raises(ExperimentBlocked, match='cannot-reactivate'):
        ledger.activate()


def test_unresolved_reservation_survives_restart_and_never_retries(ledger):
    ledger.activate()
    ledger.reserve()  # Crash before/after submission is intentionally ambiguous.
    fresh = PracticeExperiment(ledger.path.parent, ledger.id, 'acct-123')
    with pytest.raises(ExperimentBlocked, match='submission-unresolved'):
        fresh.reserve()
    assert fresh.status()['pending'] == 1
    assert fresh.status()['filled_count'] == 0


def test_confirmed_opening_counts_once_and_duplicate_trade_cannot_spend_another_slot(ledger):
    ledger.activate()
    token, _ = ledger.reserve()
    ledger.confirm_opening(token, '101')
    ledger.confirm_opening(token, '101')
    second, _ = ledger.reserve()
    with pytest.raises(ExperimentBlocked):
        ledger.confirm_opening(second, '101')
    assert ledger.status()['filled_count'] == 1
    assert ledger.status()['pending'] == 1


def test_atomic_reservation_allows_only_one_concurrent_submitter(ledger):
    ledger.activate()
    def reserve():
        try:
            return ledger.reserve()
        except ExperimentBlocked:
            return None
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: reserve(), range(4)))
    assert sum(value is not None for value in results) == 1
    assert ledger.status()['pending'] == 1


def test_only_unsent_skips_release_slots_and_pause_does_not_reset_phase(ledger):
    ledger.activate()
    token, _ = ledger.reserve()
    ledger.skip_before_submission(token)
    assert ledger.status()['filled_count'] == 0
    confirm(ledger, '101')
    ledger.pause()
    assert ledger.status()['filled_count'] == 1
    with pytest.raises(ExperimentBlocked, match='not-active'):
        ledger.reserve()
    with pytest.raises(ExperimentBlocked, match='cannot-reactivate'):
        ledger.activate()


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'wrong-account', 'count-mismatch'])
def test_unavailable_or_inconsistent_ledger_fails_closed(ledger, damage):
    ledger.activate()
    if damage == 'missing':
        ledger.path.unlink()
    elif damage == 'corrupt':
        ledger.path.write_bytes(b'not a database')
    elif damage == 'wrong-account':
        ledger = PracticeExperiment(ledger.path.parent, ledger.id, 'different-account')
    else:
        with sqlite3.connect(ledger.path) as connection:
            connection.execute('UPDATE experiments SET filled_count=1')
    with pytest.raises(ExperimentBlocked):
        ledger.reserve()


class ExperimentClient(DummyClient):
    def __init__(self):
        super().__init__({})
        self.requests = []
        self.submitted = []
        self.metadata = {'name': 'AUD_USD', 'type': 'CURRENCY',
                         'minimumTradeSize': '1', 'tradeUnitsPrecision': 0}
        self.account = {'id': 'acct-123', 'currency': 'AUD', 'openTradeCount': 0,
                        'openPositionCount': 0, 'pendingOrderCount': 0}
        self.summary_last_transaction_id = '999'
        self.failure = None

    def get(self, path, params=None):
        self.requests.append(('GET', path))
        if path.endswith('/summary'):
            return DummyResponse(200, {
                'account': deepcopy(self.account),
                'lastTransactionID': self.summary_last_transaction_id,
            })
        if path.endswith('/instruments'):
            assert params == {'instruments': 'AUD_USD'}
            return DummyResponse(200, {'instruments': [deepcopy(self.metadata)]})
        return super().get(path, params=params)

    def post(self, path, json):
        self.requests.append(('POST', path))
        self.submitted.append(deepcopy(json))
        if self.failure == 'timeout':
            raise RuntimeError('private-transport-error')
        if self.failure == 'rejected':
            return DummyResponse(201, {'orderRejectTransaction': {'id': '11'}})
        if self.failure == 'malformed':
            return DummyResponse(201, {})
        response = super().post(path, json)
        ticket = '900' if self.failure == 'historical' else str(1000 + len(self.submitted))
        self.recorder['trade_id'] = ticket
        response.payload['orderFillTransaction']['tradeOpened']['tradeID'] = ticket
        if self.failure == 'partial':
            response.payload['orderFillTransaction']['tradeOpened']['units'] = '1'
        return response


@pytest.fixture
def experiment_broker(ledger, monkeypatch):
    _configure_settings(monkeypatch)
    monkeypatch.setenv('MOSSY_STATE_PATH', str(ledger.path.parent))
    monkeypatch.setenv('MOSSY_PRACTICE_EXPERIMENT_ID', ledger.id)
    monkeypatch.setenv('MAX_RISK_PER_TRADE_CCY', '0.20')
    monkeypatch.setenv('HARD_MAX_LOSS_CCY', '0.20')
    ledger.activate()
    client = ExperimentClient()
    monkeypatch.setattr(Broker, '_client', lambda self: client)
    monkeypatch.setattr(Broker, 'conversion_rate', lambda *args: 1.0)
    return Broker(), client, ledger


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_exact_ten_then_normal_sizing_returns_after_restart(experiment_broker, side):
    broker, client, plan = experiment_broker
    for number in range(10):
        broker = Broker()
        client.recorder['closed'] = True  # Simulate verified closure before next signal.
        result = broker.place_order('AUD_USD', side, 100, sl_distance=0.001)
        assert result['status'] == 'SENT'
        assert abs(int(client.submitted[-1]['order']['units'])) == 10
    assert plan.status()['filled_count'] == 10
    client.recorder['closed'] = True
    assert Broker().place_order('AUD_USD', side, 100, sl_distance=0.001)['status'] == 'SENT'
    assert [abs(int(order['order']['units'])) for order in client.submitted] == [10] * 10 + [100]
    assert plan.status()['state'] == 'complete'


def test_open_tenth_trade_blocks_normal_sizing_handoff(experiment_broker):
    broker, client, plan = experiment_broker
    for _ in range(10):
        client.recorder['closed'] = True
        assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert plan.status()['state'] == 'complete'
    assert client.recorder['closed'] is False
    assert Broker().place_order('AUD_USD', 'BUY', 100, sl_distance=0.001) == {
        'status': 'BLOCKED', 'reason': 'experiment-completion-handoff-failed'}
    assert len(client.submitted) == 10


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'wrong-account', 'count-mismatch', 'pending'])
def test_completed_ledger_damage_blocks_champion_handoff(experiment_broker, damage):
    broker, client, plan = experiment_broker
    for ticket in range(300, 310):
        confirm(plan, ticket)
    if damage == 'missing':
        plan.path.unlink()
    elif damage == 'corrupt':
        plan.path.write_bytes(b'not a database')
    else:
        with sqlite3.connect(plan.path) as connection:
            if damage == 'wrong-account':
                connection.execute("UPDATE experiments SET account_hash='wrong'")
            elif damage == 'count-mismatch':
                connection.execute('UPDATE experiments SET filled_count=9')
            else:
                connection.execute(
                    "INSERT INTO experiment_intents VALUES "
                    "('pending-token',? ,10,'reserved',NULL,?)",
                    (plan.id, datetime.now(timezone.utc).isoformat()),
                )
    result = broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)
    assert result['status'] == 'BLOCKED'
    assert client.submitted == []


def test_paused_experiment_does_not_restore_champion_sizing(experiment_broker):
    broker, client, plan = experiment_broker
    plan.pause()
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001) == {
        'status': 'BLOCKED', 'reason': 'experiment-not-active'}
    assert client.submitted == []


def test_completed_experiment_restores_other_champion_instruments(experiment_broker):
    broker, client, plan = experiment_broker
    for _ in range(10):
        client.recorder['closed'] = True
        assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert plan.status()['state'] == 'complete'
    client.recorder['closed'] = True
    assert broker.place_order('GBP_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert client.submitted[-1]['order']['instrument'] == 'GBP_USD'
    assert client.submitted[-1]['order']['units'] == '100'


@pytest.mark.parametrize('instrument,units,reason', [
    ('GBP_USD', 100, 'experiment-practice-audusd-only'),
    ('AUD_USD', 9, 'experiment-exact-size-exceeds-strategy-budget'),
    ('AUD_USD', 0, 'invalid-order-units'),
    ('AUD_USD', 2.5, 'invalid-order-units'),
])
def test_no_pair_expansion_rounding_up_or_risk_budget_increase(experiment_broker, instrument, units, reason):
    broker, client, plan = experiment_broker
    assert broker.place_order(instrument, 'BUY', units, sl_distance=0.001) == {
        'status': 'BLOCKED', 'reason': reason}
    assert client.requests == []
    assert plan.status()['pending'] == 0


@pytest.mark.parametrize('field,value', [
    ('minimumTradeSize', '11'), ('minimumTradeSize', 'NaN'), ('minimumTradeSize', None),
    ('tradeUnitsPrecision', None), ('tradeUnitsPrecision', '0'),
    ('tradeUnitsPrecision', -1), ('name', 'GBP_USD'), ('type', 'CFD'),
])
def test_bad_or_unsupported_metadata_never_submits(experiment_broker, field, value):
    broker, client, plan = experiment_broker
    client.metadata[field] = value
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['pending'] == 0
    assert plan.status()['filled_count'] == 0


@pytest.mark.parametrize('field,value', [
    ('currency', 'USD'), ('id', 'different-account'), ('openTradeCount', 1),
    ('openPositionCount', 1), ('pendingOrderCount', 1), ('openTradeCount', None),
])
def test_account_or_exposure_mismatch_blocks(experiment_broker, field, value):
    broker, client, plan = experiment_broker
    client.account[field] = value
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['pending'] == 0


@pytest.mark.parametrize('value', [None, 'bad'])
def test_missing_or_invalid_top_level_transaction_cursor_blocks(experiment_broker, value):
    broker, client, plan = experiment_broker
    client.summary_last_transaction_id = value
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['pending'] == 0


def test_conflicting_nested_transaction_cursor_blocks(experiment_broker):
    broker, client, plan = experiment_broker
    client.account['lastTransactionID'] = '998'
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['pending'] == 0


def test_matching_nested_transaction_cursor_is_accepted(experiment_broker):
    broker, client, plan = experiment_broker
    client.account['lastTransactionID'] = client.summary_last_transaction_id
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert len(client.submitted) == 1
    assert plan.status()['filled_count'] == 1


def test_second_position_blocked_even_when_summary_is_stale_flat(experiment_broker):
    broker, client, plan = experiment_broker
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert len(client.submitted) == 1
    assert plan.status()['filled_count'] == 1


def test_exact_previous_trade_prevents_overlap_when_both_flat_snapshots_are_stale(
    experiment_broker, monkeypatch
):
    broker, client, plan = experiment_broker
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    monkeypatch.setattr(broker, '_read_open_trade_summaries', lambda *args: [])
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert len(client.submitted) == 1
    assert plan.status()['pending'] == 0


@pytest.mark.parametrize('failure', ['timeout', 'rejected', 'malformed'])
def test_uncertain_or_rejected_submission_never_retries_automatically(experiment_broker, failure):
    broker, client, plan = experiment_broker
    client.failure = failure
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] != 'SENT'
    assert plan.status()['pending'] == 1
    assert Broker().place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert len(client.submitted) == 1


def test_partial_fill_is_counted_but_closed_and_halted_instead_of_approximated(experiment_broker):
    broker, client, plan = experiment_broker
    client.failure = 'partial'
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'UNKNOWN'
    assert plan.status()['filled_count'] == 1
    assert client.recorder['closed'] is True
    assert broker.entry_halt_reason == 'fill-does-not-match-request'


def test_existing_halt_remains_authoritative(experiment_broker):
    broker, client, plan = experiment_broker
    broker._latch_entry_halt('unprotected-open-trade')
    assert Broker().place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.requests == []
    assert plan.status()['pending'] == 0


@pytest.mark.parametrize('setting,value', [
    ('MAX_RISK_PER_TRADE_CCY', '0.21'), ('HARD_MAX_LOSS_CCY', '0.21'),
    ('HARD_MAX_LOSS_CCY', 'NaN'), ('HARD_MAX_LOSS_CCY', 'bad'),
])
def test_experiment_requires_preserved_twenty_cent_limits(experiment_broker, monkeypatch, setting, value):
    broker, client, plan = experiment_broker
    monkeypatch.setenv(setting, value)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.requests == []


def test_failed_confirmation_persistence_closes_known_fill_and_halts(experiment_broker, monkeypatch):
    broker, client, plan = experiment_broker
    def fail(*args):
        raise ExperimentBlocked('experiment-ledger-unavailable')
    monkeypatch.setattr(PracticeExperiment, 'confirm_opening', fail)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'UNKNOWN'
    assert client.recorder['closed'] is True
    assert plan.status()['pending'] == 1
    assert broker.entry_halt_reason == 'order-transport-state-uncertain'


def test_daily_history_is_preserved_while_batch_crosses_day_and_restart(experiment_broker, monkeypatch):
    broker, client, plan = experiment_broker
    monkeypatch.setenv('MAX_TRADES_PER_DAY', '20')
    monkeypatch.setenv('MAX_CONCURRENT_POSITIONS', '1')
    config = {'cooldown_candles': 0, 'max_trades_per_day': 20,
              'max_concurrent_positions': 1}
    risk = RiskManager(config, mode='demo', demo_mode=True, state_dir=plan.path.parent)
    now = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
    assert risk.should_open(now, 1000, [], 'AUD_USD', 0.1)[0]
    for _ in range(11):
        risk.register_entry(now, 'AUD_USD')  # Historical entries, not new experiment fills.
    assert plan.status()['filled_count'] == 0
    for _ in range(9):
        assert risk.should_open(now, 1000, [], 'AUD_USD', 0.1)[0]
        client.recorder['closed'] = True
        assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
        risk.register_entry(now, 'AUD_USD')
    assert risk.should_open(now, 1000, [], 'AUD_USD', 0.1) == (False, 'daily-trade-cap')
    assert plan.status()['filled_count'] == 9
    risk = RiskManager(config, mode='demo', demo_mode=True, state_dir=plan.path.parent)
    assert risk.state.daily_entry_count == 20
    now += timedelta(days=1)
    broker = Broker()
    assert risk.should_open(now, 1000, [], 'AUD_USD', 0.1)[0]
    client.recorder['closed'] = True
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    risk.register_entry(now, 'AUD_USD')
    assert risk.state.daily_entry_count == 1
    assert plan.status()['filled_count'] == 10
    assert [abs(int(order['order']['units'])) for order in client.submitted] == [10] * 10
    # The immutable experiment remains complete while normal sizing returns.
    now += timedelta(days=1)
    assert risk.should_open(now, 1000, [], 'AUD_USD', 0.1)[0]
    client.recorder['closed'] = True
    assert Broker().place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert [abs(int(order['order']['units'])) for order in client.submitted] == [10] * 10 + [100]
    assert plan.status() == {'state': 'complete', 'filled_count': 10, 'pending': 0, 'units': 10}


def test_completion_between_sizing_and_reservation_cannot_submit_an_eleventh_trade(
    experiment_broker, monkeypatch
):
    broker, client, plan = experiment_broker
    for ticket in range(20, 29):
        confirm(plan, ticket)
    original_reserve = PracticeExperiment.reserve

    def competing_tenth_fill(self):
        token, _ = original_reserve(self)
        self.confirm_opening(token, '29')
        return original_reserve(self)

    monkeypatch.setattr(PracticeExperiment, 'reserve', competing_tenth_fill)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['filled_count'] == 10
    assert plan.status()['pending'] == 0


def test_reservation_write_failure_never_sends_an_order(experiment_broker, monkeypatch):
    broker, client, plan = experiment_broker
    def fail(*args):
        raise ExperimentBlocked('experiment-ledger-unavailable')
    monkeypatch.setattr(PracticeExperiment, 'reserve', fail)
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['pending'] == 0


def test_disabled_experiment_preserves_normal_sizing_and_creates_no_ledger(experiment_broker, monkeypatch):
    broker, client, plan = experiment_broker
    monkeypatch.delenv('MOSSY_PRACTICE_EXPERIMENT_ID')
    plan.path.unlink()
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'SENT'
    assert client.submitted[0]['order']['units'] == '100'
    assert not plan.path.exists()


def test_exact_size_above_cash_cap_is_blocked_before_reservation(experiment_broker):
    broker, client, plan = experiment_broker
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.02001)['status'] == 'BLOCKED'
    assert client.submitted == []
    assert plan.status()['pending'] == 0


def test_historical_trade_id_cannot_count_as_a_new_experiment_opening(experiment_broker):
    broker, client, plan = experiment_broker
    client.failure = 'historical'
    assert broker.place_order('AUD_USD', 'BUY', 100, sl_distance=0.001)['status'] == 'UNKNOWN'
    assert plan.status()['filled_count'] == 0
    assert plan.status()['pending'] == 1
    assert broker.entry_halt_reason == 'no-confirmed-trade-opening'
