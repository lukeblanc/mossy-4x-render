from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from order_fakes import confirmed_order_result

from app.broker import Broker, normalize_distance
from app.config import settings


class DummyResponse:
    def __init__(self, status_code: int = 201, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


class DummyClient:
    def __init__(self, recorder):
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, path: str, json):
        self.recorder["path"] = path
        self.recorder["payload"] = json
        self.recorder["method"] = "post"
        order = json["order"]
        units = int(order["units"])
        result = confirmed_order_result(order["instrument"], "BUY" if units > 0 else "SELL", abs(units))
        self.recorder["trade_id"] = result["response"]["orderFillTransaction"]["tradeOpened"]["tradeID"]
        self.recorder["fill_price"] = Decimal("1.2")
        self.recorder["closed"] = False
        return DummyResponse(payload=result["response"])

    def get(self, path: str, params=None):
        trade_id = self.recorder.get("trade_id", "2")
        if path.endswith("/summary"):
            return DummyResponse(
                status_code=200, payload={"account": {"currency": "AUD"}}
            )
        if path.endswith("/pricing"):
            currencies = str((params or {}).get("instruments") or "USD_AUD").split("_")
            return DummyResponse(
                status_code=200,
                payload={
                    "homeConversions": [
                        {
                            "currency": currency,
                            "accountGain": "0.01" if currency == "JPY" else "1.0",
                            "accountLoss": "0.01" if currency == "JPY" else "1.0",
                        }
                        for currency in currencies
                    ]
                },
            )
        if path.endswith("/openTrades"):
            if self.recorder.get("closed") or "trade_id" not in self.recorder:
                return DummyResponse(status_code=200, payload={"trades": []})
            order = self.recorder["payload"]["order"]
            return DummyResponse(
                status_code=200,
                payload={
                    "trades": [
                        {
                            "id": trade_id,
                            "instrument": order["instrument"],
                            "stopLossOrderID": "3",
                        }
                    ]
                },
            )
        if path.endswith(f"/trades/{trade_id}"):
            if self.recorder.get("closed"):
                return DummyResponse(
                    status_code=200,
                    payload={"trade": {"id": trade_id, "state": "CLOSED"}},
                )
            order = self.recorder["payload"]["order"]
            fill_price = self.recorder["fill_price"]
            distance = Decimal(order["stopLossOnFill"]["distance"])
            units = Decimal(order["units"])
            stop_price = fill_price - distance if units > 0 else fill_price + distance
            return DummyResponse(
                status_code=200,
                payload={
                    "trade": {
                        "id": trade_id,
                        "state": "OPEN",
                        "instrument": order["instrument"],
                        "price": str(fill_price),
                        "currentUnits": str(units),
                        "stopLossOrder": {
                            "id": "3",
                            "state": "PENDING",
                            "tradeID": trade_id,
                            "type": "STOP_LOSS",
                            "price": str(stop_price),
                        },
                    }
                },
            )
        return DummyResponse(status_code=404)

    def put(self, path: str, json):
        self.recorder["path"] = path
        self.recorder["payload"] = json
        self.recorder["method"] = "put"
        if f"/trades/{self.recorder.get('trade_id', '2')}/close" in path:
            trade_id = self.recorder.get("trade_id", "2")
            self.recorder["closed"] = True
            return DummyResponse(
                status_code=200,
                payload={
                    "orderFillTransaction": {
                        "tradesClosed": [{"tradeID": trade_id}]
                    }
                },
            )
        return DummyResponse(status_code=200)


class ErrorResponse:
    status_code = 503
    text = "service unavailable"

    @staticmethod
    def json():
        return {}


class ReadErrorClient(DummyClient):
    def get(self, path: str, params=None):
        return ErrorResponse()


def _configure_settings(monkeypatch):
    monkeypatch.setattr(settings, "OANDA_API_KEY", "token")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "acct-123")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "0.50")


def test_place_order_uses_absolute_tp_price_for_buy(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "EUR_USD",
        "BUY",
        100,
        sl_distance=0.00123,
        tp_distance=0.005,
        entry_price=1.2000,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["units"] == "100"
    assert order["stopLossOnFill"]["distance"] == "0.00123"
    assert order["takeProfitOnFill"]["price"] == "1.20500"
    assert "distance" not in order["takeProfitOnFill"]


def test_place_order_uses_absolute_tp_price_for_sell(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "EUR_USD",
        "SELL",
        500,
        sl_distance=0.00100,
        tp_distance=0.005,
        entry_price=1.2000,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["units"] == "-500"
    assert order["stopLossOnFill"]["distance"] == "0.00100"
    assert order["takeProfitOnFill"]["price"] == "1.19500"
    assert "distance" not in order["takeProfitOnFill"]


def test_stop_loss_distance_respects_instrument_precision(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "USD_JPY",
        "SELL",
        50,
        sl_distance=0.06901,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["units"] == "-50"
    assert order["stopLossOnFill"]["distance"] == "0.070"


def test_stop_distance_rounds_conservatively_before_sizing_or_submission():
    assert normalize_distance("EUR_USD", 0.001235) == "0.00124"


@pytest.mark.parametrize(
    "stop_distance",
    [None, 0, -0.001, float("nan"), float("inf"), float("-inf"), 0.000001],
)
def test_invalid_protective_stop_blocks_without_http_post(monkeypatch, stop_distance):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    result = Broker().place_order(
        "EUR_USD", "BUY", 100, sl_distance=stop_distance
    )

    assert result == {"status": "BLOCKED", "reason": "invalid-protective-stop"}
    assert recorded == {}


@pytest.mark.parametrize("units", [0, -1, 1.5, float("nan"), float("inf")])
def test_invalid_units_block_without_http_post(monkeypatch, units):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    result = Broker().place_order(
        "EUR_USD", "BUY", units, sl_distance=0.001
    )

    assert result == {"status": "BLOCKED", "reason": "invalid-order-units"}
    assert recorded == {}


class MissingStopClient(DummyClient):
    def post(self, path: str, json):
        self.recorder["posts"] = self.recorder.get("posts", 0) + 1
        response = super().post(path, json)
        response.payload.pop("stopLossOrderTransaction")
        return response

    def get(self, path: str, params=None):
        if "/trades/" in path and not self.recorder.get("closed"):
            return DummyResponse(
                status_code=200,
                payload={
                    "trade": {
                        "id": "2",
                        "state": "OPEN",
                        "instrument": "EUR_USD",
                        "price": "1.2",
                        "currentUnits": "100",
                    }
                },
            )
        return super().get(path, params=params)


def test_missing_stop_confirmation_closes_exact_trade_and_latches_halt(
    monkeypatch, tmp_path
):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: MissingStopClient(recorded))
    broker = Broker()

    result = broker.place_order("EUR_USD", "BUY", 100, sl_distance=0.001)

    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "protective-stop-not-confirmed"
    assert result["emergency_close"] == "confirmed"
    assert recorded["method"] == "put"
    assert recorded["path"] == "/v3/accounts/acct-123/trades/2/close"
    assert recorded["payload"] == {"units": "ALL"}
    assert broker.place_order(
        "EUR_USD", "BUY", 100, sl_distance=0.001
    ) == {"status": "BLOCKED", "reason": "protective-stop-not-confirmed"}
    assert recorded["posts"] == 1
    assert (tmp_path / "broker_entry_halt.txt").read_text().strip() == (
        "protective-stop-not-confirmed"
    )
    assert Broker().place_order(
        "EUR_USD", "BUY", 100, sl_distance=0.001
    ) == {"status": "BLOCKED", "reason": "protective-stop-not-confirmed"}


class CancelledCloseClient(MissingStopClient):
    def put(self, path: str, json):
        self.recorder["path"] = path
        self.recorder["payload"] = json
        self.recorder["method"] = "put"
        return DummyResponse(
            status_code=200,
            payload={"orderCancelTransaction": {"id": "9"}},
        )


def test_cancelled_emergency_close_is_never_reported_as_confirmed(
    monkeypatch, tmp_path
):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    monkeypatch.setattr(Broker, "_client", lambda self: CancelledCloseClient({}))

    result = Broker().place_order("EUR_USD", "BUY", 100, sl_distance=0.001)

    assert result["status"] == "UNKNOWN"
    assert result["emergency_close"] == "uncertain"


class TooWideStopClient(DummyClient):
    def get(self, path: str, params=None):
        response = super().get(path, params=params)
        if "/trades/" in path and not self.recorder.get("closed"):
            response.payload["trade"]["stopLossOrder"]["price"] = "1.19000"
        return response


def test_confirmed_stop_price_over_cash_limit_is_closed_and_halted(
    monkeypatch, tmp_path
):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: TooWideStopClient(recorded))

    result = Broker().place_order("EUR_USD", "BUY", 100, sl_distance=0.001)

    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "protective-stop-not-confirmed"
    assert result["emergency_close"] == "confirmed"


def test_pre_order_cash_risk_postcondition_blocks_without_post(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    result = Broker().place_order("EUR_USD", "BUY", 501, sl_distance=0.001)

    assert result == {"status": "BLOCKED", "reason": "cash-risk-limit-exceeded"}
    assert recorded.get("method") != "post"


class OversizedFillClient(DummyClient):
    def post(self, path: str, json):
        response = super().post(path, json)
        response.payload["orderFillTransaction"]["tradeOpened"]["units"] = "101"
        return response


def test_oversized_fill_closes_exact_trade_and_latches_halt(monkeypatch, tmp_path):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: OversizedFillClient(recorded))
    broker = Broker()

    result = broker.place_order("EUR_USD", "BUY", 100, sl_distance=0.001)

    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "fill-does-not-match-request"
    assert recorded["path"] == "/v3/accounts/acct-123/trades/2/close"


class UnprotectedOpenTradesClient(DummyClient):
    def get(self, path: str, params=None):
        if path.endswith("/openTrades") and not self.recorder.get("closed"):
            return DummyResponse(
                status_code=200,
                payload={"trades": [{"id": "2", "instrument": "EUR_USD"}]},
            )
        if path.endswith("/trades/2") and not self.recorder.get("closed"):
            return DummyResponse(
                status_code=200,
                payload={
                    "trade": {
                        "id": "2",
                        "state": "OPEN",
                        "instrument": "EUR_USD",
                        "price": "1.2",
                        "currentUnits": "100",
                    }
                },
            )
        return super().get(path, params=params)


def test_open_trade_audit_closes_unprotected_trade_and_blocks_snapshot(
    monkeypatch, tmp_path
):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    recorded = {"trade_id": "2", "closed": False}
    monkeypatch.setattr(
        Broker, "_client", lambda self: UnprotectedOpenTradesClient(recorded)
    )
    broker = Broker()

    assert broker.list_open_trades() is None
    assert recorded["closed"] is True
    assert broker._entry_halted_reason == "unprotected-open-trade"


class RiskyExistingStopClient(DummyClient):
    def get(self, path: str, params=None):
        if path.endswith("/openTrades") and not self.recorder.get("closed"):
            return DummyResponse(
                status_code=200,
                payload={
                    "trades": [
                        {
                            "id": "2",
                            "instrument": "EUR_USD",
                            "stopLossOrderID": "3",
                        }
                    ]
                },
            )
        if path.endswith("/trades/2") and not self.recorder.get("closed"):
            return DummyResponse(
                status_code=200,
                payload={
                    "trade": {
                        "id": "2",
                        "state": "OPEN",
                        "instrument": "EUR_USD",
                        "price": "1.20000",
                        "currentUnits": "100",
                        "stopLossOrder": {
                            "id": "3",
                            "state": "PENDING",
                            "type": "STOP_LOSS",
                            "tradeID": "2",
                            "price": "1.19000",
                        },
                    }
                },
            )
        return super().get(path, params=params)


def test_open_trade_audit_closes_existing_stop_above_cash_risk_limit(
    monkeypatch, tmp_path
):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    recorded = {"trade_id": "2", "closed": False}
    monkeypatch.setattr(
        Broker, "_client", lambda self: RiskyExistingStopClient(recorded)
    )

    assert Broker().list_open_trades() is None
    assert recorded["closed"] is True


class UnavailableStopAuditClient(DummyClient):
    def get(self, path: str, params=None):
        if path.endswith("/openTrades"):
            return DummyResponse(
                status_code=200,
                payload={
                    "trades": [
                        {
                            "id": "2",
                            "instrument": "EUR_USD",
                            "stopLossOrderID": "3",
                        }
                    ]
                },
            )
        if path.endswith("/trades/2"):
            return DummyResponse(status_code=503)
        return super().get(path, params=params)


def test_unavailable_exact_stop_audit_attempts_close_and_halts(
    monkeypatch, tmp_path
):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    recorded = {"trade_id": "2", "closed": False}
    monkeypatch.setattr(
        Broker, "_client", lambda self: UnavailableStopAuditClient(recorded)
    )
    broker = Broker()

    assert broker.list_open_trades() is None
    assert recorded["method"] == "put"
    assert broker._entry_halted_reason == "protective-stop-audit-unavailable"


def test_persisted_halt_clears_only_after_clean_broker_audit(monkeypatch, tmp_path):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    (tmp_path / "broker_entry_halt.txt").write_text("old-uncertain-order\n")
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient({}))
    broker = Broker()

    result = broker.connectivity_check()

    assert result["ok"] is True
    assert broker._entry_halted_reason is None
    assert not (tmp_path / "broker_entry_halt.txt").exists()


class WrongCurrencyClient(DummyClient):
    def get(self, path: str, params=None):
        if path.endswith("/summary"):
            return DummyResponse(
                status_code=200, payload={"account": {"currency": "NZD"}}
            )
        return super().get(path, params=params)


def test_non_aud_account_currency_latches_entry_halt(monkeypatch, tmp_path):
    _configure_settings(monkeypatch)
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    monkeypatch.setattr(Broker, "_client", lambda self: WrongCurrencyClient({}))
    broker = Broker()

    result = broker.connectivity_check()

    assert result == {
        "ok": False,
        "reason": "account-currency-mismatch",
        "currency": "NZD",
    }
    assert broker._entry_halted_reason == "account-currency-mismatch"


class HomeConversionClient(DummyClient):
    def get(self, path: str, params=None):
        if path.endswith("/summary"):
            return DummyResponse(status_code=200, payload={"account": {"currency": "AUD"}})
        if params and params.get("instruments") == "USD_AUD":
            return DummyResponse(status_code=400)
        return DummyResponse(
            status_code=200,
            payload={
                "homeConversions": [
                    {
                        "currency": "USD",
                        "accountGain": "1.49",
                        "accountLoss": "1.51",
                    }
                ]
            },
        )


def test_conversion_rate_uses_conservative_account_loss_factor(monkeypatch):
    _configure_settings(monkeypatch)
    monkeypatch.setattr(Broker, "_client", lambda self: HomeConversionClient({}))

    assert Broker().conversion_rate("USD", "AUD") == 1.51


def test_usd_jpy_tp_price_is_rounded(monkeypatch, capsys):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "USD_JPY",
        "BUY",
        100,
        sl_distance=0.123,
        tp_distance=0.00432,
        entry_price=156.16487,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["takeProfitOnFill"]["price"] == "156.169"

    logs = capsys.readouterr().out
    assert "[ORDER_FMT] instrument=USD_JPY raw_tp=156.16919 rounded_tp=156.169" in logs


def test_close_position_side_uses_put(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.close_position_side("EUR_USD", long_units=1, short_units=0)

    assert result["status"] == "CLOSED"
    assert recorded["method"] == "put"
    assert recorded["path"] == "/v3/accounts/acct-123/positions/EUR_USD/close"
    assert recorded["payload"] == {"longUnits": "ALL"}


def test_failed_broker_reads_are_not_reported_as_zero_or_no_positions(monkeypatch):
    _configure_settings(monkeypatch)
    monkeypatch.setattr(Broker, "_client", lambda self: ReadErrorClient({}))
    broker = Broker()

    assert broker.list_open_trades() is None
    assert broker.account_equity() is None
    assert broker.current_spread("EUR_USD") is None
    assert broker.get_unrealized_profit("EUR_USD") is None
    assert broker.position_snapshot("EUR_USD") is None
