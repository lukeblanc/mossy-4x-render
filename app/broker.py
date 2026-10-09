from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import time
from typing import Dict, Literal, Optional

import httpx

from app.config import settings
from src.cash_risk import configured_cash_risk_limit
from src.practice_experiment import ExperimentBlocked, PracticeExperiment

PRACTICE = "https://api-fxpractice.oanda.com"
LIVE = "https://api-fxtrade.oanda.com"


def read_trade_details(client, account: str, trade_id: str) -> Optional[Dict]:
    """Read one exact trade, with a bounded list fallback for detail 404s."""
    ticket = str(trade_id)
    if not ticket.isascii() or not ticket.isdigit() or int(ticket) <= 0:
        return None
    response = client.get(f"/v3/accounts/{account}/trades/{ticket}")
    if response.status_code == 404:
        # Explicit IDs and ALL include closed trades without scanning history.
        response = client.get(f"/v3/accounts/{account}/trades",
                              params={"ids": ticket, "state": "ALL", "count": 1})
        if response.status_code != 200:
            raise RuntimeError(f"broker trade list unavailable: HTTP {response.status_code}")
        payload = response.json()
        candidates = payload.get("trades") if isinstance(payload, dict) else None
        if candidates == []:
            from app.trade_transactions import read_full_close
            return read_full_close(client, account, ticket)
        if not isinstance(candidates, list) or len(candidates) != 1:
            print(f"[JOURNAL][LOOKUP] trade_id={ticket} exact_list_match=False", flush=True)
            return None
        trade = candidates[0]
        source = "exact-list"
    else:
        if response.status_code != 200:
            raise RuntimeError(f"broker trade details unavailable: HTTP {response.status_code}")
        payload = response.json()
        trade = payload.get("trade") if isinstance(payload, dict) else None
        source = "detail"
    if not isinstance(trade, dict) or str(trade.get("id")) != ticket:
        return None
    if source == "exact-list":
        print(f"[JOURNAL][LOOKUP] trade_id={ticket} source={source} state={trade.get('state')}", flush=True)
    return trade


def opened_trade_fill(payload: object) -> Optional[Dict]:
    """Return only a verified new trade, never an order or a reduced/closed trade."""
    if not isinstance(payload, dict):
        return None
    fill = payload.get("orderFillTransaction")
    if not isinstance(fill, dict):
        return None
    opened = fill.get("tradeOpened")
    if not isinstance(opened, dict):
        return None
    trade_id = str(opened.get("tradeID") or "")
    try:
        price = float(opened["price"])
        units = float(opened["units"])
        timestamp = datetime.fromisoformat(str(fill["time"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not trade_id.isascii() or not trade_id.isdigit() or int(trade_id) <= 0
            or not math.isfinite(price) or price <= 0
            or not math.isfinite(units) or units == 0 or timestamp.tzinfo is None):
        return None
    return {"trade_id": trade_id, "price": price, "units": units,
            "timestamp": timestamp.astimezone(timezone.utc)}


def _precision_for(instrument: str) -> Decimal:
    return {
        "USD_JPY": Decimal("0.001"),
        "XAU_USD": Decimal("0.01"),
    }.get(instrument, Decimal("0.00001"))


def _quantize_value(
    value: float | Decimal | str,
    precision: Decimal,
    *,
    rounding: str = ROUND_HALF_UP,
) -> Decimal:
    try:
        dec_value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid numeric value for normalization: {value}")
    if not dec_value.is_finite():
        raise ValueError(f"Non-finite numeric value for normalization: {value}")
    try:
        return dec_value.quantize(precision, rounding=rounding)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value for normalization: {value}") from exc


def normalize_price(instrument: str, price: float | Decimal | str) -> str:
    precision = _precision_for(instrument)
    dec_price = _quantize_value(price, precision)

    return str(dec_price)


def normalize_distance(instrument: str, distance: float | Decimal | str) -> str:
    precision = _precision_for(instrument)
    try:
        raw_distance = Decimal(str(distance))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid protective-stop distance: {distance}") from exc
    if not raw_distance.is_finite() or raw_distance <= 0:
        raise ValueError(f"Invalid protective-stop distance: {distance}")
    if raw_distance < precision:
        raise ValueError(
            f"Protective-stop distance is below {instrument} precision: {distance}"
        )
    # Preserve at least the strategy's requested technical distance, then size
    # from this exact rounded-up value before submitting it to the broker.
    dec_distance = _quantize_value(distance, precision, rounding=ROUND_CEILING)

    return str(dec_distance)


def _valid_transaction_id(value: object) -> bool:
    ticket = str(value or "")
    return ticket.isascii() and ticket.isdigit() and int(ticket) > 0


def _entry_halt_path() -> Path:
    configured = os.getenv("MOSSY_STATE_PATH")
    if configured:
        root = Path(configured)
    elif Path("/var/data").exists():
        root = Path("/var/data")
    else:
        root = Path("data")
    return root / "broker_entry_halt.txt"


class Broker:
    def __init__(self):
        # Guard against accidental live usage
        env_label = (getattr(settings, "OANDA_ENV", "practice") or "practice").lower()
        self.mode = (settings.MODE or "demo").lower()
        if env_label == "live" or self.mode == "live":
            print("[OANDA] Live trading is disabled in this deployment. Exiting.", flush=True)
            raise SystemExit(1)

        self.account = settings.OANDA_ACCOUNT_ID
        self.key = settings.OANDA_API_KEY
        if env_label == "practice" or self.mode == "demo":
            self.base_url = PRACTICE
        elif env_label == "live" or self.mode == "live":
            self.base_url = LIVE
        else:
            self.base_url = PRACTICE
        self._headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        self._entry_halt_path = _entry_halt_path()
        try:
            persisted_halt = self._entry_halt_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            persisted_halt = ""
        self._entry_halted_reason: Optional[str] = persisted_halt or None
        self._account_currency: Optional[str] = None
        self._loss_conversion_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, headers=self._headers, timeout=15.0)

    @property
    def entry_halt_reason(self) -> Optional[str]:
        """Observe the latch without broker calls, recovery, or other side effects."""
        return self._entry_halted_reason

    def _latch_entry_halt(self, reason: str) -> None:
        self._entry_halted_reason = reason
        try:
            self._entry_halt_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._entry_halt_path.with_suffix(".tmp")
            temporary.write_text(f"{reason}\n", encoding="utf-8")
            os.replace(temporary, self._entry_halt_path)
        except OSError as exc:
            print(f"[BROKER][CRITICAL] Could not persist entry halt: {exc}", flush=True)

    def trade_details(self, trade_id: str) -> Optional[Dict]:
        if not self.key or not self.account:
            raise RuntimeError("broker credentials unavailable")
        with self._client() as client:
            return read_trade_details(client, self.account, trade_id)

    def connectivity_check(self) -> dict:
        """Startup reconciliation; may close unsafe trades but never clears a halt.

        This is not a read-only health check. Normal monitoring must use the
        side-effect-free entry_halt_reason property to observe the latch.
        """
        if not (self.key and self.account):
            print("[OANDA] No credentials set; skipping connectivity check.")
            return {"ok": False, "reason": "no-creds"}
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/summary")
                if resp.status_code == 200:
                    data = resp.json().get("account", {})
                    balance = data.get("balance")
                    currency = data.get("currency")
                    self._account_currency = str(currency or "").upper() or None
                    if self._account_currency != "AUD":
                        self._latch_entry_halt("account-currency-mismatch")
                        print(
                            "[OANDA][CRITICAL] Expected AUD account currency; "
                            f"received {self._account_currency or 'unknown'}",
                            flush=True,
                        )
                        return {
                            "ok": False,
                            "reason": "account-currency-mismatch",
                            "currency": currency,
                        }
                    print(
                        f"[OANDA] Connected ok. Balance={balance} {currency} (mode={self.mode})",
                        flush=True,
                    )
                    if self._entry_halted_reason:
                        trades = self._read_open_trade_summaries(client)
                        if trades is not None:
                            self._audit_open_trade_protection(client, trades)
                        # A clean snapshot cannot authorize recovery of a
                        # persisted incident; approval is separate from startup.
                    return {"ok": True, "balance": balance, "currency": currency}
                print(
                    f"[OANDA] Connectivity error {resp.status_code}: {resp.text}",
                    flush=True,
                )
                return {"ok": False, "status": resp.status_code, "text": resp.text}
        except Exception as exc:
            print(f"[OANDA] Connectivity exception: {exc}", flush=True)
            return {"ok": False, "error": str(exc)}

    def _read_open_trade_summaries(self, client) -> Optional[list]:
        try:
            response = client.get(f"/v3/accounts/{self.account}/openTrades")
            if response.status_code != 200:
                return None
            payload = response.json()
            trades = payload.get("trades") if isinstance(payload, dict) else None
            if not isinstance(trades, list) or any(
                not isinstance(trade, dict)
                or not _valid_transaction_id(trade.get("id"))
                or not trade.get("instrument")
                for trade in trades
            ):
                return None
            return trades
        except Exception:
            return None

    def _open_trade_protection_status(
        self, client, summary: dict
    ) -> bool | Literal["closed"] | None:
        """Return True for protection, "closed" for closure, False/None for failure.

        The historical halt category is retained for compatibility. This audit
        logs the specific failed condition so an over-budget stop is never
        mistaken for evidence that the broker lost or removed a stop order.
        Only whitelisted diagnostic fields are logged, never raw responses.
        """

        trade_id = str(summary.get("id") or "")

        def failure(reason: str, *, unknown: bool = False, **evidence):
            details = {"trade_id": trade_id, "audit_reason": reason,
                       "status": "unknown" if unknown else "unsafe", **evidence}
            print("[BROKER][PROTECTION-AUDIT] " + json.dumps(details, sort_keys=True),
                  flush=True)
            return None if unknown else False

        try:
            trade = read_trade_details(client, self.account, trade_id)
        except Exception:
            return failure("trade-details-unavailable", unknown=True)
        if not isinstance(trade, dict):
            return failure("trade-details-unverified", unknown=True)
        if str(trade.get("id") or "") != trade_id:
            return failure("trade-id-mismatch")
        state = str(trade.get("state") or "").upper()
        if state == "CLOSED":
            # A terminal state must not override contradictory supplied fields.
            if ("instrument" in trade
                    and trade["instrument"] != summary.get("instrument")):
                return failure("closed-trade-evidence-inconsistent", unknown=True)
            if "currentUnits" in trade:
                try:
                    remaining = Decimal(str(trade["currentUnits"]))
                except (InvalidOperation, TypeError, ValueError):
                    return failure("closed-trade-evidence-inconsistent", unknown=True)
                if not remaining.is_finite() or remaining != 0:
                    return failure("closed-trade-evidence-inconsistent", unknown=True)
            print("[BROKER][PROTECTION-AUDIT] " + json.dumps({
                "trade_id": trade_id, "audit_reason": "trade-closed-during-audit",
                "status": "closed",
            }, sort_keys=True), flush=True)
            return "closed"
        if state != "OPEN":
            return failure("trade-state-unverified", unknown=True)
        if str(trade.get("instrument") or "") != str(summary.get("instrument") or ""):
            return failure("trade-instrument-mismatch")
        stop_order = trade.get("stopLossOrder") or trade.get("guaranteedStopLossOrder")
        if not isinstance(stop_order, dict):
            return failure("stop-order-missing")
        stop_id = str(stop_order.get("id") or "")
        summary_stop_id = str(
            summary.get("stopLossOrderID")
            or summary.get("guaranteedStopLossOrderID")
            or ""
        )
        if not _valid_transaction_id(stop_id):
            return failure("stop-id-invalid")
        if summary_stop_id and summary_stop_id != stop_id:
            return failure("stop-id-mismatch", stop_order_id=stop_id)
        if str(stop_order.get("state") or "").upper() != "PENDING":
            return failure("stop-not-pending", stop_order_id=stop_id)
        if str(stop_order.get("type") or "").upper() not in {
            "STOP_LOSS", "GUARANTEED_STOP_LOSS"
        }:
            return failure("stop-type-invalid", stop_order_id=stop_id)
        if str(stop_order.get("tradeID") or "") != trade_id:
            return failure("stop-trade-id-mismatch", stop_order_id=stop_id)
        try:
            units = Decimal(str(trade.get("currentUnits")))
            entry_price = Decimal(str(trade.get("price")))
            stop_price = Decimal(str(stop_order.get("price")))
        except (InvalidOperation, TypeError, ValueError):
            return failure("stop-risk-input-invalid", stop_order_id=stop_id)
        if (
            not units.is_finite()
            or units == 0
            or not entry_price.is_finite()
            or entry_price <= 0
            or not stop_price.is_finite()
            or stop_price <= 0
        ):
            return failure("stop-risk-input-invalid", stop_order_id=stop_id)
        try:
            _, quote_currency = str(trade["instrument"]).upper().split("_", 1)
        except (KeyError, ValueError):
            return failure("trade-instrument-invalid")
        conversion = self.conversion_rate(quote_currency, "AUD")
        cash_limit = configured_cash_risk_limit()
        if conversion is None or not math.isfinite(conversion) or conversion <= 0:
            return failure("loss-conversion-unavailable", unknown=True)
        if cash_limit <= 0:
            return failure("cash-risk-limit-invalid", unknown=True)
        loss_distance = (
            max(Decimal("0"), entry_price - stop_price)
            if units > 0
            else max(Decimal("0"), stop_price - entry_price)
        )
        risk = abs(units) * loss_distance * Decimal(str(conversion))
        if not risk.is_finite() or risk > Decimal(str(cash_limit)):
            return failure(
                "stop-cash-risk-exceeded", stop_order_id=stop_id,
                units=str(units), entry_price=str(entry_price), stop_price=str(stop_price),
                loss_conversion=str(conversion), planned_stop_risk=str(risk),
                cash_limit=str(cash_limit), currency="AUD",
            )
        return True

    def _close_exact_trade(self, client, trade_id: str) -> bool:
        if not _valid_transaction_id(trade_id):
            return False
        try:
            close_response = client.put(
                f"/v3/accounts/{self.account}/trades/{trade_id}/close",
                json={"units": "ALL"},
            )
            if close_response.status_code not in (200, 201):
                return False
            payload = close_response.json()
            fill = payload.get("orderFillTransaction") if isinstance(payload, dict) else None
            closed = fill.get("tradesClosed") if isinstance(fill, dict) else None
            if not isinstance(closed, list) or not any(
                isinstance(item, dict) and str(item.get("tradeID") or "") == trade_id
                for item in closed
            ):
                return False
            trade = read_trade_details(client, self.account, trade_id)
            return (
                isinstance(trade, dict)
                and str(trade.get("id") or "") == trade_id
                and str(trade.get("state") or "").upper() == "CLOSED"
            )
        except Exception:
            return False

    def _audit_open_trade_protection(self, client, trades: list) -> Optional[list]:
        """Return the audited snapshot, refreshing once after verified closure.

        Unsafe/unknown evidence retains the halt and emergency-close behavior.
        A closed trade needs no close attempt, but invalidates the list snapshot.
        Never return that stale list or clear an existing entry halt.
        """

        found_issue = False
        closed_ids: set[str] = set()
        for attempt in range(2):
            refresh_needed = False
            for trade in trades:
                trade_id = str(trade.get("id") or "")
                # A confirmed closed ID still in the refreshed list is
                # inconsistent evidence, even if a later detail says OPEN.
                if trade_id in closed_ids:
                    refresh_needed = True
                    continue
                status = self._open_trade_protection_status(client, trade)
                if status == "closed":
                    closed_ids.add(trade_id)
                    refresh_needed = True
                    continue
                if status is True:
                    continue
                found_issue = True
                if status is None:
                    self._latch_entry_halt("protective-stop-audit-unavailable")
                    close_ok = self._close_exact_trade(client, trade_id)
                    print(
                        "[BROKER][CRITICAL] protective-stop audit unavailable; "
                        f"trade_id={trade_id} emergency_close={close_ok} "
                        "new_entries_halted=true",
                        flush=True,
                    )
                    continue
                self._latch_entry_halt("unprotected-open-trade")
                close_ok = self._close_exact_trade(client, trade_id)
                print(
                    "[BROKER][CRITICAL] open-trade protection audit failed; "
                    f"trade_id={trade_id} emergency_close={close_ok} "
                    "new_entries_halted=true",
                    flush=True,
                )
            if not refresh_needed:
                return None if found_issue else trades
            refreshed = self._read_open_trade_summaries(client) if attempt == 0 else None
            if refreshed is None:
                self._latch_entry_halt(
                    self.entry_halt_reason or "protective-stop-audit-unavailable"
                )
                print("[BROKER][PROTECTION-AUDIT] " + json.dumps({
                    "audit_reason": ("open-trades-refresh-unavailable" if attempt == 0
                                     else "open-trades-refresh-inconsistent"),
                    "status": "unknown",
                }, sort_keys=True), flush=True)
                return None
            trades = refreshed
        return None

    def _halt_unknown_order_state(self, client, reason: str, response: dict) -> dict:
        self._latch_entry_halt(reason)
        trades = self._read_open_trade_summaries(client)
        audited = trades is not None
        if trades is not None:
            self._audit_open_trade_protection(client, trades)
        return {
            "status": "UNKNOWN",
            "reason": reason,
            "open_trade_audit": "completed" if audited else "unavailable",
            "response": response,
        }

    def _halt_and_close_trade(
        self,
        client,
        *,
        trade_id: str,
        reason: str,
        response: dict,
    ) -> dict:
        """Close one uncertain fill and persist an entry halt."""

        self._latch_entry_halt(reason)
        close_ok = self._close_exact_trade(client, trade_id)
        print(
            f"[BROKER][CRITICAL] {reason}; trade_id={trade_id} "
            f"emergency_close={close_ok} new_entries_halted=true",
            flush=True,
        )
        return {
            "status": "UNKNOWN",
            "reason": reason,
            "emergency_close": "confirmed" if close_ok else "uncertain",
            "response": response,
        }

    def _experiment_preflight(self, client, units: int, previous_trade: str | None) -> str | None:
        """Fresh account-specific checks, after durable reservation, before POST."""
        try:
            response = client.get(f"/v3/accounts/{self.account}/summary")
            if response.status_code != 200:
                return None
            account = response.json().get("account")
            if (not isinstance(account, dict) or account.get("id") != self.account
                    or account.get("currency") != "AUD"
                    or any(type(account.get(key)) is not int or account[key] != 0
                           for key in ("openTradeCount", "openPositionCount", "pendingOrderCount"))):
                return None
            watermark = str(account.get("lastTransactionID") or "")
            if not _valid_transaction_id(watermark):
                return None
            if self._read_open_trade_summaries(client) != []:
                return None
            if previous_trade and self._open_trade_protection_status(
                client, {"id": previous_trade, "instrument": "AUD_USD"}
            ) != "closed":
                # Even two stale flat lists cannot authorize overlap with our
                # last known opening: require exact positive closure evidence.
                return None
            response = client.get(f"/v3/accounts/{self.account}/instruments",
                                  params={"instruments": "AUD_USD"})
            if response.status_code != 200:
                return None
            instruments = response.json().get("instruments")
            if not isinstance(instruments, list) or len(instruments) != 1:
                return None
            specification = instruments[0]
            if (not isinstance(specification, dict) or specification.get("name") != "AUD_USD"
                    or specification.get("type") != "CURRENCY"):
                return None
            precision = specification.get("tradeUnitsPrecision")
            minimum = Decimal(str(specification.get("minimumTradeSize")))
            if (type(precision) is int and 0 <= precision <= 8
                    and minimum.is_finite() and 0 < minimum <= units):
                return watermark
            return None
        except Exception:
            return None

    def place_order(
        self,
        instrument: str,
        signal: str,
        units: float,
        *,
        sl_distance: float | None = None,
        tp_distance: float | None = None,
        entry_price: float | None = None,
    ) -> dict:
        side = signal.upper()
        if side not in ("BUY", "SELL"):
            print(f"[BROKER] Ignoring unknown signal: {signal}", flush=True)
            return {"status": "IGNORED", "reason": "unknown-signal"}

        if self._entry_halted_reason:
            print(
                f"[BROKER][HALT] Entry blocked: {self._entry_halted_reason}",
                flush=True,
            )
            return {"status": "BLOCKED", "reason": self._entry_halted_reason}

        try:
            units_value = float(units)
        except (TypeError, ValueError, OverflowError):
            units_value = 0.0
        if (
            not math.isfinite(units_value)
            or units_value <= 0
            or not units_value.is_integer()
        ):
            print(f"[BROKER][BLOCK] Invalid order units: {units}", flush=True)
            return {"status": "BLOCKED", "reason": "invalid-order-units"}

        experiment = None
        reservation = None
        try:
            experiment = PracticeExperiment.configured(self._entry_halt_path.parent, self.account)
            if experiment is not None:
                if (self.mode != "demo" or self.base_url != PRACTICE
                        or str(settings.OANDA_ENV).lower() != "practice"
                        or instrument != "AUD_USD"):
                    raise ExperimentBlocked("experiment-practice-audusd-only")
                target_units = experiment.ready_units()
                if units_value < target_units:
                    raise ExperimentBlocked("experiment-exact-size-exceeds-strategy-budget")
                hard_loss = Decimal(os.getenv("HARD_MAX_LOSS_CCY", "NaN"))
                if (not hard_loss.is_finite() or not 0 < hard_loss <= Decimal("0.20")
                        or not 0 < configured_cash_risk_limit() <= 0.20):
                    raise ExperimentBlocked("experiment-twenty-cent-limits-required")
                # AUD is the base AND account currency: these whole units are
                # exact A$ notional, not margin or the allowed cash loss.
                units_value = float(target_units)
        except (ExperimentBlocked, InvalidOperation) as exc:
            reason = str(exc) if isinstance(exc, ExperimentBlocked) else "experiment-limits-invalid"
            return {"status": "BLOCKED", "reason": reason}

        try:
            normalized_sl_distance = normalize_distance(instrument, sl_distance)
        except (TypeError, ValueError, InvalidOperation):
            print(
                f"[BROKER][BLOCK] {instrument} {side} missing a valid protective stop",
                flush=True,
            )
            return {"status": "BLOCKED", "reason": "invalid-protective-stop"}

        if self.mode == "simulation":
            print(
                f"[BROKER] {self.mode.upper()} SIMULATED {side} order for {instrument} size={units}",
                flush=True,
            )
            return {"status": "SIMULATED"}

        if not (self.key and self.account):
            print(
                f"[BROKER] {self.mode.upper()} order failed: missing credentials.",
                flush=True,
            )
            return {"status": "ERROR", "reason": "missing-creds"}

        try:
            _, quote_currency = instrument.upper().split("_", 1)
        except ValueError:
            return {"status": "BLOCKED", "reason": "invalid-instrument"}
        cash_risk_limit = configured_cash_risk_limit()
        loss_conversion = self.conversion_rate(quote_currency, "AUD")
        if cash_risk_limit <= 0:
            return {"status": "BLOCKED", "reason": "invalid-cash-risk-cap"}
        if loss_conversion is None:
            return {"status": "BLOCKED", "reason": "loss-conversion-unavailable"}
        try:
            planned_stop_risk = (
                Decimal(str(units_value))
                * Decimal(normalized_sl_distance)
                * Decimal(str(loss_conversion))
            )
            cash_limit_decimal = Decimal(str(cash_risk_limit))
        except InvalidOperation:
            return {"status": "BLOCKED", "reason": "invalid-planned-stop-risk"}
        if (
            not planned_stop_risk.is_finite()
            or planned_stop_risk <= 0
            or planned_stop_risk > cash_limit_decimal
        ):
            print(
                f"[BROKER][BLOCK] planned_stop_risk={planned_stop_risk} "
                f"limit={cash_limit_decimal} {instrument}",
                flush=True,
            )
            return {"status": "BLOCKED", "reason": "cash-risk-limit-exceeded"}

        trade_units = int(units_value if side == "BUY" else -units_value)
        order_payload = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(trade_units),
            "stopLossOnFill": {
                "timeInForce": "GTC",
                "distance": normalized_sl_distance,
            },
        }
        if (
            entry_price is not None
            and tp_distance is not None
            and tp_distance > 0
        ):
            try:
                entry_val = Decimal(str(entry_price))
                tp_val = Decimal(str(tp_distance))
            except (TypeError, ValueError, InvalidOperation):
                entry_val = None
                tp_val = None
            if (
                entry_val is not None
                and tp_val is not None
                and entry_val.is_finite()
                and tp_val.is_finite()
                and entry_val > 0
                and tp_val > 0
            ):
                if side == "BUY":
                    tp_price = entry_val + tp_val
                else:
                    tp_price = entry_val - tp_val
                rounded_tp = normalize_price(instrument, tp_price)
                print(
                    f"[ORDER_FMT] instrument={instrument} raw_tp={tp_price} rounded_tp={rounded_tp}",
                    flush=True,
                )
                order_payload["takeProfitOnFill"] = {
                    "timeInForce": "GTC",
                    "price": rounded_tp,
                }

        payload = {"order": order_payload}

        try:
            with self._client() as client:
                if experiment is not None:
                    try:
                        reservation, reserved_units = experiment.reserve()
                        watermark = (self._experiment_preflight(
                            client, reserved_units, experiment.last_opened_trade()
                        ) if reserved_units == abs(trade_units) else None)
                        if watermark is None:
                            experiment.skip_before_submission(reservation)
                            return {"status": "BLOCKED", "reason": "experiment-preflight-failed"}
                    except ExperimentBlocked as exc:
                        return {"status": "BLOCKED", "reason": str(exc)}
                resp = client.post(f"/v3/accounts/{self.account}/orders", json=payload)
                if resp.status_code in (200, 201):
                    data = resp.json()
                    if not isinstance(data, dict):
                        return self._halt_unknown_order_state(
                            client, "invalid-order-response", {}
                        )
                    if data.get("orderCancelTransaction"):
                        return {"status": "CANCELLED", "response": data}
                    if data.get("orderRejectTransaction"):
                        return {"status": "REJECTED", "response": data}
                    filled = opened_trade_fill(data)
                    if filled is None:
                        return self._halt_unknown_order_state(
                            client, "no-confirmed-trade-opening", data
                        )
                    if experiment is not None:
                        if int(filled["trade_id"]) <= int(watermark):
                            return self._halt_unknown_order_state(
                                client, "no-confirmed-trade-opening", data
                            )
                        try:
                            # Count even a subsequently emergency-closed fill.
                            experiment.confirm_opening(reservation, filled["trade_id"])
                        except ExperimentBlocked:
                            return self._halt_and_close_trade(
                                client, trade_id=filled["trade_id"],
                                reason="order-transport-state-uncertain", response=data,
                            )
                    if (
                        data["orderFillTransaction"].get("instrument") != instrument
                        or filled["units"] * trade_units <= 0
                        or abs(filled["units"]) > abs(trade_units)
                        or (experiment is not None and abs(filled["units"]) != abs(trade_units))
                    ):
                        return self._halt_and_close_trade(
                            client,
                            trade_id=filled["trade_id"],
                            reason="fill-does-not-match-request",
                            response=data,
                        )
                    try:
                        trade = read_trade_details(
                            client, self.account, filled["trade_id"]
                        )
                    except Exception:
                        trade = None
                    stop_order = (
                        (
                            trade.get("stopLossOrder")
                            or trade.get("guaranteedStopLossOrder")
                        )
                        if isinstance(trade, dict)
                        else None
                    )
                    stop_trade_id = (
                        str(stop_order.get("tradeID") or "")
                        if isinstance(stop_order, dict)
                        else ""
                    )
                    stop_type = (
                        str(stop_order.get("type") or "").upper()
                        if isinstance(stop_order, dict)
                        else ""
                    )
                    try:
                        current_units = Decimal(str(trade.get("currentUnits")))
                        trade_price = Decimal(str(trade.get("price")))
                        stop_price = Decimal(str(stop_order.get("price")))
                        submitted_distance = Decimal(normalized_sl_distance)
                        conversion_decimal = Decimal(str(loss_conversion))
                    except (AttributeError, InvalidOperation, TypeError, ValueError):
                        current_units = Decimal("NaN")
                        trade_price = Decimal("NaN")
                        stop_price = Decimal("NaN")
                        submitted_distance = Decimal("NaN")
                        conversion_decimal = Decimal("NaN")
                    numeric_stop_fields_valid = all(
                        value.is_finite()
                        for value in (
                            current_units,
                            trade_price,
                            stop_price,
                            submitted_distance,
                            conversion_decimal,
                        )
                    )
                    if numeric_stop_fields_valid:
                        actual_distance = abs(trade_price - stop_price)
                        actual_stop_risk = (
                            abs(current_units) * actual_distance * conversion_decimal
                        )
                        stop_direction_ok = (
                            (side == "BUY" and stop_price < trade_price)
                            or (side == "SELL" and stop_price > trade_price)
                        )
                    else:
                        actual_distance = Decimal("NaN")
                        actual_stop_risk = Decimal("NaN")
                        stop_direction_ok = False
                    if (
                        not isinstance(trade, dict)
                        or str(trade.get("id") or "") != filled["trade_id"]
                        or str(trade.get("state") or "").upper() != "OPEN"
                        or str(trade.get("instrument") or "") != instrument
                        or not current_units.is_finite()
                        or current_units == 0
                        or current_units * Decimal(str(trade_units)) <= 0
                        or abs(current_units) > abs(Decimal(str(trade_units)))
                        or not isinstance(stop_order, dict)
                        or not _valid_transaction_id(stop_order.get("id"))
                        or str(stop_order.get("state") or "").upper() != "PENDING"
                        or stop_type not in {"STOP_LOSS", "GUARANTEED_STOP_LOSS"}
                        or stop_trade_id != filled["trade_id"]
                        or not trade_price.is_finite()
                        or not stop_price.is_finite()
                        or trade_price <= 0
                        or stop_price <= 0
                        or not stop_direction_ok
                        or not actual_stop_risk.is_finite()
                        or actual_stop_risk <= 0
                        or actual_stop_risk > cash_limit_decimal
                        or actual_distance > submitted_distance + _precision_for(instrument)
                    ):
                        return self._halt_and_close_trade(
                            client,
                            trade_id=filled["trade_id"],
                            reason="protective-stop-not-confirmed",
                            response=data,
                        )
                    print(
                        f"[BROKER][STOP-VERIFIED] trade_id={filled['trade_id']} "
                        f"stop_order_id={stop_order['id']} "
                        f"planned_stop_risk={actual_stop_risk:.5f} AUD",
                        flush=True,
                    )
                    print(f"[OANDA] DEMO TRADE OPENED trade_id={filled['trade_id']} "
                          f"price={filled['price']} units={filled['units']}", flush=True)
                    return {"status": "SENT", "response": data}
                if self.mode == "demo":
                    print(f"[OANDA] DEMO ORDER FAILED {resp.text}", flush=True)
                else:
                    print(
                        f"[BROKER] LIVE order error {resp.status_code}: {resp.text}",
                        flush=True,
                    )
                if resp.status_code >= 500:
                    result = self._halt_unknown_order_state(
                        client, "order-http-state-uncertain", {}
                    )
                    result.update({"code": resp.status_code, "text": resp.text})
                    return result
                return {"status": "ERROR", "code": resp.status_code, "text": resp.text}
        except Exception as exc:
            if self.mode == "demo":
                print(f"[OANDA] DEMO ORDER FAILED {exc}", flush=True)
            else:
                print(f"[BROKER] LIVE order exception: {exc}", flush=True)
            self._latch_entry_halt("order-transport-state-uncertain")
            audit_status = "unavailable"
            try:
                with self._client() as audit_client:
                    trades = self._read_open_trade_summaries(audit_client)
                    if trades is not None:
                        self._audit_open_trade_protection(audit_client, trades)
                        audit_status = "completed"
            except Exception:
                pass
            return {
                "status": "UNKNOWN",
                "reason": "order-transport-state-uncertain",
                "open_trade_audit": audit_status,
                "error": str(exc),
            }

    def list_open_trades(self) -> Optional[list]:
        """Return currently open trades, or ``None`` when the broker cannot be read.

        An empty list is a valid broker answer.  It must not also be used as the
        error value because callers use this result to prevent duplicate exposure.
        """
        if self.mode == "simulation":
            return []
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                trades = self._read_open_trade_summaries(client)
                if trades is None:
                    return None
                return self._audit_open_trade_protection(client, trades)
        except Exception as exc:
            print(f"[OANDA] Exception fetching open trades: {exc}", flush=True)
        return None

    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        """Return the unrealized P/L for the given instrument in account currency."""
        if not instrument:
            return None
        if self.mode == "simulation":
            return 0.0
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/positions/{instrument}"
                )
                if resp.status_code != 200:
                    return None
                position = resp.json().get("position", {}) or {}
                unrealized = position.get("unrealizedPL")
                if unrealized is not None:
                    try:
                        value = float(unrealized)
                        return value if math.isfinite(value) else None
                    except (TypeError, ValueError):
                        return None
                total = 0.0
                found = False
                for side in ("long", "short"):
                    side_pl = (position.get(side) or {}).get("unrealizedPL")
                    try:
                        total += float(side_pl)
                        found = True
                    except (TypeError, ValueError):
                        continue
                return total if found and math.isfinite(total) else None
        except Exception as exc:
            print(
                f"[OANDA] Exception fetching unrealized P/L for {instrument}: {exc}",
                flush=True,
            )
            return None

    def position_snapshot(self, instrument: str) -> Optional[Dict]:
        """Return the broker position payload for the instrument, or None on error."""

        if not instrument:
            return None
        if self.mode == "simulation":
            return {
                "instrument": instrument,
                "long": {"units": "0"},
                "short": {"units": "0"},
                "longUnits": "0",
                "shortUnits": "0",
            }
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/positions/{instrument}")
                if resp.status_code != 200:
                    return None
                position = resp.json().get("position")
                if not isinstance(position, dict):
                    return None
                for side in ("long", "short"):
                    if not math.isfinite(float(position[side]["units"])):
                        return None
                return position
        except Exception:
            return None

    def close_position_side(self, instrument: str, long_units: float, short_units: float) -> Dict:
        """Close a position using side-specific payloads for the OANDA positions close endpoint."""

        if not instrument:
            return {"status": "ERROR", "reason": "invalid-instrument"}

        if long_units > 0 and short_units == 0:
            payload: Dict[str, str] = {"longUnits": "ALL"}
        elif short_units < 0 and long_units == 0:
            payload = {"shortUnits": "ALL"}
        elif long_units != 0 or short_units != 0:
            payload = {"longUnits": "ALL", "shortUnits": "ALL"}
        else:
            payload = {"longUnits": "0", "shortUnits": "0"}

        if self.mode == "simulation":
            print(f"[BROKER] SIMULATION close position {instrument} payload={payload}", flush=True)
            return {"status": "SIMULATED", "payload": payload}
        if not (self.key and self.account):
            print(
                f"[BROKER] {self.mode.upper()} close failed: missing credentials.",
                flush=True,
            )
            return {"status": "ERROR", "reason": "missing-creds"}

        try:
            with self._client() as client:
                resp = client.put(
                    f"/v3/accounts/{self.account}/positions/{instrument}/close",
                    json=payload,
                )
                if resp.status_code in (200, 201):
                    print(f"[OANDA] Closed position {instrument} payload={payload}", flush=True)
                    return {"status": "CLOSED", "response": resp.json()}
                print(
                    f"[OANDA] Failed to close {instrument} status={resp.status_code} body={resp.text}",
                    flush=True,
                )
                return {
                    "status": "ERROR",
                    "code": resp.status_code,
                    "text": resp.text,
                }
        except Exception as exc:
            print(
                f"[OANDA] Exception closing position {instrument}: {exc}", flush=True
            )
            return {"status": "ERROR", "error": str(exc)}

    # Backwards-compatible wrapper.
    def close_position(
        self,
        instrument: str,
        *,
        long_units: str | None = "ALL",
        short_units: str | None = "ALL",
        trade_id: str | None = None,
    ) -> Dict:
        try:
            long_val = 0.0 if long_units is None else float(long_units)
        except (TypeError, ValueError):
            long_val = 0.0
        try:
            short_val = 0.0 if short_units is None else float(short_units)
        except (TypeError, ValueError):
            short_val = 0.0
        return self.close_position_side(instrument, long_val, short_val)

    def account_equity(self) -> Optional[float]:
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/summary")
                if resp.status_code == 200:
                    data = resp.json().get("account", {})
                    nav = data.get("NAV")
                    try:
                        equity = float(nav)
                        return equity if math.isfinite(equity) and equity > 0 else None
                    except (TypeError, ValueError):
                        return None
        except Exception as exc:
            print(f"[OANDA] Exception fetching equity: {exc}", flush=True)
        return None

    def current_spread(self, instrument: str) -> Optional[float]:
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/pricing",
                    params={"instruments": instrument},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json().get("prices", [])
                if not data:
                    return None
                price = data[0]
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    return None
                try:
                    bid = float(bids[0]["price"])
                    ask = float(asks[0]["price"])
                except (KeyError, TypeError, ValueError):
                    return None
                spread = ask - bid
                if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask < bid:
                    return None
                pip_size = self._pip_size(instrument)
                if pip_size <= 0:
                    return None
                return spread / pip_size
        except Exception as exc:
            print(f"[OANDA] Exception fetching spread for {instrument}: {exc}", flush=True)
            return None

    def mid_price(self, instrument: str) -> float | None:
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/pricing",
                    params={"instruments": instrument},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json().get("prices", [])
                if not data:
                    return None
                price = data[0]
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    return None
                bid = float(bids[0]["price"])
                ask = float(asks[0]["price"])
                return (bid + ask) / 2.0
        except Exception:
            return None

    def conversion_rate(self, from_ccy: str, to_ccy: str) -> float | None:
        source = (from_ccy or "").upper()
        target = (to_ccy or "").upper()
        if not source or not target:
            return None
        if source == target:
            return 1.0
        if not (self.key and self.account):
            return None

        cache_key = (source, target)
        cached = self._loss_conversion_cache.get(cache_key)
        now_monotonic = time.monotonic()
        if cached is not None and now_monotonic - cached[0] <= 15.0:
            return cached[1]

        # OANDA's accountLoss home-conversion factor is deliberately used
        # instead of a mid price: loss-side conversion is the conservative
        # value for sizing a protective stop in the account currency.
        candidates = (f"{source}_{target}", f"{target}_{source}")
        try:
            with self._client() as client:
                if self._account_currency is None:
                    summary = client.get(f"/v3/accounts/{self.account}/summary")
                    if summary.status_code != 200:
                        return None
                    account = summary.json().get("account", {})
                    self._account_currency = str(account.get("currency") or "").upper() or None
                if self._account_currency != target:
                    return None

                for instrument in candidates:
                    response = client.get(
                        f"/v3/accounts/{self.account}/pricing",
                        params={
                            "instruments": instrument,
                            "includeHomeConversions": "true",
                        },
                    )
                    if response.status_code != 200:
                        continue
                    payload = response.json()
                    conversions = payload.get("homeConversions")
                    if not isinstance(conversions, list):
                        continue
                    for conversion in conversions:
                        if not isinstance(conversion, dict):
                            continue
                        if str(conversion.get("currency") or "").upper() != source:
                            continue
                        try:
                            factor = float(conversion["accountLoss"])
                        except (KeyError, TypeError, ValueError, OverflowError):
                            return None
                        if math.isfinite(factor) and factor > 0:
                            self._loss_conversion_cache[cache_key] = (
                                now_monotonic,
                                factor,
                            )
                            return factor
                        return None
        except Exception as exc:
            print(f"[OANDA] Loss-side conversion unavailable: {exc}", flush=True)
        return None

    def close_all_positions(self) -> None:
        if not (self.key and self.account):
            return
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/openPositions")
                if resp.status_code != 200:
                    return
                for position in resp.json().get("positions", []):
                    instrument = position.get("instrument")
                    if not instrument:
                        continue
                    payload: Dict[str, str] = {}
                    long_units = position.get("long", {}).get("units")
                    short_units = position.get("short", {}).get("units")
                    if long_units and float(long_units) != 0:
                        payload["longUnits"] = "ALL"
                    if short_units and float(short_units) != 0:
                        payload["shortUnits"] = "ALL"
                    if not payload:
                        continue
                    client.put(
                        f"/v3/accounts/{self.account}/positions/{instrument}/close",
                        json=payload,
                    )
        except Exception as exc:
            print(f"[OANDA] Exception closing positions: {exc}", flush=True)

    @staticmethod
    def _pip_size(instrument: str) -> float:
        if instrument.endswith("JPY"):
            return 0.01
        if instrument.startswith("XAU"):
            return 0.1
        if instrument.startswith("XAG"):
            return 0.01
        return 0.0001
