"""Record practice forex and gold quotes and measure the first profitable delayed entry."""

import argparse
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx


DEFAULT_INSTRUMENTS = ("AUD_USD", "GBP_USD", "EUR_USD", "XAU_USD")
MAX_QUOTE_AGE_SECONDS = 15
STATUS_INTERVAL_SECONDS = 60
PRACTICE_HOST = "https://api-fxpractice.oanda.com"


def utc(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError("Expected a timestamp with a UTC offset")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="microseconds")


def database(path):
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS practice_quotes (
            instrument TEXT NOT NULL,
            broker_at TEXT NOT NULL,
            received_at TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL,
            PRIMARY KEY (instrument, broker_at)
        )
    """)
    legacy = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gold_quotes'"
    ).fetchone()
    if legacy:
        with connection:
            connection.execute("""
                INSERT OR IGNORE INTO practice_quotes
                SELECT 'XAU_USD', broker_at, received_at, bid, ask FROM gold_quotes
            """)
    return connection


def record(connection, payload, received_at, instruments=DEFAULT_INSTRUMENTS):
    received_at = utc(received_at)
    seen, states, pending = set(), {}, []
    for price in payload["prices"]:
        instrument = price["instrument"]
        if instrument not in instruments:
            continue
        if instrument in seen:
            raise ValueError("Duplicate instrument in response")
        seen.add(instrument)
        broker_at = utc(price["time"])
        age = (datetime.fromisoformat(received_at) - datetime.fromisoformat(broker_at)).total_seconds()
        if age < 0:
            raise ValueError("Quote timestamp is in the future")
        tradeable = price.get("tradeable")
        if tradeable is None:
            if price.get("status") not in ("tradeable", "non-tradeable", "invalid"):
                raise ValueError("Missing tradability")
            tradeable = price.get("status") == "tradeable"
        if tradeable is not True:
            # Closed weekend quotes are not an HTTP/recorder failure or tradable evidence.
            if tradeable is False and price.get("status") in (None, "non-tradeable", "invalid"):
                states[instrument] = "market_closed" if price.get("status") != "invalid" else "invalid"
                continue
            raise ValueError("Invalid tradability value")
        if age > MAX_QUOTE_AGE_SECONDS:
            states[instrument] = "stale"
            continue
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            raise ValueError("Invalid bid/ask")
        pending.append((instrument, broker_at, received_at, bid, ask))
        states[instrument] = "fresh"
    before = connection.total_changes
    with connection:
        connection.executemany("INSERT OR IGNORE INTO practice_quotes VALUES (?, ?, ?, ?, ?)", pending)
    inserted = connection.total_changes - before
    for instrument in instruments:
        states.setdefault(instrument, "missing")
    values = set(states.values())
    state = ("fresh" if values == {"fresh"} else "market_closed" if values == {"market_closed"}
             else "incomplete" if values & {"missing", "invalid"} else "stale" if "stale" in values else "partial")
    return {"state": state, "instruments": states, "new_rows": inserted}


def delayed_entry(connection, *, side, fill, opened, closed, costs, target, instrument="XAU_USD"):
    if side not in ("BUY", "SELL"):
        raise ValueError("Expected BUY or SELL")
    if not all(math.isfinite(x) for x in (fill, costs, target)) or fill <= 0 or costs < 0 or target <= 0:
        raise ValueError("Invalid fill, target or round-trip cost per unit")
    opened, closed = utc(opened), utc(closed)
    if closed <= opened:
        raise ValueError("Close must follow entry")
    rows = connection.execute(
        """SELECT broker_at, received_at, bid, ask FROM practice_quotes
           WHERE instrument = ? AND broker_at >= ? AND received_at < ?
           ORDER BY received_at, broker_at""",
        (instrument, opened, closed),
    )
    observed = 0
    for broker_at, received_at, bid, ask in rows:
        age = (datetime.fromisoformat(received_at) - datetime.fromisoformat(broker_at)).total_seconds()
        if not 0 <= age <= MAX_QUOTE_AGE_SECONDS:
            continue
        observed += 1
        if side == "BUY":
            if bid - fill <= costs:
                continue
            entry, room, penalty = ask, target - ask, ask - fill
        else:
            if fill - ask <= costs:
                continue
            entry, room, penalty = bid, bid - target, fill - bid
        return {
            "status": "observed" if room > 0 else "target_already_reached",
            "broker_quote_at": broker_at,
            "available_at": received_at,
            "simulated_entry": entry,
            "entry_price_penalty": penalty,
            "distance_to_original_target": room,
            "observations_checked": observed,
        }
    return {"status": "no_positive_observation" if observed else "no_quotes",
            "observations_checked": observed}


def coverage(connection, instruments=DEFAULT_INSTRUMENTS):
    rows = connection.execute("SELECT instrument, COUNT(*), MIN(broker_at), MAX(broker_at) "
                              "FROM practice_quotes GROUP BY instrument").fetchall()
    found = {i: {"rows": n, "first_quote_utc": first, "last_quote_utc": last}
             for i, n, first, last in rows}
    return {i: found.get(i, {"rows": 0, "first_quote_utc": None, "last_quote_utc": None})
            for i in instruments}


def save_status(path, status):
    if path is None:
        return
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".quotes-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(status, output, sort_keys=True, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(connection, seconds, instruments, *, account, key, client=None, stop_event=None, status_path=None):
    if not math.isfinite(seconds) or seconds < 1:
        raise ValueError("Polling interval must be positive")
    stop_event = stop_event or threading.Event()
    if client is None:
        with httpx.Client(timeout=10, follow_redirects=False) as owned:
            return collect(connection, seconds, instruments, account=account, key=key,
                           client=owned, stop_event=stop_event, status_path=status_path)
    url = f"{PRACTICE_HOST}/v3/accounts/{quote(account, safe='')}/pricing"
    failures, last_log, last_state, last_success = 0, None, None, None
    while not stop_event.is_set():
        observed = datetime.now(timezone.utc).isoformat()
        status = {"schema": "mossy.practice-quotes.v1", "observed_at_utc": observed,
                  "source": "oanda-practice", "poll_interval_seconds": seconds,
                  "max_quote_age_seconds": MAX_QUOTE_AGE_SECONDS}
        try:
            response = client.get(url, params={"instruments": ",".join(instruments)},
                                  headers={"Authorization": f"Bearer {key}"}, follow_redirects=False)
            response.raise_for_status()
            observed = datetime.now(timezone.utc).isoformat()
            status["observed_at_utc"] = observed
            status.update(record(connection, response.json(), observed, instruments))
            failures, delay = 0, seconds
            last_success = observed
        except (httpx.HTTPError, OSError, ValueError, KeyError, IndexError, TypeError, sqlite3.Error) as exc:
            failures += 1
            delay = min(60, seconds * 2 ** min(failures, 4))
            status.update(state="error", error_type=type(exc).__name__)
            if isinstance(exc, httpx.HTTPStatusError):
                status["http_status"] = exc.response.status_code
                if exc.response.status_code in (401, 403):
                    delay = 300
                elif exc.response.status_code == 429:
                    try:
                        delay = min(300, max(delay, int(exc.response.headers.get("retry-after", "0"))))
                    except ValueError:
                        pass
            # Never log exception text, response bodies, credentials or account URLs.
        status.update(last_successful_response_utc=last_success, consecutive_failures=failures,
                      next_poll_seconds=delay)
        stamp = time.monotonic()
        signature = (status["state"], status.get("error_type"), status.get("http_status"))
        if last_log is None or stamp - last_log >= STATUS_INTERVAL_SECONDS or signature != last_state:
            try:
                status["coverage"] = coverage(connection, instruments)
                save_status(status_path, status)
            except (OSError, sqlite3.Error) as exc:
                status.update(state="storage_error", error_type=type(exc).__name__)
            print("[PRACTICE_QUOTES] " + json.dumps(status, sort_keys=True), flush=True)
            last_log, last_state = stamp, signature
        stop_event.wait(delay)


def collect_for_worker(path, account, key):
    """Keep network and SQLite work off the trading event loop."""
    if os.getenv("MODE", "").lower() != "demo" or os.getenv("OANDA_ENV", "").lower() != "practice":
        raise ValueError("practice-demo-mode-required")
    path = Path(path)
    while True:
        connection = None
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            os.chmod(path, 0o600)
            connection = database(path)
            print("[PRACTICE_QUOTES] collector started", flush=True)
            collect(connection, 5, DEFAULT_INSTRUMENTS, account=account, key=key,
                    status_path=path.with_suffix(".status.json"))
            return
        except (OSError, sqlite3.Error) as exc:
            print(f"[PRACTICE_QUOTES] storage unavailable ({type(exc).__name__}); retry_seconds=60", flush=True)
        finally:
            if connection is not None:
                connection.close()
        threading.Event().wait(60)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--seconds", type=int, default=5)
    capture.add_argument("--instruments", nargs="+", choices=DEFAULT_INSTRUMENTS, default=DEFAULT_INSTRUMENTS)
    replay = commands.add_parser("replay")
    replay.add_argument("--instrument", choices=DEFAULT_INSTRUMENTS, default="XAU_USD")
    replay.add_argument("--side", choices=["BUY", "SELL"], required=True)
    replay.add_argument("--fill", type=float, required=True)
    replay.add_argument("--opened", required=True)
    replay.add_argument("--closed", required=True)
    replay.add_argument("--costs", type=float, required=True, help="Round-trip costs in USD per unit, excluding spread")
    replay.add_argument("--target", type=float, required=True)
    args = parser.parse_args()
    if args.command == "collect" and args.seconds < 1:
        parser.error("--seconds must be positive")
    if args.command == "collect" and (os.getenv("MODE") != "demo" or os.getenv("OANDA_ENV") != "practice"):
        parser.error("Quote collection requires explicit demo/practice mode")
    with database(args.database) as connection:
        if args.command == "collect":
            collect(connection, args.seconds, args.instruments,
                    account=os.environ["OANDA_ACCOUNT_ID"], key=os.environ["OANDA_API_KEY"])
        else:
            print(json.dumps(delayed_entry(
                connection, side=args.side, fill=args.fill, opened=args.opened,
                closed=args.closed, costs=args.costs, target=args.target, instrument=args.instrument,
            ), indent=2))


if __name__ == "__main__":
    main()
