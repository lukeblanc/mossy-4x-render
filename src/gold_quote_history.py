"""Record practice gold quotes and measure the first profitable delayed entry."""

import argparse
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def utc(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError("Expected a timestamp with a UTC offset")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="microseconds")


def database(path):
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS gold_quotes (
            broker_at TEXT PRIMARY KEY,
            received_at TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL
        )
    """)
    return connection


def record(connection, payload, received_at):
    for price in payload["prices"]:
        if price["instrument"] != "XAU_USD" or not price.get("tradeable"):
            continue
        broker_at = utc(price["time"])
        received_at = utc(received_at)
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            raise ValueError("Invalid gold bid/ask")
        if broker_at > received_at:
            raise ValueError("Gold quote timestamp is in the future")
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO gold_quotes VALUES (?, ?, ?, ?)",
                (broker_at, received_at, bid, ask),
            )


def delayed_entry(connection, *, side, fill, opened, closed, costs, target):
    if side not in ("BUY", "SELL"):
        raise ValueError("Expected BUY or SELL")
    if not all(math.isfinite(x) for x in (fill, costs, target)) or fill <= 0 or costs < 0:
        raise ValueError("Invalid fill, target or round-trip cost per unit")
    opened, closed = utc(opened), utc(closed)
    if closed <= opened:
        raise ValueError("Close must follow entry")
    rows = connection.execute(
        """SELECT broker_at, received_at, bid, ask FROM gold_quotes
           WHERE broker_at >= ? AND received_at < ?
           ORDER BY received_at, broker_at""",
        (opened, closed),
    )
    observed = 0
    for broker_at, received_at, bid, ask in rows:
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


def collect(connection, seconds):
    account = quote(os.environ["OANDA_ACCOUNT_ID"], safe="")
    key = os.environ["OANDA_API_KEY"]
    request = Request(
        f"https://api-fxpractice.oanda.com/v3/accounts/{account}/pricing?instruments=XAU_USD",
        headers={"Authorization": f"Bearer {key}"},
    )
    while True:
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.load(response)
            record(connection, payload, datetime.now(timezone.utc).isoformat())
        except (URLError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f"Gold quote unavailable ({type(exc).__name__})", flush=True)
        time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--seconds", type=int, default=5)
    replay = commands.add_parser("replay")
    replay.add_argument("--side", choices=["BUY", "SELL"], required=True)
    replay.add_argument("--fill", type=float, required=True)
    replay.add_argument("--opened", required=True)
    replay.add_argument("--closed", required=True)
    replay.add_argument("--costs", type=float, required=True, help="Round-trip costs in USD per unit, excluding spread")
    replay.add_argument("--target", type=float, required=True)
    args = parser.parse_args()
    if args.command == "collect" and args.seconds < 1:
        parser.error("--seconds must be positive")
    with database(args.database) as connection:
        if args.command == "collect":
            collect(connection, args.seconds)
        else:
            print(json.dumps(delayed_entry(
                connection, side=args.side, fill=args.fill, opened=args.opened,
                closed=args.closed, costs=args.costs, target=args.target,
            ), indent=2))


if __name__ == "__main__":
    main()
