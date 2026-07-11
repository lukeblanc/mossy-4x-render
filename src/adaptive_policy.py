from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src import session_filter


@dataclass(frozen=True)
class MarketContext:
    instrument: str
    side: str
    session: str
    rsi_bucket: str
    trend_bucket: str
    timestamp_utc: str

    @property
    def setup_key(self) -> str:
        return "|".join(
            (
                self.instrument,
                self.side,
                self.session,
                self.rsi_bucket,
                self.trend_bucket,
            )
        )


@dataclass(frozen=True)
class PolicyDecision:
    instrument: str
    setup_key: str
    risk_scale: float
    blocked: bool
    reason: str
    exact_samples: int = 0
    pair_side_samples: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    loss_streak: int = 0


_CONTEXTS: dict[str, MarketContext] = {}
_CONTEXT_LOCK = threading.RLock()
_DECISION_CACHE: dict[str, tuple[float, PolicyDecision]] = {}
_LAST_LOGGED: dict[str, str] = {}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _journal_path() -> Path:
    configured = os.getenv("MOSSY_STATE_PATH")
    if configured:
        root = Path(configured)
    elif Path("/var/data").exists():
        root = Path("/var/data")
    else:
        root = Path("data")
    return root / "trade_journal.db"


def _normalize_side(value: object, indicators: Mapping[str, Any] | None = None) -> str:
    label = str(value or "").strip().upper()
    if label in {"BUY", "LONG"}:
        return "BUY"
    if label in {"SELL", "SHORT"}:
        return "SELL"

    indicators = indicators or {}
    fast = _safe_float(indicators.get("ema_fast"))
    slow = _safe_float(indicators.get("ema_slow"))
    if math.isfinite(fast) and math.isfinite(slow):
        if fast > slow:
            return "BUY"
        if fast < slow:
            return "SELL"
    return "UNKNOWN"


def _normalize_session(value: object, timestamp_utc: datetime | None = None) -> str:
    label = str(value or "").strip().upper()
    if label:
        if label.startswith("LONDON"):
            return "LONDON"
        if label.startswith("NEWYORK") or label.startswith("NEW_YORK"):
            return "NEWYORK"
        if label in {"OFF_SESSION", "ADHOC", "SESSION"}:
            return label

    if timestamp_utc is not None:
        try:
            snapshot = session_filter.current_session(timestamp_utc, mode=os.getenv("SESSION_MODE", "SOFT"))
        except Exception:
            snapshot = None
        if snapshot is not None:
            return str(snapshot.name or "SESSION").upper()
    return "OFF_SESSION"


def _rsi_bucket(value: object) -> str:
    rsi = _safe_float(value)
    if not math.isfinite(rsi):
        return "RSI_UNKNOWN"
    if rsi < 45.0:
        return "RSI_LT45"
    if rsi < 50.0:
        return "RSI_45_50"
    if rsi < 55.0:
        return "RSI_50_55"
    return "RSI_GE55"


def _trend_bucket(side: str, indicators: Mapping[str, Any]) -> str:
    fast = _safe_float(indicators.get("ema_fast"))
    slow = _safe_float(indicators.get("ema_slow"))
    trend_fast = _safe_float(
        indicators.get("ema_trend_fast", indicators.get("ema50"))
    )
    trend_slow = _safe_float(
        indicators.get("ema_trend_slow", indicators.get("ema200"))
    )

    short_aligned = False
    long_aligned = False
    if math.isfinite(fast) and math.isfinite(slow):
        short_aligned = fast > slow if side == "BUY" else fast < slow
    if math.isfinite(trend_fast) and math.isfinite(trend_slow):
        long_aligned = trend_fast > trend_slow if side == "BUY" else trend_fast < trend_slow

    if short_aligned and long_aligned:
        return "TREND_FULL"
    if short_aligned or long_aligned:
        return "TREND_PARTIAL"
    return "TREND_COUNTER"


def _build_context(
    instrument: str,
    indicators: Mapping[str, Any],
    now_utc: datetime,
    *,
    side: object = None,
    session: object = None,
) -> MarketContext:
    normalized_side = _normalize_side(side, indicators)
    return MarketContext(
        instrument=str(instrument or "").strip().upper(),
        side=normalized_side,
        session=_normalize_session(session, now_utc),
        rsi_bucket=_rsi_bucket(indicators.get("rsi")),
        trend_bucket=_trend_bucket(normalized_side, indicators),
        timestamp_utc=now_utc.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
    )


def publish_market_context(
    instrument: str,
    indicators: Mapping[str, Any],
    now_utc: datetime,
    *,
    side: object = None,
    session: object = None,
) -> MarketContext:
    """Publish the most recent evaluated setup for later risk sizing.

    Context is kept in memory only. The permanent source of truth remains the
    SQLite trade journal, so a restart cannot invent or erase trade outcomes.
    """

    context = _build_context(
        instrument,
        indicators,
        now_utc,
        side=side,
        session=session,
    )
    if context.instrument:
        with _CONTEXT_LOCK:
            _CONTEXTS[context.instrument] = context
            _DECISION_CACHE.pop(context.instrument, None)
    return context


def _context_from_row(row: sqlite3.Row) -> Optional[MarketContext]:
    try:
        indicators_raw = row["indicators_snapshot"] or "{}"
        indicators = json.loads(indicators_raw) if isinstance(indicators_raw, str) else dict(indicators_raw)
    except Exception:
        indicators = {}

    ts_value = row["timestamp_utc"] or row["exit_timestamp_utc"]
    try:
        ts = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        ts = datetime.now(timezone.utc)

    instrument = str(row["instrument"] or "").strip().upper()
    if not instrument:
        return None
    return _build_context(
        instrument,
        indicators,
        ts,
        side=row["side"],
        session=row["session_id"],
    )


def _metrics(rows: list[sqlite3.Row]) -> dict[str, float | int | Optional[datetime]]:
    pnl_values = [float(row["realized_pnl_ccy"] or 0.0) for row in rows]
    wins = [pnl for pnl in pnl_values if pnl > 0]
    losses = [pnl for pnl in pnl_values if pnl < 0]
    sample = len(pnl_values)
    win_rate = len(wins) / sample if sample else 0.0
    expectancy = sum(pnl_values) / sample if sample else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    loss_streak = 0
    for pnl in pnl_values:
        if pnl < 0:
            loss_streak += 1
        else:
            break

    last_ts: Optional[datetime] = None
    if rows:
        raw_ts = rows[0]["exit_timestamp_utc"]
        try:
            last_ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
        except Exception:
            last_ts = None

    return {
        "sample": sample,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "loss_streak": loss_streak,
        "last_ts": last_ts,
    }


def _read_history(db_path: Path, instrument: str, lookback: int) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(trades)").fetchall()
                if len(row) > 1
            }
            required = {
                "instrument",
                "side",
                "session_id",
                "indicators_snapshot",
                "realized_pnl_ccy",
                "timestamp_utc",
                "exit_timestamp_utc",
            }
            if not required.issubset(columns):
                return []
            return conn.execute(
                """
                SELECT instrument, side, session_id, indicators_snapshot,
                       realized_pnl_ccy, timestamp_utc, exit_timestamp_utc
                FROM trades
                WHERE instrument = ?
                  AND exit_timestamp_utc IS NOT NULL
                  AND realized_pnl_ccy IS NOT NULL
                ORDER BY exit_timestamp_utc DESC
                LIMIT ?
                """,
                (instrument, max(10, int(lookback))),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return []


def _decision_for_context(
    context: MarketContext,
    rows: list[sqlite3.Row],
    *,
    now_utc: datetime,
) -> PolicyDecision:
    exact_rows: list[sqlite3.Row] = []
    pair_side_rows: list[sqlite3.Row] = []
    for row in rows:
        historical = _context_from_row(row)
        if historical is None:
            continue
        if historical.side == context.side:
            pair_side_rows.append(row)
        if historical.setup_key == context.setup_key:
            exact_rows.append(row)

    exact = _metrics(exact_rows)
    pair_side = _metrics(pair_side_rows)
    exact_n = int(exact["sample"])
    pair_n = int(pair_side["sample"])
    min_exact = max(3, int(os.getenv("ADAPTIVE_POLICY_MIN_EXACT", "6")))
    min_pair_side = max(min_exact, int(os.getenv("ADAPTIVE_POLICY_MIN_PAIR_SIDE", "12")))
    block_minutes = max(30.0, float(os.getenv("ADAPTIVE_POLICY_BLOCK_MINUTES", "240")))
    floor_scale = max(0.1, min(1.0, float(os.getenv("ADAPTIVE_POLICY_FLOOR_SCALE", "0.25"))))

    scale = 1.0
    reason = "insufficient-setup-history"
    severe = False

    if exact_n >= min_exact:
        reason = "setup-performing-normally"
        if int(exact["loss_streak"]) >= 4:
            severe = True
            reason = "setup-four-loss-streak"
        elif exact_n >= 10 and float(exact["win_rate"]) < 0.25 and float(exact["expectancy"]) < 0:
            severe = True
            reason = "setup-persistently-negative"
        elif int(exact["loss_streak"]) >= 3:
            scale = min(scale, 0.5)
            reason = "setup-three-loss-streak"
        elif float(exact["win_rate"]) < 0.35 and float(exact["expectancy"]) < 0:
            scale = min(scale, 0.5)
            reason = "setup-low-winrate-negative-expectancy"
        elif float(exact["profit_factor"]) < 0.9:
            scale = min(scale, 0.65)
            reason = "setup-profit-factor-below-one"
        elif float(exact["expectancy"]) < 0:
            scale = min(scale, 0.75)
            reason = "setup-negative-expectancy"

    if pair_n >= min_pair_side:
        if int(pair_side["loss_streak"]) >= 3:
            scale = min(scale, 0.65)
            if reason in {"insufficient-setup-history", "setup-performing-normally"}:
                reason = "pair-side-loss-streak"
        elif float(pair_side["profit_factor"]) < 0.8 and float(pair_side["expectancy"]) < 0:
            scale = min(scale, 0.6)
            if reason in {"insufficient-setup-history", "setup-performing-normally"}:
                reason = "pair-side-negative"

    blocked = False
    if severe:
        last_ts = exact.get("last_ts")
        age_minutes = None
        if isinstance(last_ts, datetime):
            age_minutes = max(0.0, (now_utc - last_ts.astimezone(timezone.utc)).total_seconds() / 60.0)
        if age_minutes is None or age_minutes < block_minutes:
            scale = 0.0
            blocked = True
        else:
            scale = min(scale, 0.5)
            reason = f"{reason}-cooldown-expired-probe"

    if not blocked:
        scale = max(floor_scale, min(1.0, scale))

    return PolicyDecision(
        instrument=context.instrument,
        setup_key=context.setup_key,
        risk_scale=scale,
        blocked=blocked,
        reason=reason,
        exact_samples=exact_n,
        pair_side_samples=pair_n,
        win_rate=float(exact["win_rate"]),
        expectancy=float(exact["expectancy"]),
        profit_factor=float(exact["profit_factor"]),
        loss_streak=int(exact["loss_streak"]),
    )


def _write_audit(decision: PolicyDecision, context: MarketContext) -> None:
    try:
        path = _journal_path().parent / "learning_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "context": asdict(context),
            "decision": asdict(decision),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def evaluate_instrument_policy(
    instrument: str,
    *,
    db_path: Path | str | None = None,
    now_utc: datetime | None = None,
) -> PolicyDecision:
    normalized = str(instrument or "").strip().upper()
    if not _as_bool(os.getenv("ADAPTIVE_POLICY_ENABLED", "true")):
        return PolicyDecision(normalized, "disabled", 1.0, False, "policy-disabled")

    with _CONTEXT_LOCK:
        context = _CONTEXTS.get(normalized)
    if context is None:
        return PolicyDecision(normalized, "missing", 1.0, False, "no-market-context")

    ttl = max(1.0, float(os.getenv("ADAPTIVE_POLICY_CACHE_SECONDS", "30")))
    cache_key = f"{normalized}:{context.setup_key}"
    now_monotonic = time.monotonic()
    with _CONTEXT_LOCK:
        cached = _DECISION_CACHE.get(cache_key)
        if cached and now_monotonic - cached[0] <= ttl:
            return cached[1]

    history = _read_history(
        Path(db_path) if db_path is not None else _journal_path(),
        normalized,
        int(os.getenv("ADAPTIVE_POLICY_LOOKBACK", "200")),
    )
    decision = _decision_for_context(
        context,
        history,
        now_utc=(now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc),
    )

    with _CONTEXT_LOCK:
        _DECISION_CACHE[cache_key] = (now_monotonic, decision)
        signature = f"{decision.setup_key}:{decision.risk_scale:.3f}:{decision.reason}:{decision.exact_samples}"
        if _LAST_LOGGED.get(normalized) != signature:
            print(
                f"[LEARNING] instrument={normalized} setup={decision.setup_key} "
                f"scale={decision.risk_scale:.2f} blocked={str(decision.blocked).lower()} "
                f"exact_n={decision.exact_samples} pair_side_n={decision.pair_side_samples} "
                f"win_rate={decision.win_rate:.3f} expectancy={decision.expectancy:.3f} "
                f"pf={decision.profit_factor:.3f} loss_streak={decision.loss_streak} "
                f"reason={decision.reason}",
                flush=True,
            )
            _LAST_LOGGED[normalized] = signature

    _write_audit(decision, context)
    return decision


def clear_policy_caches() -> None:
    with _CONTEXT_LOCK:
        _CONTEXTS.clear()
        _DECISION_CACHE.clear()
        _LAST_LOGGED.clear()


__all__ = [
    "MarketContext",
    "PolicyDecision",
    "publish_market_context",
    "evaluate_instrument_policy",
    "clear_policy_caches",
]
