from __future__ import annotations

import base64
import json
import math
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PERTH = ZoneInfo("Australia/Perth")
DEFAULT_REPOSITORY = "lukeblanc/mossy-4x-render"
DEFAULT_BRANCH = "main"
_MONITOR_STARTED = False
_MONITOR_LOCK = threading.Lock()


@dataclass(frozen=True)
class SegmentMetrics:
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    expectancy: float
    profit_factor: float


@dataclass(frozen=True)
class WeeklyOpsReport:
    generated_utc: str
    week_start_utc: str
    week_end_utc: str
    status: str
    readiness_score: int
    journal_path: str
    journal_exists: bool
    journal_last_modified_utc: str | None
    open_trades: int
    total: SegmentMetrics
    max_drawdown: float
    longest_losing_streak: int
    best_instrument: str | None
    worst_instrument: str | None
    by_instrument: dict[str, SegmentMetrics]
    by_direction: dict[str, SegmentMetrics]
    by_session: dict[str, SegmentMetrics]
    by_exit_reason: dict[str, SegmentMetrics]
    alerts: tuple[str, ...]


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _journal_path() -> Path:
    configured = os.getenv("MOSSY_STATE_PATH")
    if configured:
        return Path(configured) / "trade_journal.db"
    if Path("/var/data").exists():
        return Path("/var/data/trade_journal.db")
    return Path("data/trade_journal.db")


def _report_dir() -> Path:
    configured = os.getenv("ALGO_REPORT_DIR")
    if configured:
        return Path(configured)
    journal = _journal_path()
    return journal.parent / "algo-reports"


def _week_bounds(now_utc: datetime | None = None) -> tuple[datetime, datetime]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    week_end = now
    week_start = week_end - timedelta(days=7)
    return week_start, week_end


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        if len(row) > 1
    }


def _load_closed_trades(db_path: Path, start_utc: datetime, end_utc: datetime) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path, timeout=3.0)
        conn.row_factory = sqlite3.Row
        try:
            columns = _table_columns(conn, "trades")
            required = {"exit_timestamp_utc", "realized_pnl_ccy"}
            if not required.issubset(columns):
                return []
            optional = {
                "instrument": "''",
                "side": "''",
                "session_id": "''",
                "exit_reason": "''",
                "duration_seconds": "NULL",
                "spread_at_entry": "NULL",
                "spread_at_exit": "NULL",
                "max_profit_ccy": "NULL",
                "broker_confirmed": "NULL",
            }
            select_parts = ["exit_timestamp_utc", "realized_pnl_ccy"]
            for name, fallback in optional.items():
                select_parts.append(name if name in columns else f"{fallback} AS {name}")
            rows = conn.execute(
                f"""
                SELECT {', '.join(select_parts)}
                FROM trades
                WHERE exit_timestamp_utc IS NOT NULL
                  AND realized_pnl_ccy IS NOT NULL
                  AND exit_timestamp_utc >= ?
                  AND exit_timestamp_utc <= ?
                ORDER BY exit_timestamp_utc ASC
                """,
                (
                    start_utc.replace(microsecond=0).isoformat(),
                    end_utc.replace(microsecond=0).isoformat(),
                ),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []

    trades: list[dict[str, Any]] = []
    for row in rows:
        pnl = _safe_float(row["realized_pnl_ccy"])
        timestamp = _parse_timestamp(row["exit_timestamp_utc"])
        if pnl is None or timestamp is None:
            continue
        trades.append(
            {
                "timestamp": timestamp,
                "pnl": pnl,
                "instrument": str(row["instrument"] or "UNKNOWN").upper(),
                "direction": str(row["side"] or "UNKNOWN").upper(),
                "session": str(row["session_id"] or "UNKNOWN").upper(),
                "exit_reason": str(row["exit_reason"] or "UNKNOWN").upper(),
                "duration_seconds": _safe_float(row["duration_seconds"]),
                "spread_at_entry": _safe_float(row["spread_at_entry"]),
                "spread_at_exit": _safe_float(row["spread_at_exit"]),
                "max_profit_ccy": _safe_float(row["max_profit_ccy"]),
                "broker_confirmed": row["broker_confirmed"],
            }
        )
    return trades


def _count_open_trades(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path, timeout=3.0) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE exit_timestamp_utc IS NULL"
            ).fetchone()
            return int(row[0] if row else 0)
    except sqlite3.Error:
        return 0


def _metrics(trades: list[dict[str, Any]]) -> SegmentMetrics:
    pnl = [float(trade["pnl"]) for trade in trades]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    total = len(pnl)
    return SegmentMetrics(
        trades=total,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / total if total else 0.0,
        net_pnl=sum(pnl),
        expectancy=sum(pnl) / total if total else 0.0,
        profit_factor=(
            gross_profit / gross_loss
            if gross_loss > 0
            else (99.0 if gross_profit > 0 else 0.0)
        ),
    )


def _segments(trades: list[dict[str, Any]], key: str) -> dict[str, SegmentMetrics]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        label = str(trade.get(key) or "UNKNOWN")
        grouped.setdefault(label, []).append(trade)
    return {label: _metrics(items) for label, items in sorted(grouped.items())}


def _drawdown_and_streak(trades: list[dict[str, Any]]) -> tuple[float, int]:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    current_streak = 0
    longest_streak = 0
    for trade in trades:
        pnl = float(trade["pnl"])
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        if pnl < 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0
    return max_drawdown, longest_streak


def _readiness(total: SegmentMetrics, alerts: list[str], journal_exists: bool) -> tuple[int, str]:
    if not journal_exists:
        return 20, "STAND DOWN"
    score = 80
    if total.trades == 0:
        score -= 15
    elif total.trades < 5:
        score -= 5
    if total.trades >= 5 and total.profit_factor < 0.8:
        score -= 20
    elif total.trades >= 5 and total.profit_factor < 1.0:
        score -= 10
    if total.trades >= 5 and total.expectancy < 0:
        score -= 10
    score -= min(20, len(alerts) * 5)
    score = max(0, min(100, score))
    status = "GO" if score >= 80 else "CAUTION" if score >= 55 else "STAND DOWN"
    return score, status


def build_weekly_report(
    db_path: Path | str | None = None,
    *,
    now_utc: datetime | None = None,
) -> WeeklyOpsReport:
    path = Path(db_path) if db_path is not None else _journal_path()
    start_utc, end_utc = _week_bounds(now_utc)
    trades = _load_closed_trades(path, start_utc, end_utc)
    total = _metrics(trades)
    max_drawdown, longest_streak = _drawdown_and_streak(trades)
    by_instrument = _segments(trades, "instrument")
    by_direction = _segments(trades, "direction")
    by_session = _segments(trades, "session")
    by_exit_reason = _segments(trades, "exit_reason")

    alerts: list[str] = []
    journal_exists = path.exists()
    if not journal_exists:
        alerts.append("trade journal missing")
    if journal_exists and total.trades == 0:
        alerts.append("no broker-confirmed closed trades in the last seven days")
    if total.trades >= 5 and total.profit_factor < 1.0:
        alerts.append("weekly profit factor below 1.0")
    if total.trades >= 5 and total.expectancy < 0:
        alerts.append("weekly expectancy is negative")
    if longest_streak >= 3:
        alerts.append(f"losing streak reached {longest_streak}")
    unconfirmed = sum(
        1
        for trade in trades
        if trade.get("broker_confirmed") not in (1, True, "1", "true", "True")
    )
    if unconfirmed:
        alerts.append(f"{unconfirmed} closed trade rows are not broker-confirmed")

    best = max(by_instrument, key=lambda key: by_instrument[key].net_pnl) if by_instrument else None
    worst = min(by_instrument, key=lambda key: by_instrument[key].net_pnl) if by_instrument else None
    readiness_score, status = _readiness(total, alerts, journal_exists)
    modified = None
    if journal_exists:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    return WeeklyOpsReport(
        generated_utc=(now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        week_start_utc=start_utc.replace(microsecond=0).isoformat(),
        week_end_utc=end_utc.replace(microsecond=0).isoformat(),
        status=status,
        readiness_score=readiness_score,
        journal_path=str(path),
        journal_exists=journal_exists,
        journal_last_modified_utc=modified,
        open_trades=_count_open_trades(path),
        total=total,
        max_drawdown=max_drawdown,
        longest_losing_streak=longest_streak,
        best_instrument=best,
        worst_instrument=worst,
        by_instrument=by_instrument,
        by_direction=by_direction,
        by_session=by_session,
        by_exit_reason=by_exit_reason,
        alerts=tuple(alerts),
    )


def _metric_line(label: str, value: SegmentMetrics) -> str:
    return (
        f"- **{label}:** trades={value.trades}, wins={value.wins}, losses={value.losses}, "
        f"win rate={value.win_rate:.1%}, net={value.net_pnl:.2f}, "
        f"expectancy={value.expectancy:.3f}, profit factor={value.profit_factor:.3f}"
    )


def render_markdown(report: WeeklyOpsReport) -> str:
    lines = [
        "# Mossy 4X Weekly ALGO Operations Report",
        "",
        f"**Generated UTC:** {report.generated_utc}",
        f"**Window:** {report.week_start_utc} to {report.week_end_utc}",
        f"**Trading status:** {report.status}",
        f"**Readiness score:** {report.readiness_score}/100",
        "",
        "## Verified performance",
        _metric_line("All closed trades", report.total),
        f"- **Maximum reconstructed drawdown:** {report.max_drawdown:.2f}",
        f"- **Longest losing streak:** {report.longest_losing_streak}",
        f"- **Open journal rows:** {report.open_trades}",
        f"- **Best instrument:** {report.best_instrument or 'not enough data'}",
        f"- **Worst instrument:** {report.worst_instrument or 'not enough data'}",
        "",
        "## By instrument",
    ]
    lines.extend(_metric_line(name, metrics) for name, metrics in report.by_instrument.items())
    if not report.by_instrument:
        lines.append("- No closed trades in this reporting window.")
    lines.extend(["", "## By direction"])
    lines.extend(_metric_line(name, metrics) for name, metrics in report.by_direction.items())
    lines.extend(["", "## By session"])
    lines.extend(_metric_line(name, metrics) for name, metrics in report.by_session.items())
    lines.extend(["", "## By exit reason"])
    lines.extend(_metric_line(name, metrics) for name, metrics in report.by_exit_reason.items())
    lines.extend(["", "## Alerts"])
    if report.alerts:
        lines.extend(f"- {alert}" for alert in report.alerts)
    else:
        lines.append("- No journal or performance alerts detected.")
    lines.extend(
        [
            "",
            "## System learning guardrails",
            "- Champion entry and execution rules were not changed by this report.",
            "- Shadow recommendations remain advisory and auto-apply remains disabled.",
            "- Performance numbers come only from completed rows in the persistent SQLite journal.",
            "",
            "## External intelligence placeholders",
            "- FX macro intelligence, next-week watch list and COT positioning require an external market-data/news step.",
            "- This operations report does not invent those values when no verified source is connected.",
            "",
        ]
    )
    return "\n".join(lines)


def save_report(report: WeeklyOpsReport, directory: Path | str | None = None) -> tuple[Path, Path, Path]:
    root = Path(directory) if directory is not None else _report_dir()
    root.mkdir(parents=True, exist_ok=True)
    generated = datetime.fromisoformat(report.generated_utc)
    week_key = generated.astimezone(PERTH).date().isoformat()
    markdown_path = root / f"{week_key}.md"
    json_path = root / f"{week_key}.json"
    latest_path = root / "LATEST_ALGO_REPORT.md"
    markdown = render_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(markdown, encoding="utf-8")
    return markdown_path, json_path, latest_path


def _github_request(method: str, url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mossy-4x-weekly-report",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _put_github_file(
    *,
    repository: str,
    branch: str,
    path: str,
    content: str,
    token: str,
    message: str,
) -> None:
    url = f"https://api.github.com/repos/{repository}/contents/{path}"
    sha = None
    try:
        existing = _github_request("GET", f"{url}?ref={branch}", token)
        sha = existing.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    _github_request("PUT", url, token, payload)


def publish_report_to_github(report: WeeklyOpsReport) -> bool:
    token = os.getenv("GITHUB_REPORT_TOKEN") or os.getenv("GH_PAT")
    if not token:
        print("[ALGO-REPORT] GitHub publish skipped: no GITHUB_REPORT_TOKEN/GH_PAT", flush=True)
        return False
    repository = os.getenv("GITHUB_REPORT_REPOSITORY", DEFAULT_REPOSITORY)
    branch = os.getenv("GITHUB_REPORT_BRANCH", DEFAULT_BRANCH)
    generated = datetime.fromisoformat(report.generated_utc).astimezone(PERTH)
    week_key = generated.date().isoformat()
    markdown = render_markdown(report)
    archive_path = f"reports/algo-weekly/{week_key}.md"
    latest_path = "reports/algo-weekly/LATEST_ALGO_REPORT.md"
    try:
        _put_github_file(
            repository=repository,
            branch=branch,
            path=archive_path,
            content=markdown,
            token=token,
            message=f"Add weekly ALGO operations report for {week_key}",
        )
        _put_github_file(
            repository=repository,
            branch=branch,
            path=latest_path,
            content=markdown,
            token=token,
            message=f"Update latest ALGO operations report for {week_key}",
        )
    except Exception as exc:
        print(f"[ALGO-REPORT][WARN] GitHub publish failed error={exc}", flush=True)
        return False
    print(f"[ALGO-REPORT] published week={week_key} repository={repository}", flush=True)
    return True


def generate_and_publish(now_utc: datetime | None = None) -> WeeklyOpsReport:
    report = build_weekly_report(now_utc=now_utc)
    markdown_path, json_path, _ = save_report(report)
    published = publish_report_to_github(report)
    print(
        f"[ALGO-REPORT] generated status={report.status} readiness={report.readiness_score} "
        f"trades={report.total.trades} net={report.total.net_pnl:.2f} "
        f"pf={report.total.profit_factor:.3f} markdown={markdown_path} json={json_path} "
        f"github_published={str(published).lower()}",
        flush=True,
    )
    return report


def _published_marker(root: Path) -> Path:
    return root / ".last_weekly_report"


def _due(now_utc: datetime, marker: Path) -> bool:
    local = now_utc.astimezone(PERTH)
    if local.weekday() != 6 or local.hour < 19:
        return False
    key = local.date().isoformat()
    try:
        return marker.read_text(encoding="utf-8").strip() != key
    except OSError:
        return True


def _monitor_loop() -> None:
    root = _report_dir()
    marker = _published_marker(root)
    interval = max(900, int(os.getenv("ALGO_REPORT_CHECK_SECONDS", "3600")))
    while True:
        now = datetime.now(timezone.utc)
        if _due(now, marker):
            try:
                generate_and_publish(now)
                root.mkdir(parents=True, exist_ok=True)
                marker.write_text(now.astimezone(PERTH).date().isoformat(), encoding="utf-8")
            except Exception as exc:
                print(f"[ALGO-REPORT][WARN] generation failed error={exc}", flush=True)
        time.sleep(interval)


def start_weekly_ops_monitor() -> bool:
    global _MONITOR_STARTED
    enabled = str(os.getenv("ALGO_WEEKLY_REPORT_ENABLED", "true")).strip().lower() in {
        "1", "true", "yes", "on", "y"
    }
    running_on_render = bool(
        os.getenv("RENDER_GIT_COMMIT")
        or os.getenv("RENDER_SERVICE_ID")
        or os.getenv("RENDER_INSTANCE_ID")
    )
    if not enabled or not running_on_render:
        return False
    with _MONITOR_LOCK:
        if _MONITOR_STARTED:
            return True
        thread = threading.Thread(target=_monitor_loop, name="mossy-weekly-ops", daemon=True)
        thread.start()
        _MONITOR_STARTED = True
    print("[ALGO-REPORT] weekly operations monitor started", flush=True)
    return True


__all__ = [
    "SegmentMetrics",
    "WeeklyOpsReport",
    "build_weekly_report",
    "generate_and_publish",
    "publish_report_to_github",
    "render_markdown",
    "save_report",
    "start_weekly_ops_monitor",
]
