from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:  # pragma: no cover - optional dependency
    load_dotenv()

from mcp import optimiser, patcher  # noqa: E402  # isort: skip
from mcp.patcher import PatchError  # noqa: E402  # isort: skip

DATABASE_PATH = Path(__file__).resolve().parent / "logs.db"
API_KEY_ENV = "MCP_API_KEY"

app = FastAPI(title="MCP Control Panel")


def init_db() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                pnl REAL NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(DATABASE_PATH))


init_db()


class TradeEntry(BaseModel):
    timestamp: datetime = Field(..., description="Time of the trade event")
    symbol: str = Field(..., description="Symbol traded")
    pnl: float = Field(..., description="Profit or loss for the trade")
    metadata: dict | None = Field(
        default=None, description="Additional metadata for the trade"
    )


class TradeBatch(BaseModel):
    trades: List[TradeEntry]


def require_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> None:
    expected_key = os.getenv(API_KEY_ENV)
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server missing MCP_API_KEY configuration",
        )
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


@app.post("/ingest-logs")
def ingest_logs(batch: TradeBatch, _: None = Depends(require_api_key)) -> dict:
    rows = []
    for trade in batch.trades:
        payload = trade.dict()
        payload["timestamp"] = trade.timestamp.isoformat()
        rows.append(
            (
                trade.timestamp.isoformat(),
                trade.symbol,
                float(trade.pnl),
                json.dumps(payload),
            )
        )

    with get_db_connection() as conn:
        conn.executemany(
            "INSERT INTO trade_logs (timestamp, symbol, pnl, raw_json) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    return {"inserted": len(rows)}


@app.get("/health")
def health() -> dict:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT COALESCE(SUM(pnl), 0.0) FROM trade_logs")
        total_pl = cursor.fetchone()[0]
    return {"status": "ok", "total_pl": float(total_pl)}


@app.post("/optimise")
def run_optimisation(_: None = Depends(require_api_key)) -> dict:
    max_drawdown_limit = float(os.getenv("MAX_DRAWDOWN_PCT", "20"))
    min_winrate = float(os.getenv("MIN_WINRATE", "0.5"))

    try:
        result = optimiser.run_optimisation(
            DATABASE_PATH, max_drawdown_limit, min_winrate
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimisation failed: {exc}",
        ) from exc

    result["patch_applied"] = False

    if result.get("passed_safety"):
        try:
            patcher.push_patch(json.dumps(result["best_params"]))
        except PatchError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to push patch: {exc}",
            ) from exc
        result["patch_applied"] = True

    return result
