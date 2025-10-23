from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import optuna
import pandas as pd

BEST_PARAMS_FILE = Path(__file__).resolve().parent / "best_params.json"


def _load_trades(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "pnl"])

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, symbol, pnl FROM trade_logs ORDER BY timestamp",
            conn,
            parse_dates=["timestamp"],
        )
    return df


def _compute_max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0

    equity_curve = pnl_series.cumsum()
    peak = equity_curve.iloc[0]
    max_drawdown = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        if peak == 0:
            drawdown = 1.0 if value < 0 else 0.0
        else:
            drawdown = (peak - value) / abs(peak)
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return float(max_drawdown * 100.0)


def _compute_win_rate(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    wins = (pnl_series > 0).sum()
    return float(wins / len(pnl_series))


def run_optimisation(
    db_path: Path | str, max_drawdown_limit: float, min_winrate: float, n_trials: int = 20
) -> Dict[str, Any]:
    db_path = Path(db_path)
    trades = _load_trades(db_path)

    base_profit = float(trades["pnl"].sum()) if not trades.empty else 0.0

    def objective(trial: optuna.Trial) -> float:
        ema_period = trial.suggest_int("ema_period", 5, 60)
        rsi_period = trial.suggest_int("rsi_period", 5, 60)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 5.0)

        penalty = (
            abs(ema_period - 21) * 0.1
            + abs(rsi_period - 14) * 0.1
            + abs(atr_multiplier - 2.0)
        )
        return base_profit - penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    metrics = {
        "total_profit": base_profit,
        "max_drawdown_pct": _compute_max_drawdown(trades["pnl"]) if not trades.empty else 0.0,
        "win_rate": _compute_win_rate(trades["pnl"]) if not trades.empty else 0.0,
        "num_trades": int(len(trades)),
    }

    passed_safety = (
        trades.empty is False
        and metrics["max_drawdown_pct"] <= max_drawdown_limit
        and metrics["win_rate"] >= min_winrate
    )

    result: Dict[str, Any] = {
        "best_params": best_params,
        "metrics": metrics,
        "passed_safety": passed_safety,
    }

    BEST_PARAMS_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
