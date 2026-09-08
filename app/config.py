"""Application configuration for the Render worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _apply_render_safe_demo_profile() -> None:
    """Force the approved demo and self-learning profile before imports.

    The learner may only reduce or temporarily block the already-capped risk.
    Shadow learning may recommend strategy filters, but it cannot apply them.
    """

    running_on_render = bool(
        os.getenv("RENDER_GIT_COMMIT")
        or os.getenv("RENDER_SERVICE_ID")
        or os.getenv("RENDER_INSTANCE_ID")
    )
    enabled_default = "true" if running_on_render else "false"
    if not _as_bool(os.getenv("MOSSY_SAFE_DEMO_PROFILE", enabled_default)):
        return

    safe_values = {
        "MODE": "demo",
        "OANDA_ENV": "practice",
        "BASE_URL": "https://api-fxpractice.oanda.com/v3",
        "SESSION_MODE": "SOFT",
        "AGGRESSIVE_MODE": "false",
        "AGGRESSIVE_TEST_MODE": "false",
        "AGGRESSIVE_TEST_RISK_PCT": "0.25",
        "INSTRUMENTS": "AUD_USD,GBP_USD",
        "MERGE_DEFAULT_INSTRUMENTS": "false",
        "ENABLE_RISK_CAP": "true",
        "MAX_RISK_PER_TRADE_CAP_PCT": "0.5",
        "MAX_RISK_PER_TRADE": "0.0025",
        "ALLOW_HIGH_RISK": "false",
        "DAILY_LOSS_CAP_PCT": "0.01",
        "WEEKLY_LOSS_CAP_PCT": "0.03",
        "MAX_DRAWDOWN_CAP_PCT": "0.05",
        "MAX_OPEN_TRADES": "2",
        "COOLDOWN_CANDLES": "9",
        "TP_ENABLED": "true",
        "ADAPTIVE_TUNING_ENABLED": "true",
        "ADAPTIVE_WINDOW_START_UTC": "2026-07-13T12:47:00+00:00",
        "ADAPTIVE_RUN_TAG": "MINI_RUN",
        "ADAPTIVE_LOOKBACK": "80",
        "ADAPTIVE_MIN_SAMPLE": "20",
        "ADAPTIVE_POLICY_ENABLED": "true",
        "ADAPTIVE_POLICY_LOOKBACK": "200",
        "ADAPTIVE_POLICY_MIN_EXACT": "6",
        "ADAPTIVE_POLICY_MIN_PAIR_SIDE": "12",
        "ADAPTIVE_POLICY_BLOCK_MINUTES": "240",
        "ADAPTIVE_POLICY_FLOOR_SCALE": "0.25",
        "ADAPTIVE_POLICY_CACHE_SECONDS": "30",
        "SHADOW_LEARNING_ENABLED": "true",
        "SHADOW_COHORT_START_UTC": "2026-07-13T12:47:00+00:00",
        "SHADOW_INTERVAL_SECONDS": "3600",
        "SHADOW_TRAIN_RATIO": "0.70",
        "SHADOW_MIN_TRAIN": "50",
        "SHADOW_MIN_VALIDATION": "30",
        "SHADOW_MIN_COVERAGE": "0.50",
        "ENABLE_PROJECTOR": "true",
        "VERBOSE_MARKET_LOGS": "false",
        "OPEN_TRADES_CACHE_TTL_SECONDS": "15",
    }
    # Preserve stricter operational limits regardless of Python import order.
    preserved = {key: os.environ[key] for key in (
        "ADAPTIVE_MIN_SAMPLE", "SHADOW_MIN_TRAIN", "SHADOW_MIN_VALIDATION",
        "SHADOW_MIN_COVERAGE", "DAILY_LOSS_CAP_PCT", "WEEKLY_LOSS_CAP_PCT",
        "MAX_DRAWDOWN_CAP_PCT", "MAX_RISK_PER_TRADE_CAP_PCT", "MAX_RISK_PER_TRADE",
    ) if key in os.environ}
    os.environ.update(safe_values)
    for key in ("DAILY_LOSS_CAP_PCT", "WEEKLY_LOSS_CAP_PCT", "MAX_DRAWDOWN_CAP_PCT", "MAX_RISK_PER_TRADE_CAP_PCT", "MAX_RISK_PER_TRADE"):
        try:
            value = float(preserved.get(key, safe_values[key]))
            if 0 < value < float(safe_values[key]):
                os.environ[key] = str(value)
        except (TypeError, ValueError):
            pass
    for key in ("ADAPTIVE_MIN_SAMPLE", "SHADOW_MIN_TRAIN", "SHADOW_MIN_VALIDATION", "SHADOW_MIN_COVERAGE"):
        if key in preserved:
            os.environ[key] = preserved[key]
    from src import apply_runtime_safety_floors
    apply_runtime_safety_floors()
    os.environ["MAX_OPEN_TRADES"] = os.environ["MAX_CONCURRENT_POSITIONS"]

    # A restart is not evidence of a deposit or permission to erase losses.
    os.environ["RESET_MAX_DRAWDOWN_HALT"] = "false"
    os.environ["RESET_WEEKLY_LOSS_CAP"] = "false"

    print(
        "[SAFE-DEMO] enforced mode=demo oanda_env=practice "
        "instruments=AUD_USD,GBP_USD session=SOFT aggressive=false "
        "risk_cap_pct=0.5 adaptive_policy=true lifetime_memory=true "
        "shadow_learning=true shadow_auto_apply=false automatic_risk_resets=false",
        flush=True,
    )


_apply_render_safe_demo_profile()


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    OANDA_API_KEY: str = Field(
        "",
        description="OANDA API key used for authenticated requests.",
        validation_alias=AliasChoices("OANDA_API_KEY", "OANDA_API_TOKEN"),
    )
    OANDA_ACCOUNT_ID: str = Field(
        "",
        description="OANDA account identifier.",
        validation_alias=AliasChoices("OANDA_ACCOUNT_ID", "ACCOUNT_ID"),
    )
    OANDA_ENV: str = Field(
        "practice",
        description="Target OANDA environment: practice or live.",
    )
    BASE_URL: str = Field(
        "https://api-fxpractice.oanda.com/v3",
        description="Base REST API URL for OANDA.",
    )
    MODE: str = Field(
        "demo",
        description="Run mode for the bot: demo, live, or simulation.",
    )
    TZ: str = Field(
        "UTC",
        description="Timezone label shown in heartbeat logs.",
    )
    HEARTBEAT_SECONDS: int = Field(
        30,
        description="Seconds between heartbeat log entries.",
    )
    DECISION_SECONDS: int = Field(
        60,
        description="Seconds between strategy evaluation cycles.",
    )
    MAX_SILENCE_SECONDS: int = Field(
        180,
        description="Maximum silence in seconds before watchdog alerts.",
    )
    ERROR_BURST_THRESHOLD: int = Field(
        3,
        description="Errors within the rolling window that trigger alerts.",
    )
    INSTRUMENT: str = Field(
        "EUR_USD",
        description="Primary instrument traded by the worker.",
    )
    ORDER_SIZE: int = Field(
        1000,
        description="Default order size used for demo trades.",
    )
    STRAT_TIMEFRAME: str = Field(
        "M5",
        description="Granularity for fetched candles (OANDA notation).",
    )
    STRAT_EMA_FAST: int = Field(
        12,
        description="Fast EMA lookback length for the crossover strategy.",
    )
    STRAT_EMA_SLOW: int = Field(
        26,
        description="Slow EMA lookback length for the crossover strategy.",
    )
    STRAT_RSI_LEN: int = Field(
        14,
        description="RSI lookback length.",
    )
    STRAT_RSI_BUY: float = Field(
        52.0,
        description="RSI threshold required to issue a BUY signal.",
    )
    STRAT_RSI_SELL: float = Field(
        48.0,
        description="RSI threshold required to issue a SELL signal.",
    )
    ADX_FILTER: float = Field(
        20.0,
        description="Minimum ADX value required before enabling trade signals.",
    )
    ATR_LEN: int = Field(
        14,
        description="ATR lookback length.",
    )
    MIN_ATR: float = Field(
        0.00005,
        description="Minimum ATR required before issuing a trade signal.",
    )
    STRAT_COOLDOWN_BARS: int = Field(
        9,
        description="Bars to wait after a trade before considering a new one.",
    )
    SL_ATR_MULT: float = Field(
        1.2,
        description="ATR multiplier applied to stop loss distance.",
    )
    TP_ATR_MULT: float = Field(
        1.0,
        description="ATR multiplier applied to take profit distance.",
    )
    INSTRUMENT_ATR_MULTIPLIERS: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Optional per-instrument overrides for ATR SL/TP multipliers.",
    )
    METRIC_SUMMARY_INTERVAL: int = Field(
        10,
        description="Decision count between summary metric log lines.",
    )
    ALERT_EMAIL: str = Field(
        "",
        description="Email address to receive watchdog alert emails.",
    )
    SMTP_HOST: str = Field(
        "",
        description="SMTP host used for watchdog alert emails.",
    )
    SMTP_PORT: int = Field(
        587,
        description="SMTP port used for watchdog alert emails.",
    )
    SMTP_USER: str = Field(
        "",
        description="SMTP username for authentication.",
    )
    SMTP_PASS: str = Field(
        "",
        description="SMTP password for authentication.",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))


settings = Settings()
