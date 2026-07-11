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
    """Force the approved demo profile before any runtime module reads env vars.

    Render service-level environment variables can outlive changes to render.yaml.
    This profile prevents stale aggressive/test settings from overriding the
    approved OANDA practice configuration. It is enabled automatically on Render
    and may be explicitly controlled with MOSSY_SAFE_DEMO_PROFILE.
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
        "ALLOW_HIGH_RISK": "false",
        "DAILY_LOSS_CAP_PCT": "0.01",
        "WEEKLY_LOSS_CAP_PCT": "0.03",
        "MAX_DRAWDOWN_CAP_PCT": "0.05",
        "MAX_OPEN_TRADES": "3",
        "COOLDOWN_CANDLES": "9",
        "TP_ENABLED": "true",
        "ADAPTIVE_TUNING_ENABLED": "true",
        "VERBOSE_MARKET_LOGS": "false",
        "OPEN_TRADES_CACHE_TTL_SECONDS": "15",
    }
    os.environ.update(safe_values)

    # Clear the pre-hardening max-drawdown latch once, while preserving all
    # future drawdown halts. The marker is stored beside the persistent journal.
    state_root_value = os.getenv("MOSSY_STATE_PATH")
    if state_root_value:
        state_root = Path(state_root_value)
    elif Path("/var/data").exists():
        state_root = Path("/var/data")
    else:
        state_root = Path("data")

    marker = state_root / ".safe_demo_profile_20260711_applied"
    reset_requested = False
    try:
        state_root.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            os.environ["RESET_MAX_DRAWDOWN_HALT"] = "true"
            marker.write_text("safe demo profile applied\n", encoding="utf-8")
            reset_requested = True
        else:
            os.environ["RESET_MAX_DRAWDOWN_HALT"] = "false"
    except OSError as exc:
        # If the marker cannot be written, do not repeatedly clear a genuine
        # future halt. Operators can still request a reset explicitly in Render.
        os.environ["RESET_MAX_DRAWDOWN_HALT"] = "false"
        print(f"[SAFE-DEMO][WARN] unable to write migration marker: {exc}", flush=True)

    print(
        "[SAFE-DEMO] enforced mode=demo oanda_env=practice "
        "instruments=AUD_USD,GBP_USD session=SOFT aggressive=false "
        "risk_cap_pct=0.5 one_time_drawdown_reset="
        f"{str(reset_requested).lower()}",
        flush=True,
    )


_apply_render_safe_demo_profile()


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    # ------------------------------------------------------------------
    # OANDA connectivity
    # ------------------------------------------------------------------
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
        description="Base REST API URL for OANDA requests.",
    )
    MODE: str = Field(
        "demo",
        description="Run mode for the bot: demo, live, or simulation.",
    )

    # ------------------------------------------------------------------
    # Logging & scheduling
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Strategy inputs
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    ALERT_EMAIL: str = Field(
        "",
        description="Email address to receive watchdog alerts.",
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

    # Trading parameters
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))


settings = Settings()
