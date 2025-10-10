from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _normalize_risk_value(raw: float) -> Tuple[float, float]:
    """Return (fractional_value, display_percent)."""

    if raw is None:
        return 0.0, 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if value <= 0.0:
        return 0.0, 0.0
    if value <= 0.05:
        # Already expressed as a decimal fraction (e.g. 0.0025 -> 0.25%).
        return value, value * 100.0
    # Expressed as a percent value (e.g. 0.25 -> 0.25%).
    return value / 100.0, value


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_bound(value: float) -> str:
    if not isfinite(value):
        return "∞"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}m"
    if value >= 1_000:
        trimmed = value / 1_000
        rounded = round(trimmed)
        if abs(trimmed - rounded) < 1e-3:
            return f"{int(rounded)}k"
        return f"{trimmed:.1f}k"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.0f}"


@dataclass(frozen=True)
class RiskTier:
    name: str
    min_equity: float
    max_equity: float
    risk_fraction: float
    display_percent: float

    @property
    def range_label(self) -> str:
        upper = "∞" if not isfinite(self.max_equity) else _format_bound(self.max_equity)
        lower = _format_bound(self.min_equity)
        if not isfinite(self.max_equity):
            return f"{lower}+"
        return f"{lower}-{upper}"

    def contains(self, equity: float) -> bool:
        if equity < self.min_equity:
            return False
        if not isfinite(self.max_equity):
            return True
        if equity == self.max_equity:
            # Upper bound is exclusive to avoid overlapping tiers unless max is inf.
            return False
        return equity < self.max_equity


class RiskScaler:
    DEFAULT_TIERS: Sequence[Dict[str, float]] = (
        {"min": 0, "max": 2_000, "risk_pct": 0.25},
        {"min": 2_000, "max": 4_000, "risk_pct": 0.5},
        {"min": 4_000, "max": 6_000, "risk_pct": 0.75},
        {"min": 6_000, "max": 10_000, "risk_pct": 1.0},
        {"min": 10_000, "max": float("inf"), "risk_pct": 1.25},
    )

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        default_risk_pct: Optional[float] = None,
    ) -> None:
        self._config = config or {}
        self._block = dict(self._config.get("risk_scaler", {}) or {})
        raw_default = default_risk_pct
        if raw_default is None:
            raw_default = (
                ((self._config.get("risk") or {}).get("risk_per_trade_pct"))
                if isinstance(self._config, Mapping)
                else None
            )
        fraction, display = _normalize_risk_value(raw_default or 0.0)
        self._default_fraction = fraction
        self._default_display = display
        self.enabled = bool(self._block.get("enabled", False))
        tiers = self._build_tiers(self._block.get("tiers")) if self.enabled else []
        if self.enabled and not tiers:
            tiers = self._build_tiers(self.DEFAULT_TIERS)
        self._tiers: List[RiskTier] = tiers
        self._last_tier: Optional[RiskTier] = None

    def _build_tiers(self, spec: Optional[Iterable[Mapping[str, Any]]]) -> List[RiskTier]:
        if not spec:
            return []
        tiers: List[RiskTier] = []
        for index, tier_spec in enumerate(spec):
            try:
                min_eq = _coerce_float(tier_spec.get("min"), 0.0)
                raw_max = tier_spec.get("max")
                max_eq = float("inf") if raw_max is None else _coerce_float(raw_max, float("inf"))
                raw_risk = tier_spec.get("risk_pct")
            except AttributeError:
                continue
            fraction, display = _normalize_risk_value(raw_risk)
            if fraction <= 0:
                continue
            if max_eq <= min_eq:
                max_eq = float("inf")
            tiers.append(
                RiskTier(
                    name=f"Tier{len(tiers) + 1}",
                    min_equity=min_eq,
                    max_equity=max_eq,
                    risk_fraction=fraction,
                    display_percent=display,
                )
            )
        tiers.sort(key=lambda t: t.min_equity)
        # Re-label tiers after sorting to keep names monotonic.
        relabeled: List[RiskTier] = []
        for idx, tier in enumerate(tiers, start=1):
            relabeled.append(
                RiskTier(
                    name=f"Tier{idx}",
                    min_equity=tier.min_equity,
                    max_equity=tier.max_equity,
                    risk_fraction=tier.risk_fraction,
                    display_percent=tier.display_percent,
                )
            )
        return relabeled

    @property
    def tiers(self) -> Sequence[RiskTier]:
        return tuple(self._tiers)

    @property
    def last_tier(self) -> Optional[RiskTier]:
        return self._last_tier

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled and bool(self._tiers),
            "default_risk_pct": self._default_display,
            "tiers": [
                {
                    "name": tier.name,
                    "min": tier.min_equity,
                    "max": tier.max_equity if isfinite(tier.max_equity) else None,
                    "risk_pct": tier.display_percent,
                    "range": tier.range_label,
                }
                for tier in self._tiers
            ],
        }

    def get_risk_pct(self, equity: float) -> float:
        self._last_tier = None
        try:
            value = float(equity)
        except (TypeError, ValueError):
            return self._default_fraction
        if value <= 0 or not self._tiers:
            return self._default_fraction
        for tier in self._tiers:
            if tier.contains(value):
                self._last_tier = tier
                return tier.risk_fraction
        # Fall back to the highest tier.
        self._last_tier = self._tiers[-1]
        return self._last_tier.risk_fraction

    @staticmethod
    def to_percent(risk_fraction: float) -> float:
        try:
            value = float(risk_fraction)
        except (TypeError, ValueError):
            return 0.0
        if value <= 0:
            return 0.0
        return value * 100.0

