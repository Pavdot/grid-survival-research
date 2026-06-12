from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GridRiskConfig:
    taker_fee: float
    maker_fee: float
    slippage_bps: float
    spacing_atr_multiplier: float
    max_levels: int
    base_position_size_pct: float
    sizing_sequence: tuple[float, ...]
    max_grid_loss_pct: float
    max_daily_loss_pct: float
    max_total_exposure_pct: float
    max_holding_hours: float
    stop_on_regime_break: bool
    stop_on_volatility_shock: bool

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 10000.0


def _is_exponential_like(sequence: list[float]) -> bool:
    if len(sequence) < 3:
        return False
    ratios = [sequence[i] / sequence[i - 1] for i in range(1, len(sequence)) if sequence[i - 1] != 0]
    return bool(ratios) and min(ratios) >= 1.8 and max(ratios) <= 2.2


def validate_strategy_config(config: dict[str, Any]) -> GridRiskConfig:
    grid = config["grid"]
    risk = config["risk"]
    fees = config["fees"]

    if grid.get("allow_exponential_martingale", False):
        raise ValueError("Exponential martingale is explicitly forbidden")
    if grid.get("sizing_mode") != "linear":
        raise ValueError("Only bounded linear sizing is allowed in this MVP")

    sequence = [float(value) for value in grid["sizing_sequence"]]
    if _is_exponential_like(sequence) or sequence[:5] == [1, 2, 4, 8, 16]:
        raise ValueError("Exponential 1/2/4/8/16-style sizing is forbidden")

    max_levels = int(grid["max_levels"])
    if max_levels <= 0:
        raise ValueError("grid.max_levels must be positive")
    if len(sequence) < max_levels:
        raise ValueError("sizing_sequence must provide at least max_levels entries")

    max_grid_loss = float(risk["max_grid_loss_pct"])
    max_exposure = float(risk["max_total_exposure_pct"])
    base_size = float(grid["base_position_size_pct"])
    if max_grid_loss <= 0:
        raise ValueError("risk.max_grid_loss_pct is required and must be positive")
    if max_exposure <= 0 or max_exposure > 1:
        raise ValueError("risk.max_total_exposure_pct must be within (0, 1]")
    if base_size <= 0:
        raise ValueError("grid.base_position_size_pct must be positive")
    if base_size > max_exposure:
        raise ValueError("grid.base_position_size_pct cannot exceed max_total_exposure_pct")
    if float(risk["max_holding_hours"]) <= 0:
        raise ValueError("risk.max_holding_hours must be positive")

    return GridRiskConfig(
        taker_fee=float(fees["taker_fee"]),
        maker_fee=float(fees["maker_fee"]),
        slippage_bps=float(fees["slippage_bps"]),
        spacing_atr_multiplier=float(grid["spacing_atr_multiplier"]),
        max_levels=max_levels,
        base_position_size_pct=base_size,
        sizing_sequence=tuple(sequence[:max_levels]),
        max_grid_loss_pct=max_grid_loss,
        max_daily_loss_pct=float(risk["max_daily_loss_pct"]),
        max_total_exposure_pct=max_exposure,
        max_holding_hours=float(risk["max_holding_hours"]),
        stop_on_regime_break=bool(risk["stop_on_regime_break"]),
        stop_on_volatility_shock=bool(risk["stop_on_volatility_shock"]),
    )


def default_level_sizes(risk: GridRiskConfig, constant: bool = False) -> list[float]:
    if constant:
        return [risk.base_position_size_pct] * risk.max_levels
    return [risk.base_position_size_pct * value for value in risk.sizing_sequence]
