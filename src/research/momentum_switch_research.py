from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.walk_forward import temporal_train_validation_test_split
from src.data.validate_data import load_processed
from src.utils.config_loader import load_settings, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"


@dataclass(frozen=True)
class MomentumCandidate:
    name: str
    signal_type: str
    params: dict[str, Any]


def monthly_return_from_equity(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or float(equity.iloc[0]) <= 0:
        return 0.0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    if total_return <= -1:
        return -1.0
    days = max((equity.index[-1] - equity.index[0]) / pd.Timedelta(days=1), 1 / 24)
    return float((1 + total_return) ** (30.4375 / days) - 1)


def rsi(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("RSI window must be > 1")
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    relative_strength = up.rolling(window, min_periods=window).mean() / down.rolling(window, min_periods=window).mean()
    return 100 - 100 / (1 + relative_strength)


def make_candidates(config: dict[str, Any]) -> list[MomentumCandidate]:
    search = config["search"]
    base_candidates: list[MomentumCandidate] = []
    include_rsi_momentum = bool(search.get("include_rsi_momentum", True))
    include_rsi_mean_reversion = bool(search.get("include_rsi_mean_reversion", True))
    include_rsi_long_flat = bool(search.get("include_rsi_long_flat", False))
    for window in search["rsi_windows"]:
        for low, high in search["rsi_threshold_pairs"]:
            if low >= high:
                raise ValueError("RSI low threshold must be below high threshold")
            if include_rsi_momentum:
                base_candidates.append(
                    MomentumCandidate(
                        name=f"rsi_momentum_{window}_{low}_{high}",
                        signal_type="rsi_momentum",
                        params={"window": int(window), "low": float(low), "high": float(high)},
                    )
                )
            if include_rsi_mean_reversion:
                base_candidates.append(
                    MomentumCandidate(
                        name=f"rsi_mean_reversion_{window}_{low}_{high}",
                        signal_type="rsi_mean_reversion",
                        params={"window": int(window), "low": float(low), "high": float(high)},
                    )
                )
            if include_rsi_long_flat:
                base_candidates.append(
                    MomentumCandidate(
                        name=f"rsi_long_flat_momentum_{window}_{low}_{high}",
                        signal_type="rsi_long_flat_momentum",
                        params={"window": int(window), "low": float(low), "high": float(high)},
                    )
                )
                base_candidates.append(
                    MomentumCandidate(
                        name=f"rsi_long_flat_mean_reversion_{window}_{low}_{high}",
                        signal_type="rsi_long_flat_mean_reversion",
                        params={"window": int(window), "low": float(low), "high": float(high)},
                    )
                )
    for fast in search["ema_fast_windows"]:
        for slow in search["ema_slow_windows"]:
            if int(fast) >= int(slow):
                continue
            if search["include_long_short"]:
                base_candidates.append(
                    MomentumCandidate(
                        name=f"ema_long_short_{fast}_{slow}",
                        signal_type="ema_long_short",
                        params={"fast": int(fast), "slow": int(slow)},
                    )
                )
            if search["include_long_only"]:
                base_candidates.append(
                    MomentumCandidate(
                        name=f"ema_long_only_{fast}_{slow}",
                        signal_type="ema_long_only",
                        params={"fast": int(fast), "slow": int(slow)},
                    )
                )
    for window in search.get("donchian_windows", []):
        window = int(window)
        if window <= 1:
            raise ValueError("donchian_windows must be > 1")
        if search.get("include_long_short", True):
            base_candidates.append(
                MomentumCandidate(
                    name=f"donchian_long_short_{window}",
                    signal_type="donchian_long_short",
                    params={"window": window},
                )
            )
        if search.get("include_long_only", True):
            base_candidates.append(
                MomentumCandidate(
                    name=f"donchian_long_only_{window}",
                    signal_type="donchian_long_only",
                    params={"window": window},
                )
            )
    include_rsi_ema_momentum = bool(search.get("include_rsi_ema_momentum", True))
    include_rsi_ema_long_only = bool(search.get("include_rsi_ema_long_only", False))
    for window in search.get("rsi_ema_windows", []):
        for low, high in search.get("rsi_ema_threshold_pairs", search["rsi_threshold_pairs"]):
            if low >= high:
                raise ValueError("RSI EMA low threshold must be below high threshold")
            for fast in search.get("rsi_ema_fast_windows", []):
                for slow in search.get("rsi_ema_slow_windows", []):
                    if int(fast) >= int(slow):
                        continue
                    params = {
                        "window": int(window),
                        "low": float(low),
                        "high": float(high),
                        "fast": int(fast),
                        "slow": int(slow),
                    }
                    if include_rsi_ema_momentum:
                        base_candidates.append(
                            MomentumCandidate(
                                name=f"rsi_ema_momentum_{window}_{low}_{high}_{fast}_{slow}",
                                signal_type="rsi_ema_momentum",
                                params=params,
                            )
                        )
                    if include_rsi_ema_long_only:
                        base_candidates.append(
                            MomentumCandidate(
                                name=f"rsi_ema_long_only_{window}_{low}_{high}_{fast}_{slow}",
                                signal_type="rsi_ema_long_only",
                                params=params,
                            )
                        )
    candidates: list[MomentumCandidate] = []
    position_values = search.get("max_position_pcts")
    if position_values:
        for candidate in base_candidates:
            for value in position_values:
                position = float(value)
                if position <= 0 or position > 1:
                    raise ValueError("max_position_pcts must be within (0, 1]")
                params = {**candidate.params, "max_position_pct": position}
                candidates.append(
                    MomentumCandidate(
                        name=f"{candidate.name}_pos{position:g}".replace(".", "p"),
                        signal_type=candidate.signal_type,
                        params=params,
                    )
                )
    else:
        candidates = base_candidates
    if not candidates:
        raise ValueError("No momentum candidates generated")
    return candidates


def build_signal(signal_frame: pd.DataFrame, candidate: MomentumCandidate) -> pd.Series:
    close = signal_frame["close"]
    if candidate.signal_type in {
        "rsi_momentum",
        "rsi_mean_reversion",
        "rsi_long_flat_momentum",
        "rsi_long_flat_mean_reversion",
    }:
        values = rsi(close, int(candidate.params["window"]))
        signal = pd.Series(np.nan, index=signal_frame.index)
        low = float(candidate.params["low"])
        high = float(candidate.params["high"])
        if candidate.signal_type == "rsi_momentum":
            signal[values > high] = 1.0
            signal[values < low] = -1.0
        elif candidate.signal_type == "rsi_mean_reversion":
            signal[values < low] = 1.0
            signal[values > high] = -1.0
        elif candidate.signal_type == "rsi_long_flat_momentum":
            signal[values > high] = 1.0
            signal[values < low] = 0.0
        else:
            signal[values < low] = 1.0
            signal[values > high] = 0.0
        return signal.ffill().fillna(0.0)

    if candidate.signal_type in {"donchian_long_short", "donchian_long_only"}:
        window = int(candidate.params["window"])
        upper = signal_frame["high"].rolling(window, min_periods=window).max().shift(1)
        lower = signal_frame["low"].rolling(window, min_periods=window).min().shift(1)
        signal = pd.Series(np.nan, index=signal_frame.index)
        signal[close > upper] = 1.0
        signal[close < lower] = -1.0 if candidate.signal_type == "donchian_long_short" else 0.0
        return signal.ffill().fillna(0.0)

    if candidate.signal_type in {"rsi_ema_momentum", "rsi_ema_long_only"}:
        values = rsi(close, int(candidate.params["window"]))
        fast = int(candidate.params["fast"])
        slow = int(candidate.params["slow"])
        fast_ema = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
        slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        signal = pd.Series(np.nan, index=signal_frame.index)
        low = float(candidate.params["low"])
        high = float(candidate.params["high"])
        long_condition = (values > high) & (fast_ema > slow_ema)
        signal[long_condition] = 1.0
        if candidate.signal_type == "rsi_ema_momentum":
            signal[(values < low) & (fast_ema < slow_ema)] = -1.0
        else:
            signal[(values < low) | (fast_ema < slow_ema)] = 0.0
        return signal.ffill().fillna(0.0)

    fast = int(candidate.params["fast"])
    slow = int(candidate.params["slow"])
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    if candidate.signal_type == "ema_long_short":
        return np.sign(fast_ema - slow_ema).fillna(0.0)
    if candidate.signal_type == "ema_long_only":
        return (fast_ema > slow_ema).astype(float).fillna(0.0)
    raise ValueError(f"Unsupported signal_type: {candidate.signal_type}")


def backtest_signal(
    base_frame: pd.DataFrame,
    signal: pd.Series,
    config: dict[str, Any],
    max_position_pct: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    execution = config["execution"]
    max_position = float(execution["max_position_pct"] if max_position_pct is None else max_position_pct)
    if max_position <= 0 or max_position > 1:
        raise ValueError("max_position_pct must be within (0, 1]")
    fee_rate = float(execution["fee_rate"])
    slippage_pct = float(execution["slippage_bps"]) / 10000.0
    max_total_loss_pct = float(execution["max_total_loss_pct"])
    if max_total_loss_pct <= 0:
        raise ValueError("max_total_loss_pct must be positive")
    raw_position = signal.reindex(base_frame.index, method="ffill").fillna(0.0).clip(-1, 1)
    returns = base_frame["close"].pct_change().shift(-1).fillna(0.0)

    position = raw_position * max_position
    turnover = position.diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(float(position.iloc[0]))
    costs = turnover * (fee_rate + slippage_pct)
    gross_returns = position * returns
    strategy_returns = gross_returns - costs
    equity = (1 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    killed = drawdown.le(-max_total_loss_pct)
    if killed.any():
        kill_pos = int(np.flatnonzero(killed.to_numpy())[0])
        position = position.copy()
        position.iloc[kill_pos + 1 :] = 0.0
        turnover = position.diff().abs()
        if not turnover.empty:
            turnover.iloc[0] = abs(float(position.iloc[0]))
        costs = turnover * (fee_rate + slippage_pct)
        gross_returns = position * returns
        strategy_returns = gross_returns - costs
        equity = (1 + strategy_returns).cumprod()
        killed_values = np.zeros(len(base_frame), dtype=int)
        killed_values[kill_pos:] = 1
    else:
        killed_values = np.zeros(len(base_frame), dtype=int)
    trades = pd.DataFrame(
        {
            "position": position,
            "turnover": turnover,
            "strategy_return": strategy_returns,
            "gross_return": gross_returns,
            "cost": costs,
            "risk_killed": killed_values,
        },
        index=base_frame.index,
    )
    return equity.rename("equity"), trades


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict[str, Any]:
    returns = trades["strategy_return"]
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    turnover_events = trades["turnover"].gt(0)
    active = trades["position"].abs().gt(0)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "monthly_return": monthly_return_from_equity(equity),
        "expectancy_per_bar": float(returns.mean()),
        "realized_pnl": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "trades": int(turnover_events.sum()),
        "exposure_time": float(active.mean()),
        "average_position": float(trades["position"].abs().mean()),
        "win_bar_rate": float(returns.gt(0).mean()),
        "fees_slippage_paid": float(trades["cost"].sum()),
        "risk_kill_triggered": int(trades["risk_killed"].max()) if "risk_killed" in trades else 0,
    }


def evaluate_candidate(
    base_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    candidate: MomentumCandidate,
    split_indexes: dict[str, pd.Index],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    signal = build_signal(signal_frame, candidate)
    max_position_pct = candidate.params.get("max_position_pct")
    rows: list[dict[str, Any]] = []
    for split_name, index in split_indexes.items():
        split_frame = base_frame.loc[index]
        if split_frame.empty:
            raise ValueError(f"Empty split for {split_name}")
        split_equity, split_trades = backtest_signal(split_frame, signal, config, max_position_pct=max_position_pct)
        rows.append(
            {
                "name": candidate.name,
                "signal_type": candidate.signal_type,
                "params": json.dumps(candidate.params, sort_keys=True),
                "max_position_pct": float(max_position_pct or config["execution"]["max_position_pct"]),
                "split": split_name,
                **summarize(split_equity, split_trades),
            }
        )
    return rows


def select_best(validation_summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_summary.empty:
        raise ValueError("Validation summary is empty")
    if set(validation_summary["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Momentum selection must use validation rows only")
    max_drawdown_constraint = float(config["execution"]["max_drawdown_constraint"])
    eligible = validation_summary[validation_summary["max_drawdown"] >= max_drawdown_constraint].copy()
    if eligible.empty:
        eligible = validation_summary.copy()
    eligible = eligible.sort_values(
        by=["monthly_return", "total_return", "max_drawdown", "trades"],
        ascending=[False, False, False, True],
    )
    selected = eligible.iloc[0].to_dict()
    selected["selected_from_validation_only"] = True
    selected["max_drawdown_constraint"] = max_drawdown_constraint
    return selected


def candidate_from_row(row: dict[str, Any]) -> MomentumCandidate:
    return MomentumCandidate(
        name=str(row["name"]),
        signal_type=str(row["signal_type"]),
        params=json.loads(row["params"]),
    )


def decide(test_metrics: dict[str, Any]) -> str:
    if float(test_metrics.get("total_return", 0.0)) > 0 and float(test_metrics.get("max_drawdown", 0.0)) >= -0.20:
        return "momentum switch candidate viable"
    if float(test_metrics.get("total_return", 0.0)) > 0:
        return "positive but drawdown too high"
    return "momentum switch not viable"


def write_report(output_dir: Path, selected: dict[str, Any], validation_exact: dict[str, Any], test_exact: dict[str, Any], decision: str) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 004 - Momentum Switch Research",
        "",
        "## Decision",
        f"`{decision}`",
        "",
        "## Selected Candidate",
        "```json",
        json.dumps(selected, indent=2, default=str),
        "```",
        "",
        "## Validation Metrics",
        "```json",
        json.dumps(validation_exact, indent=2, default=str),
        "```",
        "",
        "## Test Metrics",
        "```json",
        json.dumps(test_exact, indent=2, default=str),
        "```",
        "",
        "## Interpretation",
        "This is not a martingale/grid entry module. It is a bounded 1h momentum switch tested because grid economics were negative after costs. Signals use closed 1h candles and are applied to 5m returns with fees and slippage on position changes.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frame = load_processed(project_path("data/processed/btcusdt_5m.parquet"))
    signal_frame = load_processed(project_path("data/processed/btcusdt_1h.parquet"))
    settings = load_settings()
    split = temporal_train_validation_test_split(
        base_frame.index,
        train_fraction=float(settings["validation"]["train_fraction"]),
        validation_fraction=float(settings["validation"]["validation_fraction"]),
        embargo_bars=int(settings["validation"]["embargo_bars"]),
    )
    split_indexes = {"validation": split.validation, "test": split.test}
    candidates = make_candidates(config)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(evaluate_candidate(base_frame, signal_frame, candidate, split_indexes, config))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "momentum_search_summary.csv", index=False)
    selected = select_best(summary[summary["split"] == SEARCH_SPLIT], config)
    selected_name = str(selected["name"])
    exact_rows = summary[summary["name"] == selected_name].copy()
    validation_exact = exact_rows[exact_rows["split"] == "validation"].iloc[0].to_dict()
    test_exact = exact_rows[exact_rows["split"] == "test"].iloc[0].to_dict()

    selected_candidate = candidate_from_row(selected)
    signal = build_signal(signal_frame, selected_candidate)
    equity, trades = backtest_signal(base_frame, signal, config)
    equity.to_frame().to_csv(output_dir / "selected_equity_full.csv")
    trades.to_csv(output_dir / "selected_bar_returns_full.csv")

    decision = decide(test_exact)
    payload = {
        "decision": decision,
        "selected_candidate": selected,
        "validation_exact": validation_exact,
        "test_exact": test_exact,
        "candidate_count": len(candidates),
    }
    (output_dir / "selected_momentum_candidate.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, selected, validation_exact, test_exact, decision)
    LOGGER.info("Wrote momentum switch iteration outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded 1h momentum switch research iteration.")
    parser.add_argument("--config", default="config/research_iteration_momentum.yaml")
    args = parser.parse_args()
    payload = run_iteration(args.config)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
