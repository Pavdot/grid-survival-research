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
    candidates: list[MomentumCandidate] = []
    for window in search["rsi_windows"]:
        for low, high in search["rsi_threshold_pairs"]:
            if low >= high:
                raise ValueError("RSI low threshold must be below high threshold")
            candidates.append(
                MomentumCandidate(
                    name=f"rsi_momentum_{window}_{low}_{high}",
                    signal_type="rsi_momentum",
                    params={"window": int(window), "low": float(low), "high": float(high)},
                )
            )
            candidates.append(
                MomentumCandidate(
                    name=f"rsi_mean_reversion_{window}_{low}_{high}",
                    signal_type="rsi_mean_reversion",
                    params={"window": int(window), "low": float(low), "high": float(high)},
                )
            )
    for fast in search["ema_fast_windows"]:
        for slow in search["ema_slow_windows"]:
            if int(fast) >= int(slow):
                continue
            if search["include_long_short"]:
                candidates.append(
                    MomentumCandidate(
                        name=f"ema_long_short_{fast}_{slow}",
                        signal_type="ema_long_short",
                        params={"fast": int(fast), "slow": int(slow)},
                    )
                )
            if search["include_long_only"]:
                candidates.append(
                    MomentumCandidate(
                        name=f"ema_long_only_{fast}_{slow}",
                        signal_type="ema_long_only",
                        params={"fast": int(fast), "slow": int(slow)},
                    )
                )
    if not candidates:
        raise ValueError("No momentum candidates generated")
    return candidates


def build_signal(signal_frame: pd.DataFrame, candidate: MomentumCandidate) -> pd.Series:
    close = signal_frame["close"]
    if candidate.signal_type in {"rsi_momentum", "rsi_mean_reversion"}:
        values = rsi(close, int(candidate.params["window"]))
        signal = pd.Series(0.0, index=signal_frame.index)
        low = float(candidate.params["low"])
        high = float(candidate.params["high"])
        if candidate.signal_type == "rsi_momentum":
            signal[values > high] = 1.0
            signal[values < low] = -1.0
        else:
            signal[values < low] = 1.0
            signal[values > high] = -1.0
        return signal.replace(0, np.nan).ffill().fillna(0.0)

    fast = int(candidate.params["fast"])
    slow = int(candidate.params["slow"])
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    if candidate.signal_type == "ema_long_short":
        return np.sign(fast_ema - slow_ema).fillna(0.0)
    if candidate.signal_type == "ema_long_only":
        return (fast_ema > slow_ema).astype(float).fillna(0.0)
    raise ValueError(f"Unsupported signal_type: {candidate.signal_type}")


def backtest_signal(base_frame: pd.DataFrame, signal: pd.Series, config: dict[str, Any]) -> tuple[pd.Series, pd.DataFrame]:
    execution = config["execution"]
    max_position = float(execution["max_position_pct"])
    if max_position <= 0 or max_position > 1:
        raise ValueError("max_position_pct must be within (0, 1]")
    fee_rate = float(execution["fee_rate"])
    slippage_pct = float(execution["slippage_bps"]) / 10000.0
    max_total_loss_pct = float(execution["max_total_loss_pct"])
    if max_total_loss_pct <= 0:
        raise ValueError("max_total_loss_pct must be positive")
    raw_position = signal.reindex(base_frame.index, method="ffill").fillna(0.0).clip(-1, 1)
    returns = base_frame["close"].pct_change().shift(-1).fillna(0.0)

    position_values: list[float] = []
    turnover_values: list[float] = []
    strategy_return_values: list[float] = []
    gross_return_values: list[float] = []
    cost_values: list[float] = []
    killed_values: list[int] = []
    equity_values: list[float] = []
    previous_position = 0.0
    current_equity = 1.0
    peak_equity = 1.0
    killed = False

    for timestamp in base_frame.index:
        desired_position = 0.0 if killed else float(raw_position.loc[timestamp]) * max_position
        turnover = abs(desired_position - previous_position)
        cost = turnover * (fee_rate + slippage_pct)
        gross_return = desired_position * float(returns.loc[timestamp])
        strategy_return = gross_return - cost
        current_equity *= 1 + strategy_return
        peak_equity = max(peak_equity, current_equity)
        drawdown = current_equity / peak_equity - 1
        if drawdown <= -max_total_loss_pct:
            killed = True
        position_values.append(desired_position)
        turnover_values.append(turnover)
        strategy_return_values.append(strategy_return)
        gross_return_values.append(gross_return)
        cost_values.append(cost)
        killed_values.append(int(killed))
        equity_values.append(current_equity)
        previous_position = desired_position

    position = pd.Series(position_values, index=base_frame.index)
    turnover = pd.Series(turnover_values, index=base_frame.index)
    strategy_returns = pd.Series(strategy_return_values, index=base_frame.index)
    gross_returns = pd.Series(gross_return_values, index=base_frame.index)
    costs = pd.Series(cost_values, index=base_frame.index)
    equity = pd.Series(equity_values, index=base_frame.index)
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
    drawdown = equity / equity.cummax() - 1
    turnover_events = trades["turnover"].gt(0)
    active = trades["position"].abs().gt(0)
    return {
        "total_return": float(equity.iloc[-1] - 1),
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
    rows: list[dict[str, Any]] = []
    for split_name, index in split_indexes.items():
        split_frame = base_frame.loc[index]
        if split_frame.empty:
            raise ValueError(f"Empty split for {split_name}")
        split_equity, split_trades = backtest_signal(split_frame, signal, config)
        rows.append(
            {
                "name": candidate.name,
                "signal_type": candidate.signal_type,
                "params": json.dumps(candidate.params, sort_keys=True),
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
        by=["total_return", "max_drawdown", "trades"],
        ascending=[False, False, True],
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
