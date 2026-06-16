from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import drawdown_series
from src.research.monthly_target_martingale_research import monthly_return_from_equity
from src.utils.config_loader import project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_ITERATION_DIR = "reports/research_iterations/iteration_017_fundamental_trend_escape_best_policy_expanded"
DEFAULT_VARIANT = "fundamental_trend_escape_entry_only"


@dataclass(frozen=True)
class MonteCarloConfig:
    iteration_dir: Path
    variant: str
    iterations: int = 5000
    seed: int = 42
    target_monthly_return: float = 0.20
    output_dir: Path | None = None


@dataclass(frozen=True)
class VariantOutputs:
    fold_summary: pd.DataFrame
    trades: pd.DataFrame
    oos_equity: pd.Series


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return project_path(value)


def monthly_return_from_total(total_return: float, days: float) -> float:
    if days <= 0:
        raise ValueError("days must be positive")
    if total_return <= -1:
        return -1.0
    return float((1.0 + float(total_return)) ** (30.4375 / float(days)) - 1.0)


def max_drawdown_from_values(values: list[float] | np.ndarray | pd.Series) -> float:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        raise ValueError("equity values are empty")
    return float(drawdown_series(series).min())


def _require_columns(frame: pd.DataFrame, columns: list[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def load_fold_summary(iteration_dir: Path, variant: str) -> pd.DataFrame:
    path = iteration_dir / f"walk_forward_fold_summary_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"fold summary not found: {path}")
    frame = pd.read_csv(path)
    required = [
        "fold_id",
        "test_start",
        "test_end",
        "test_total_return",
        "test_monthly_return",
        "test_positive",
        "test_target_reached",
        "test_equity_ruined",
    ]
    _require_columns(frame, required, path)
    if frame.empty:
        raise ValueError("fold summary is empty")
    frame = frame.copy()
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame["test_start"] = pd.to_datetime(frame["test_start"], utc=True)
    frame["test_end"] = pd.to_datetime(frame["test_end"], utc=True)
    if frame["test_start"].isna().any() or frame["test_end"].isna().any():
        raise ValueError("fold summary contains invalid timestamps")
    if (frame["test_end"] <= frame["test_start"]).any():
        raise ValueError("fold summary contains non-positive test windows")
    if frame["fold_id"].duplicated().any():
        raise ValueError("fold summary contains duplicated fold_id values")
    return frame.sort_values("fold_id").reset_index(drop=True)


def load_oos_equity(iteration_dir: Path, variant: str) -> pd.Series:
    path = iteration_dir / f"walk_forward_oos_equity_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"OOS equity not found: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, ["timestamp", "equity"], path)
    if frame.empty:
        raise ValueError("OOS equity is empty")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamp.isna().any():
        raise ValueError("OOS equity contains invalid timestamps")
    equity = pd.Series(frame["equity"].astype(float).to_numpy(), index=timestamp, name="equity")
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    if equity.empty:
        raise ValueError("OOS equity is empty after timestamp normalization")
    return equity


def load_trades_by_fold(iteration_dir: Path, variant: str, fold_ids: list[int]) -> pd.DataFrame:
    trade_dir = iteration_dir / "selected_fold_trades"
    if not trade_dir.exists():
        raise FileNotFoundError(f"selected fold trades directory not found: {trade_dir}")
    frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        path = trade_dir / f"{variant}_fold_{fold_id:03d}_trades.csv"
        if not path.exists():
            raise FileNotFoundError(f"trade file not found: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            frame = pd.DataFrame({"realized_pnl": pd.Series(dtype=float)})
        _require_columns(frame, ["realized_pnl"], path)
        frame = frame.copy()
        frame["fold_id"] = int(fold_id)
        if "exit_timestamp" in frame.columns:
            frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True)
        frames.append(frame)
    trades = pd.concat(frames, ignore_index=True)
    if trades.empty:
        raise ValueError("all selected fold trade files are empty")
    if trades["realized_pnl"].isna().any():
        raise ValueError("trades contain missing realized_pnl values")
    trades["realized_pnl"] = trades["realized_pnl"].astype(float)
    return trades


def load_variant_outputs(iteration_dir: Path, variant: str) -> VariantOutputs:
    fold_summary = load_fold_summary(iteration_dir, variant)
    oos_equity = load_oos_equity(iteration_dir, variant)
    trades = load_trades_by_fold(iteration_dir, variant, fold_summary["fold_id"].astype(int).tolist())
    return VariantOutputs(fold_summary=fold_summary, trades=trades, oos_equity=oos_equity)


def observed_days(fold_summary: pd.DataFrame) -> float:
    starts = pd.to_datetime(fold_summary["test_start"], utc=True)
    ends = pd.to_datetime(fold_summary["test_end"], utc=True)
    return float(((ends - starts) / pd.Timedelta(days=1)).sum())


def equity_from_trade_pnl(pnl: np.ndarray) -> np.ndarray:
    pnl = np.asarray(pnl, dtype=float)
    return np.concatenate(([1.0], 1.0 + np.cumsum(pnl)))


def summarize_equity_path(equity: np.ndarray, days: float, target_monthly_return: float) -> dict[str, Any]:
    if len(equity) < 2:
        raise ValueError("equity path must contain at least two points")
    total_return = float(equity[-1] / equity[0] - 1.0) if float(equity[0]) != 0 else -1.0
    monthly_return = monthly_return_from_total(total_return, days)
    max_drawdown = max_drawdown_from_values(equity)
    equity_ruined = bool(np.any(np.asarray(equity, dtype=float) <= 0))
    return {
        "total_return": total_return,
        "monthly_return": monthly_return,
        "max_drawdown": max_drawdown,
        "equity_ruined": equity_ruined,
        "target_reached": bool(monthly_return >= target_monthly_return),
        "positive": bool(total_return > 0),
    }


def fold_bootstrap(
    fold_summary: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
    target_monthly_return: float,
) -> pd.DataFrame:
    returns = fold_summary["test_total_return"].astype(float).to_numpy()
    monthly = fold_summary["test_monthly_return"].astype(float).to_numpy()
    if len(returns) == 0:
        raise ValueError("fold returns are empty")
    days = observed_days(fold_summary)
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        sample_idx = rng.integers(0, len(returns), size=len(returns))
        sampled_returns = returns[sample_idx]
        equity = np.concatenate(([1.0], np.cumprod(1.0 + sampled_returns)))
        summary = summarize_equity_path(equity, days, target_monthly_return)
        summary.update(
            {
                "method": "fold_bootstrap",
                "iteration": iteration,
                "positive_fold_rate": float((sampled_returns > 0).mean()),
                "target_fold_rate": float((monthly[sample_idx] >= target_monthly_return).mean()),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def trade_bootstrap(
    trades: pd.DataFrame,
    days: float,
    iterations: int,
    rng: np.random.Generator,
    target_monthly_return: float,
) -> pd.DataFrame:
    pnl = trades["realized_pnl"].astype(float).to_numpy()
    if len(pnl) == 0:
        raise ValueError("trade pnl is empty")
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        equity = equity_from_trade_pnl(sample)
        summary = summarize_equity_path(equity, days, target_monthly_return)
        summary.update(
            {
                "method": "trade_bootstrap",
                "iteration": iteration,
                "positive_fold_rate": np.nan,
                "target_fold_rate": np.nan,
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def fold_block_bootstrap(
    fold_summary: pd.DataFrame,
    trades: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
    target_monthly_return: float,
) -> pd.DataFrame:
    fold_ids = fold_summary["fold_id"].astype(int).to_numpy()
    if len(fold_ids) == 0:
        raise ValueError("fold ids are empty")
    days = observed_days(fold_summary)
    pnl_by_fold = {
        int(fold_id): trades.loc[trades["fold_id"].astype(int) == int(fold_id), "realized_pnl"].astype(float).to_numpy()
        for fold_id in fold_ids
    }
    monthly_by_fold = {
        int(row["fold_id"]): float(row["test_monthly_return"]) for _, row in fold_summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        sampled_folds = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        current_equity = 1.0
        stitched_values = [current_equity]
        fold_returns: list[float] = []
        target_flags: list[bool] = []
        for fold_id in sampled_folds:
            pnl = pnl_by_fold[int(fold_id)]
            fold_equity = equity_from_trade_pnl(pnl) if len(pnl) else np.array([1.0])
            fold_return = float(fold_equity[-1] - 1.0)
            fold_returns.append(fold_return)
            target_flags.append(monthly_by_fold[int(fold_id)] >= target_monthly_return)
            stitched = current_equity * fold_equity
            stitched_values.extend(stitched[1:].tolist())
            current_equity = float(stitched[-1])
        summary = summarize_equity_path(np.asarray(stitched_values, dtype=float), days, target_monthly_return)
        summary.update(
            {
                "method": "fold_block_bootstrap",
                "iteration": iteration,
                "positive_fold_rate": float((np.asarray(fold_returns) > 0).mean()),
                "target_fold_rate": float(np.asarray(target_flags, dtype=bool).mean()),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def _quantile(series: pd.Series, value: float) -> float:
    return float(series.astype(float).quantile(value))


def summarize_samples(samples: pd.DataFrame) -> list[dict[str, Any]]:
    if samples.empty:
        raise ValueError("Monte Carlo samples are empty")
    rows: list[dict[str, Any]] = []
    for method, frame in samples.groupby("method", sort=True):
        monthly = frame["monthly_return"].astype(float)
        total = frame["total_return"].astype(float)
        drawdown = frame["max_drawdown"].astype(float)
        rows.append(
            {
                "method": method,
                "iterations": int(len(frame)),
                "monthly_return_p05": _quantile(monthly, 0.05),
                "monthly_return_p25": _quantile(monthly, 0.25),
                "monthly_return_p50": _quantile(monthly, 0.50),
                "monthly_return_p75": _quantile(monthly, 0.75),
                "monthly_return_p95": _quantile(monthly, 0.95),
                "total_return_p05": _quantile(total, 0.05),
                "total_return_p50": _quantile(total, 0.50),
                "total_return_p95": _quantile(total, 0.95),
                "max_drawdown_p05": _quantile(drawdown, 0.05),
                "max_drawdown_p50": _quantile(drawdown, 0.50),
                "max_drawdown_p95": _quantile(drawdown, 0.95),
                "positive_probability": float(frame["positive"].astype(bool).mean()),
                "target_probability": float(frame["target_reached"].astype(bool).mean()),
                "ruin_probability": float(frame["equity_ruined"].astype(bool).mean()),
                "avg_positive_fold_rate": float(frame["positive_fold_rate"].mean(skipna=True)),
                "avg_target_fold_rate": float(frame["target_fold_rate"].mean(skipna=True)),
            }
        )
    return rows


def decide_monte_carlo(observed: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    by_method = {row["method"]: row for row in summaries}
    block = by_method.get("fold_block_bootstrap") or by_method.get("fold_bootstrap")
    if block is None:
        raise ValueError("Monte Carlo summaries must include fold_block_bootstrap or fold_bootstrap")
    observed_monthly = float(observed["monthly_return"])
    target = float(observed["target_monthly_return"])
    median = float(block["monthly_return_p50"])
    p05 = float(block["monthly_return_p05"])
    target_probability = float(block["target_probability"])
    ruin_probability = float(block["ruin_probability"])
    if median <= 0 or ruin_probability >= 0.05:
        return "not robust"
    if observed_monthly >= target and target_probability < 0.60:
        return "target likely overfit"
    if observed_monthly < target and target_probability < 0.40:
        return "positive but target not robust"
    if p05 > 0 and target_probability >= 0.60 and ruin_probability <= 0.01:
        return "robust"
    return "fragile positive edge"


def observed_metrics(outputs: VariantOutputs, target_monthly_return: float) -> dict[str, Any]:
    equity = outputs.oos_equity.dropna().astype(float)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) != 0 else -1.0
    monthly_return = monthly_return_from_equity(equity)
    return {
        "fold_count": int(len(outputs.fold_summary)),
        "trade_count": int(len(outputs.trades)),
        "observed_days": observed_days(outputs.fold_summary),
        "total_return": total_return,
        "monthly_return": monthly_return,
        "max_drawdown": float(drawdown_series(equity).min()),
        "positive_fold_rate": float(outputs.fold_summary["test_positive"].astype(bool).mean()),
        "target_fold_rate": float(outputs.fold_summary["test_target_reached"].astype(bool).mean()),
        "equity_ruined": bool((equity <= 0).any() or outputs.fold_summary["test_equity_ruined"].astype(bool).any()),
        "target_monthly_return": float(target_monthly_return),
    }


def run_monte_carlo(config: MonteCarloConfig) -> dict[str, Any]:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.target_monthly_return <= -1:
        raise ValueError("target_monthly_return must be greater than -100%")
    iteration_dir = resolve_path(config.iteration_dir)
    output_dir = resolve_path(config.output_dir) if config.output_dir is not None else iteration_dir / "monte_carlo"
    outputs = load_variant_outputs(iteration_dir, config.variant)
    rng = np.random.default_rng(config.seed)
    days = observed_days(outputs.fold_summary)
    fold_samples = fold_bootstrap(outputs.fold_summary, config.iterations, rng, config.target_monthly_return)
    trade_samples = trade_bootstrap(outputs.trades, days, config.iterations, rng, config.target_monthly_return)
    block_samples = fold_block_bootstrap(outputs.fold_summary, outputs.trades, config.iterations, rng, config.target_monthly_return)
    samples = pd.concat([fold_samples, trade_samples, block_samples], ignore_index=True)
    summaries = summarize_samples(samples)
    observed = observed_metrics(outputs, config.target_monthly_return)
    decision = decide_monte_carlo(observed, summaries)
    payload = {
        "config": {**asdict(config), "iteration_dir": str(iteration_dir), "output_dir": str(output_dir)},
        "observed": observed,
        "monte_carlo_summary": summaries,
        "decision": decision,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_variant = config.variant.replace("/", "_")
    samples.to_csv(output_dir / f"monte_carlo_samples_{safe_variant}.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / f"monte_carlo_summary_{safe_variant}.csv", index=False)
    with (output_dir / f"monte_carlo_payload_{safe_variant}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    write_report(output_dir / f"monte_carlo_report_{safe_variant}.md", payload)
    return payload


def _pct(value: float) -> str:
    return f"{value:.2%}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    observed = payload["observed"]
    summaries = payload["monte_carlo_summary"]
    lines = [
        "# Monte Carlo OOS Robustness",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "## Observed walk-forward",
        "",
        f"- Folds: {observed['fold_count']}",
        f"- Trades: {observed['trade_count']}",
        f"- Monthly return: {_pct(float(observed['monthly_return']))}",
        f"- Total return: {_pct(float(observed['total_return']))}",
        f"- Max drawdown: {_pct(float(observed['max_drawdown']))}",
        f"- Positive fold rate: {_pct(float(observed['positive_fold_rate']))}",
        f"- Target fold rate: {_pct(float(observed['target_fold_rate']))}",
        f"- Equity ruined: {observed['equity_ruined']}",
        "",
        "## Monte Carlo summary",
        "",
        "| method | p05 monthly | median monthly | p95 monthly | target probability | positive probability | ruin probability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {p05} | {p50} | {p95} | {target} | {positive} | {ruin} |".format(
                method=row["method"],
                p05=_pct(float(row["monthly_return_p05"])),
                p50=_pct(float(row["monthly_return_p50"])),
                p95=_pct(float(row["monthly_return_p95"])),
                target=_pct(float(row["target_probability"])),
                positive=_pct(float(row["positive_probability"])),
                ruin=_pct(float(row["ruin_probability"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The runner uses only already selected out-of-sample fold trades and equity curves. "
            "It does not reselect candidates, and therefore tests robustness of the walk-forward result rather than optimizing a new strategy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo robustness checks on existing OOS walk-forward outputs.")
    parser.add_argument("--iteration-dir", default=DEFAULT_ITERATION_DIR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-monthly-return", type=float, default=0.20)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MonteCarloConfig(
        iteration_dir=Path(args.iteration_dir),
        variant=str(args.variant),
        iterations=int(args.iterations),
        seed=int(args.seed),
        target_monthly_return=float(args.target_monthly_return),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    payload = run_monte_carlo(config)
    LOGGER.info("Monte Carlo decision for %s: %s", config.variant, payload["decision"])


if __name__ == "__main__":
    main()
