from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.labeling.grid_engine import _realize
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/research_iteration_cost_ledger_validation_029a.yaml"


@dataclass(frozen=True)
class LedgerOrder:
    event_type: str
    side: str
    action: str
    reference_price: float
    notional_pct: float
    execution_style: str
    fee_rate: float
    slippage_bps: float
    spread_bps: float
    fill_fraction: float = 1.0
    allow_partial: bool = False


@dataclass(frozen=True)
class LedgerCase:
    case_id: str
    description: str
    orders: tuple[LedgerOrder, ...]
    expected_status: str = "ok"


def _is_buy(side: str, action: str) -> bool:
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if action not in {"entry", "exit"}:
        raise ValueError("action must be entry or exit")
    return (side == "long" and action == "entry") or (side == "short" and action == "exit")


def execution_price(order: LedgerOrder) -> float:
    if order.execution_style not in {"maker", "taker"}:
        raise ValueError("execution_style must be maker or taker")
    mid = float(order.reference_price)
    half_spread = mid * float(order.spread_bps) / 20000.0
    buy = _is_buy(order.side, order.action)
    if order.execution_style == "maker":
        price = mid - half_spread if buy else mid + half_spread
    else:
        price = mid + half_spread if buy else mid - half_spread
    slippage = float(order.slippage_bps) / 10000.0
    return float(price * (1 + slippage) if buy else price * (1 - slippage))


def validate_order(order: LedgerOrder) -> None:
    if order.notional_pct <= 0:
        raise ValueError("notional_pct must be positive")
    if order.reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if order.fee_rate < 0 or order.slippage_bps < 0 or order.spread_bps < 0:
        raise ValueError("fee, slippage and spread must be non-negative")
    if order.fill_fraction < 0 or order.fill_fraction > 1:
        raise ValueError("fill_fraction must stay within [0, 1]")
    if 0 < order.fill_fraction < 1 and not order.allow_partial:
        raise ValueError("partial fills are not supported by this ledger")


def order_cost_breakdown(order: LedgerOrder) -> dict[str, float | str]:
    validate_order(order)
    filled_notional = float(order.notional_pct) * float(order.fill_fraction)
    if filled_notional == 0:
        return {
            **asdict(order),
            "fill_price": np.nan,
            "filled_notional_pct": 0.0,
            "quantity": 0.0,
            "fees": 0.0,
            "spread_cost": 0.0,
            "slippage_cost": 0.0,
        }
    price = execution_price(order)
    quantity = filled_notional / price
    reference = float(order.reference_price)
    taker_half_spread = reference * float(order.spread_bps) / 20000.0
    spread_cost = abs(taker_half_spread * quantity)
    slippage_cost = filled_notional * float(order.slippage_bps) / 10000.0
    return {
        **asdict(order),
        "fill_price": price,
        "filled_notional_pct": filled_notional,
        "quantity": quantity,
        "fees": filled_notional * float(order.fee_rate),
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
    }


def reconcile_case(case: LedgerCase) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fills: list[tuple[float, float]] = []
    entry_fees = 0.0
    exit_order: LedgerOrder | None = None
    error: str | None = None
    try:
        for order_index, order in enumerate(case.orders):
            row = order_cost_breakdown(order)
            row["case_id"] = case.case_id
            row["order_index"] = order_index
            rows.append(row)
            if order.fill_fraction == 0:
                continue
            if order.action == "entry":
                fills.append((float(row["fill_price"]), float(row["quantity"])))
                entry_fees += float(row["fees"])
            else:
                exit_order = order
    except ValueError as exc:
        error = str(exc)

    if error:
        ledger = pd.DataFrame(rows)
        summary = {
            "case_id": case.case_id,
            "status": "expected_rejected" if case.expected_status == "rejected" else "failed",
            "error": error,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "slippage_cost": 0.0,
            "spread_cost": 0.0,
            "net_pnl": 0.0,
            "equity_delta": 0.0,
            "engine_pnl": 0.0,
            "abs_diff": 0.0 if case.expected_status == "rejected" else np.inf,
        }
        return ledger, summary

    if not fills or exit_order is None:
        raise ValueError(f"{case.case_id} must contain at least one filled entry and one exit")
    exit_row = order_cost_breakdown(exit_order)
    side = str(case.orders[0].side)
    engine_pnl, engine_fees = _realize(
        fills,
        float(exit_row["fill_price"]),
        float(exit_order.fee_rate),
        entry_fees,
        side=side,
    )
    if side == "short":
        gross = sum(qty * (fill_price - float(exit_row["fill_price"])) for fill_price, qty in fills)
    else:
        gross = sum(qty * (float(exit_row["fill_price"]) - fill_price) for fill_price, qty in fills)
    fees = float(engine_fees)
    net = float(gross - fees)
    ledger = pd.DataFrame(rows)
    spread_cost = float(pd.to_numeric(ledger["spread_cost"], errors="coerce").fillna(0).sum())
    slippage_cost = float(pd.to_numeric(ledger["slippage_cost"], errors="coerce").fillna(0).sum())
    summary = {
        "case_id": case.case_id,
        "status": "ok",
        "error": "",
        "gross_pnl": float(gross),
        "fees": fees,
        "slippage_cost": slippage_cost,
        "spread_cost": spread_cost,
        "net_pnl": net,
        "equity_delta": net,
        "engine_pnl": float(engine_pnl),
        "abs_diff": abs(net - float(engine_pnl)),
    }
    return ledger, summary


def default_cases(config: dict[str, Any]) -> list[LedgerCase]:
    ledger = config.get("ledger", {})
    fee = float(ledger.get("default_fee_rate", 0.0001))
    slip = float(ledger.get("default_slippage_bps", 0.5))
    spread = float(ledger.get("default_spread_bps", 0.2))
    return [
        LedgerCase(
            "long_taker_tp",
            "Long taker entry exits at take profit.",
            (
                LedgerOrder("entry", "long", "entry", 100.0, 1.0, "taker", fee, slip, spread),
                LedgerOrder("take_profit", "long", "exit", 101.0, 1.0, "taker", fee, slip, spread),
            ),
        ),
        LedgerCase(
            "short_taker_tp",
            "Short taker entry exits at take profit.",
            (
                LedgerOrder("entry", "short", "entry", 100.0, 1.0, "taker", fee, slip, spread),
                LedgerOrder("take_profit", "short", "exit", 99.0, 1.0, "taker", fee, slip, spread),
            ),
        ),
        LedgerCase(
            "long_add_forced_stop",
            "Long grid adds a level and exits with forced stop.",
            (
                LedgerOrder("entry", "long", "entry", 100.0, 1.0, "taker", fee, slip, spread),
                LedgerOrder("add", "long", "entry", 99.0, 1.15, "taker", fee, slip, spread),
                LedgerOrder("max_loss", "long", "exit", 97.5, 2.15, "taker", fee, slip, spread),
            ),
        ),
        LedgerCase(
            "maker_entry_taker_exit",
            "Maker-first entry with taker exit.",
            (
                LedgerOrder("entry", "long", "entry", 100.0, 1.0, "maker", fee / 2.0, 0.0, spread),
                LedgerOrder("take_profit", "long", "exit", 100.8, 1.0, "taker", fee, slip, spread),
            ),
        ),
        LedgerCase(
            "missed_add_then_tp",
            "Missed add has zero cost and zero quantity.",
            (
                LedgerOrder("entry", "long", "entry", 100.0, 1.0, "taker", fee, slip, spread),
                LedgerOrder("add", "long", "entry", 99.0, 1.15, "maker", fee / 2.0, 0.0, spread, fill_fraction=0.0),
                LedgerOrder("take_profit", "long", "exit", 100.7, 1.0, "taker", fee, slip, spread),
            ),
        ),
        LedgerCase(
            "partial_fill_rejected",
            "Partial fill is intentionally rejected in this v1 ledger.",
            (
                LedgerOrder("entry", "long", "entry", 100.0, 1.0, "maker", fee / 2.0, 0.0, spread, fill_fraction=0.5),
                LedgerOrder("take_profit", "long", "exit", 100.5, 0.5, "taker", fee, slip, spread),
            ),
            expected_status="rejected",
        ),
    ]


def run_validation(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ledgers: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for case in default_cases(config):
        ledger, summary = reconcile_case(case)
        ledgers.append(ledger)
        summaries.append(summary)
    ledger_frame = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    scenario_tol = float(config["ledger"].get("scenario_tolerance", 1e-10))
    aggregate_tol = float(config["ledger"].get("aggregate_tolerance", 1e-8))
    ok_rows = summary_frame[summary_frame["status"].eq("ok")]
    aggregate_diff = float(ok_rows["abs_diff"].sum()) if not ok_rows.empty else 0.0
    failed = summary_frame[
        (~summary_frame["status"].isin(["ok", "expected_rejected"]))
        | (summary_frame["status"].eq("ok") & summary_frame["abs_diff"].gt(scenario_tol))
    ]
    passed = failed.empty and aggregate_diff <= aggregate_tol
    ledger_frame.to_csv(output_dir / "ledger_cases.csv", index=False)
    summary_frame.to_csv(output_dir / "ledger_reconciliation.csv", index=False)
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 029A - Cost Ledger Validation",
        "",
        f"- decision: `{'passed' if passed else 'failed'}`",
        f"- scenario_tolerance: `{scenario_tol}`",
        f"- aggregate_tolerance: `{aggregate_tol}`",
        f"- aggregate_abs_diff: `{aggregate_diff:.12g}`",
        f"- cases: `{len(summary_frame)}`",
        "",
        "Partial fills are deliberately rejected until the execution simulator supports them explicitly.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Cost ledger reconciliation failed")
    return {
        "passed": passed,
        "aggregate_abs_diff": aggregate_diff,
        "ledger_cases": str(output_dir / "ledger_cases.csv"),
        "ledger_reconciliation": str(output_dir / "ledger_reconciliation.csv"),
        "report": str(report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deterministic cost ledger scenarios.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_validation(load_yaml(args.config))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
