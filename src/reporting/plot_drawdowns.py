from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_drawdown_curves(equity_files: list[Path], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for file in equity_files:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        if "equity" in df:
            dd = df["equity"] / df["equity"].cummax() - 1
            ax.plot(df.index, dd, label=file.name.replace("_equity.csv", ""))
    ax.set_title("Drawdown curve by strategy")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)

