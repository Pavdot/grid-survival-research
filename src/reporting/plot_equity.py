from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(equity_files: list[Path], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for file in equity_files:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        if "equity" in df:
            ax.plot(df.index, df["equity"], label=file.name.replace("_equity.csv", ""))
    ax.set_title("Equity curve by strategy")
    ax.set_ylabel("Equity")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)

