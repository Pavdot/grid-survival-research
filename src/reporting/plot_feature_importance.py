from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(csv_path: Path, output: Path, top_n: int = 20) -> None:
    importance = pd.read_csv(csv_path).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1])
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)

