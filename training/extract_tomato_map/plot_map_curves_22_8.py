"""Plot per-epoch validation mAP curves for the 2026-08-22 fine-tuning sweep
(log_finetuning_v{1,2,3}_1_22_8.txt -- togrow0-6 / yolo26n / togrow0005 on
the new Laboro Tomato dataset, 100 epochs each). Reads
finetuning_epochs_22_8.json (written by report_finetuning_22_8.py).

Usage:
    python3 report_finetuning_22_8.py   # if not already run
    python3 plot_map_curves_22_8.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EPOCHS_JSON = ROOT / "finetuning_epochs_22_8.json"
OUT_PNG = ROOT / "map_curves_22_8_v1_v2_v3.png"
OUT_PDF = ROOT / "map_curves_22_8_v1_v2_v3.pdf"

METRIC = "mAP50_95"
METRIC_LABEL = {"mAP50_95": r"mAP@0.5:0.95", "mAP50": r"mAP@0.5"}[METRIC]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a5cd6"]
PANEL_TITLE = {
    "v1": "(a) yolo26n + togrow0-3 (attention)",
    "v2": "(b) togrow4-6 (width scaling)",
    "v3": "(c) togrow0005 (SSL pretrain x2, v3_2 patched)",
}

# EMA smoothing factor (TensorBoard-style). Higher = smoother.
SMOOTH_WEIGHT = 0.9


def ema(values: list[float], weight: float = SMOOTH_WEIGHT) -> list[float]:
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def main() -> None:
    data = json.loads(EPOCHS_JSON.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)

    for ax, key in zip(axes, ["v1", "v2", "v3"]):
        runs = data[key]["runs"]
        name_counts = {}
        for r in runs:
            name_counts[r["name"]] = name_counts.get(r["name"], 0) + 1
        for i, run in enumerate(runs):
            if name_counts[run["name"]] > 1:
                cfg_hint = Path(run["cfg"]).stem
                label = f"{run['name']} ({cfg_hint})"
            else:
                label = run["name"]
            color = COLORS[i % len(COLORS)]
            ax.plot(run["epochs"], run[METRIC], color=color, linewidth=0.6, alpha=0.3, zorder=1)
            ax.plot(run["epochs"], ema(run[METRIC]), color=color, linewidth=1.6, label=label, zorder=2)
        ax.set_title(PANEL_TITLE[key])
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=7.5)

    axes[0].set_ylabel(METRIC_LABEL)
    fig.suptitle("Fine-tuning trên Laboro Tomato (3 lớp độ chín) — log 22/8", y=1.03)

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
