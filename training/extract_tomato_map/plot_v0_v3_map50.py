"""Thesis-ready figure: mAP@0.5 vs epoch for togrow0-3 (labeled v0-v3), the
attention-module sweep from log_finetuning_v1_1_22_8.txt (Laboro Tomato,
100 epochs each). Reads finetuning_epochs_22_8.json (written by
report_finetuning_22_8.py -- rerun that first if the source logs changed).

Usage:
    python3 plot_v0_v3_map50.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EPOCHS_JSON = ROOT / "finetuning_epochs_22_8.json"
OUT_PNG = ROOT / "map50_v0_v6.png"
OUT_PDF = ROOT / "map50_v0_v6.pdf"

# Two panels: (json source key, run_name in the log, display label) per panel.
PANEL_A = [
    ("v1", "togrow0_coco_finetune", "v0"),
    ("v1", "togrow1_coco_finetune", "v1"),
    ("v1", "togrow2_coco_finetune", "v2"),
    ("v1", "togrow3_coco_finetune", "v3"),
]
PANEL_B = [
    ("v2", "togrow4_coco_finetune", "v4"),
    ("v2", "togrow5_coco_finetune", "v5"),
    ("v2", "togrow6_coco_finetune", "v6"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]

MAX_EPOCH = 80

# EMA smoothing factor (TensorBoard-style). Higher = smoother.
SMOOTH_WEIGHT = 0.85


def ema(values: list[float], weight: float = SMOOTH_WEIGHT) -> list[float]:
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12.5,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def main() -> None:
    data = json.loads(EPOCHS_JSON.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5), sharey=True)

    for ax, panel in zip(axes, (PANEL_A, PANEL_B)):
        for i, (source, run_name, label) in enumerate(panel):
            run = next(r for r in data[source]["runs"] if r["name"] == run_name)
            epochs = [e for e in run["epochs"] if e <= MAX_EPOCH]
            values = run["mAP50"][: len(epochs)]
            color = COLORS[i % len(COLORS)]
            ax.plot(
                epochs, ema(values), color=color, linewidth=1.8,
                label=f"{label} ({run_name.replace('_coco_finetune', '')})",
            )

        ax.set_xlabel("Epoch")
        ax.set_xlim(1, MAX_EPOCH)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right")

    axes[0].set_ylabel(r"mAP@0.5")

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
