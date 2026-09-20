"""Plot per-epoch training loss for the togroth COCO-pretrain sweep.

log_traintogtorh_optim.txt / v2 / v3 are the COCO-scale pretraining runs
(val=False -> no per-epoch validation, so training loss is the only
per-epoch signal available) for each togroth width/depth variant. Runs are
split by "engine/trainer:" line (one per actual `model.train()` call) rather
than the "TRAIN <name>" banners, since some banners are printed for configs
that get skipped (checkpoint already exists) and never actually train.

train-9 (togroth_m3_width_0200, in the first log) stops at epoch 11/50 and
is never resumed -> train-10 in the second log retrains the same config
from epoch 1. That abandoned run is kept in the plot (as a short line that
stops at epoch 11) since it's a second, independent example -- on top of
the SSL sweep -- of a run that was interrupted and restarted from scratch
rather than resumed.

Usage:
    python plot_train_loss_togroth.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# (panel title, filename)
FILES = [
    ("log_traintogtorh_optim.txt", ROOT / "log_traintogtorh_optim.txt"),
    ("log_traintogtorh_optimv2.txt", ROOT / "log_traintogtorh_optimv2.txt"),
    ("log_traintogtorh_optimv3.txt", ROOT / "log_traintogtorh_optimv3.txt"),
]

# Plot all 3 loss components together (each gets its own row -> own y-scale,
# since box/cls/dfl loss live on very different scales and can't share an axis).
METRICS = ["box_loss", "cls_loss", "dfl_loss"]

OUT_PNG = ROOT / "train_loss_togroth.png"
OUT_PDF = ROOT / "train_loss_togroth.pdf"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
TRAINER_RE = re.compile(r"^engine/trainer:.*?model=([^,]+).*?\bname=([^,]+)")
EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:"
    r"\s*100%.*?(\d+)/(\d+)\s+[\d.]+it/s"
)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LINESTYLES = {"box_loss": "-", "cls_loss": "--", "dfl_loss": ":"}


def split_by_trainer(lines: list[str]) -> list[list[str]]:
    idxs = [i for i, l in enumerate(lines) if l.startswith("engine/trainer:")]
    idxs.append(len(lines))
    return [lines[a:b] for a, b in zip(idxs, idxs[1:])]


def parse_run(block: list[str]) -> tuple[str, list[int], dict[str, list[float]]]:
    m = TRAINER_RE.match(block[0])
    model = Path(m.group(1)).stem if m else "?"
    name = m.group(2) if m else "?"
    label = f"{model} ({name})"

    losses: dict[int, list[float]] = {}
    for line in block:
        m2 = EPOCH_RE.match(line)
        if m2 and m2.group(6) == m2.group(7):  # only the 100%-complete row
            epoch = int(m2.group(1))
            losses[epoch] = [float(m2.group(3)), float(m2.group(4)), float(m2.group(5))]

    epochs = sorted(losses)
    by_metric = {
        metric: [losses[e][i] for e in epochs] for i, metric in enumerate(METRICS)
    }
    return label, epochs, by_metric


def load_file(path: Path) -> list[tuple[str, list[int], dict[str, list[float]]]]:
    with open(path, "r", errors="replace", newline=None) as f:
        lines = [ANSI_RE.sub("", line).rstrip("\n") for line in f]
    blocks = split_by_trainer(lines)
    return [parse_run(b) for b in blocks]


# ---------------------------------------------------------------------------
# Plot style (academic / publication-ready)
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
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


def main() -> None:
    all_runs = [(title, load_file(path)) for title, path in FILES]

    fig, axes = plt.subplots(1, len(FILES), figsize=(13, 4.6), sharey=True)

    for ax, (title, runs) in zip(axes, all_runs):
        for i, (label, epochs, by_metric) in enumerate(runs):
            print(f"{title} -> {label}: epoch {epochs[0]}-{epochs[-1]} ({len(epochs)} logged)")
            color = COLORS[i % len(COLORS)]
            for metric in METRICS:
                values = by_metric[metric]
                # Normalize to epoch 1 = 1.0 so box/cls/dfl loss (very
                # different absolute scales) can share one axis.
                norm = [v / values[0] for v in values]
                ax.plot(
                    epochs, norm, color=color, linestyle=LINESTYLES[metric],
                    linewidth=1.4, alpha=0.9,
                )
                if epochs[-1] < 50:
                    ax.scatter([epochs[-1]], [norm[-1]], color=color, s=20, zorder=3)
            if epochs[-1] < 50:
                ax.annotate(
                    f"{label}\nbị ngắt ở epoch {epochs[-1]}",
                    xy=(epochs[-1], by_metric["box_loss"][-1] / by_metric["box_loss"][0]),
                    xytext=(6, 10),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=color,
                )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 50)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        run_handles = [
            Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=1.8, label=label)
            for i, (label, _, _) in enumerate(runs)
        ]
        ax.legend(handles=run_handles, frameon=False, loc="upper right", fontsize=7.2)

    axes[0].set_ylabel("Loss / loss(epoch 1)")

    metric_handles = [
        Line2D([0], [0], color="0.25", linestyle=LINESTYLES[m], linewidth=1.6, label=m)
        for m in METRICS
    ]
    fig.legend(
        handles=metric_handles, frameon=False, ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Training loss (chuẩn hóa theo epoch 1) — COCO pretrain sweep (togroth)", y=1.03
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
