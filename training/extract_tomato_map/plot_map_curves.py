"""Plot per-epoch validation mAP curves from Ultralytics training logs.

Reads log_finetuning_v1_3_20_8.txt and log_finetuning_v2_3_20_8.txt (each file
contains several concatenated training runs, one per "TRAIN <name>" block),
extracts the per-epoch validation mAP for every run, and draws one academic-
style figure with two panels: (a) v1 runs, (b) v2 runs.

Usage:
    python plot_map_curves.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).resolve().parent
LOG_FILES = {
    "v1": LOG_DIR / "log_finetuning_v1_3_20_8.txt",
    "v2": LOG_DIR / "log_finetuning_v2_3_20_8.txt",
}
OUT_PNG = LOG_DIR / "map_curves_v1_v2.png"
OUT_PDF = LOG_DIR / "map_curves_v1_v2.pdf"

# Which metric to plot: "mAP50_95" (mAP@0.5:0.95, COCO primary) or "mAP50".
METRIC = "mAP50_95"

# Only plot these specific epochs (set to None to plot all 25 epochs).
EPOCHS_TO_PLOT = [1, 5, 10, 16]

# Smooth the curve with a cubic B-spline interpolated through the real
# per-epoch points (the points themselves are unchanged, only the line
# between them is smoothed for readability).
SMOOTH = False
SMOOTH_POINTS = 200
METRIC_LABEL = {"mAP50_95": r"mAP@0.5:0.95", "mAP50": r"mAP@0.5"}[METRIC]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:"
    r"\s*100%.*?(\d+)/(\d+)\s+[\d.]+it/s"
)
ALL_ROW_RE = re.compile(
    r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def load_lines(path: Path) -> list[str]:
    # Python's text-mode universal newlines already turns the '\r' used by
    # Ultralytics' progress bar into line breaks, so no manual tr needed.
    with open(path, "r", errors="replace", newline=None) as f:
        return [ANSI_RE.sub("", line).rstrip("\n") for line in f]


def split_runs(lines: list[str]) -> list[list[str]]:
    idxs = [i for i, l in enumerate(lines) if l.startswith("TRAIN ")]
    idxs.append(len(lines))
    return [lines[a:b] for a, b in zip(idxs, idxs[1:])]


def parse_run(block: list[str]) -> tuple[str, list[int], list[float]]:
    name = block[0].replace("TRAIN ", "").strip()

    epoch_nums: list[int] = []
    for line in block:
        m = EPOCH_RE.match(line)
        if m and m.group(3) == m.group(4):  # only the 100%-complete row
            epoch_nums.append(int(m.group(1)))

    all_rows = []
    for line in block:
        m = ALL_ROW_RE.match(line)
        if m:
            p, r, map50, map50_95 = (float(x) for x in m.groups()[2:])
            all_rows.append({"mAP50": map50, "mAP50_95": map50_95})

    # One "all" row per epoch-end validation, plus one extra final
    # re-validation of best.pt at the end of the block -> drop it.
    values = [row[METRIC] for row in all_rows[: len(epoch_nums)]]
    return name, epoch_nums, values


def load_runs(path: Path) -> list[tuple[str, list[int], list[float]]]:
    blocks = split_runs(load_lines(path))
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
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
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

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
MARKERS = ["o", "s", "^", "D"]

PANEL_TITLE = {
    "v1": "(a) So sánh module attention",
    "v2": "(b) So sánh thu hẹp độ rộng mạng",
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, key in zip(axes, ["v1", "v2"]):
        runs = load_runs(LOG_FILES[key])
        for i, (name, epochs, values) in enumerate(runs):
            if EPOCHS_TO_PLOT is not None:
                wanted = set(EPOCHS_TO_PLOT)
                epochs, values = zip(
                    *[(e, v) for e, v in zip(epochs, values) if e in wanted]
                )
            color = COLORS[i % len(COLORS)]

            if SMOOTH and len(epochs) > 3:
                x_smooth = np.linspace(epochs[0], epochs[-1], SMOOTH_POINTS)
                spline = make_interp_spline(epochs, values, k=3)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, color=color, linewidth=1.6, label=name)
                ax.plot(
                    epochs,
                    values,
                    linestyle="none",
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=4,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                )
            else:
                ax.plot(
                    epochs,
                    values,
                    label=name,
                    color=color,
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=3.5,
                    linewidth=1.4,
                )
        ax.set_title(PANEL_TITLE[key])
        ax.set_xlabel("Epoch")
        if EPOCHS_TO_PLOT is not None:
            ax.set_xlim(EPOCHS_TO_PLOT[0], EPOCHS_TO_PLOT[-1])
            ax.set_xticks(EPOCHS_TO_PLOT)
        else:
            ax.set_xlim(1, 25)
            ax.set_xticks(range(1, 26, 4))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right")

    axes[0].set_ylabel(METRIC_LABEL)

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
