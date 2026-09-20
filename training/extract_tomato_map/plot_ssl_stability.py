"""Visualize how rough / interrupted the SSL pretraining runs were.

lambda_coeff=0.005 was trained across 3 separate log files (the machine got
interrupted and the run was resumed from checkpoint each time), plotted
alongside a separate, continuous reference run (log_ssl_yolo.txt, v_num=1,
no restarts). Instead of loss (noisy, hard to read across restarts), this
plots two SSL-health metrics that are far more diagnostic of restart damage:

  - val_pos_sim  : cosine similarity between augmented views of the same
                   image on the val set. Should climb steadily; a stall or
                   dip means the restart hurt optimization.
  - val_emb_var  : variance of the embeddings (collapse indicator). A
                   healthy SSL run keeps this in a stable band; jumps at a
                   restart boundary mean the resumed state was not clean.

Reading the 3 lambda-sweep log files' epoch ranges also exposes two very
literal signs of "interrupted / re-run" training:
  - an OVERLAP region where a later attempt resumed from a checkpoint that
    lagged behind the previous attempt's progress -> the same epochs get
    trained twice.
  - a GAP region covered by none of the 3 logs -> epochs whose training
    happened but was never captured in any log we have.

Set SPLIT below to "val" or "train" to switch which split's pos_sim /
emb_var is plotted.

Usage:
    python plot_ssl_stability.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "logs" / "ssl_lambda_sweep"

# (label, path, color) - the first 3 are one interrupted run (lambda=0.005,
# chronological attempt order by v_num); the 4th is an unrelated, continuous
# reference run.
RUNS_CFG = [
    ("Lần 1 (λ=0.005)", SWEEP_DIR / "ssl_pretrainlast.pt_lambda0.005 copy.log", "#2a78d6"),
    ("Lần 2 (λ=0.005)", SWEEP_DIR / "ssl_pretrainlast.pt_lambda0.005 copy 2.log", "#eb6834"),
    ("Lần 3 (λ=0.005, hoàn tất)", SWEEP_DIR / "ssl_pretrainlast.pt_lambda0.005.log", "#1baf7a"),
    ("log_ssl_yolo (liên tục, không ngắt)", ROOT / "log_ssl_yolo.txt", "#eda100"),
]

# Which split to plot: "val" (validation-set SSL health) or "train"
# (training-set SSL health, read at each epoch's 100% completion line).
SPLIT = "train"
SIM_KEY = f"{SPLIT}_pos_sim"
VAR_KEY = f"{SPLIT}_emb_var"

OUT_PNG = ROOT / f"ssl_stability_lambda0.005_{SPLIT}.png"
OUT_PDF = ROOT / f"ssl_stability_lambda0.005_{SPLIT}.pdf"

EPOCH_START_RE = re.compile(r"^Epoch (\d+): 100%\|")
KV_RE = re.compile(r"(\w+)=([-+]?[\d.]+(?:e[-+]?\d+)?)")


def parse_attempt(path: Path) -> list[dict]:
    # Python's text-mode universal newlines turns the tqdm '\r' updates
    # into line breaks already, so no manual preprocessing is needed.
    # Field order in the postfix varies between logs, so parse key=value
    # pairs generically instead of assuming a fixed order.
    best: dict[int, dict] = {}
    with open(path, "r", errors="replace", newline=None) as f:
        for line in f:
            m = EPOCH_START_RE.match(line)
            if not m:
                continue
            kv = dict(KV_RE.findall(line))
            if SIM_KEY not in kv or VAR_KEY not in kv:
                continue
            epoch = int(m.group(1))
            best[epoch] = {
                "epoch": epoch,
                "sim": float(kv[SIM_KEY]),
                "var": float(kv[VAR_KEY]),
            }
    return [best[e] for e in sorted(best)]


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

GAP_COLOR = "#c23b2e"

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
    runs = [(label, parse_attempt(path), color) for label, path, color in RUNS_CFG]

    for label, rows, _ in runs:
        if rows:
            print(f"{label}: epoch {rows[0]['epoch']}-{rows[-1]['epoch']} ({len(rows)} epochs logged)")

    # Detect the overlap (re-trained epochs) between lambda-sweep attempt 1
    # and 2, and the gap between attempt 2 and attempt 3. The 4th run
    # (log_ssl_yolo) is a separate, continuous run and is excluded here.
    e1 = {r["epoch"] for r in runs[0][1]}
    e2 = {r["epoch"] for r in runs[1][1]}
    overlap = sorted(e1 & e2)
    gap_start = runs[1][1][-1]["epoch"] + 1
    gap_end = runs[2][1][0]["epoch"] - 1
    print(f"Overlap (re-trained twice): epoch {overlap[0]}-{overlap[-1]} ({len(overlap)} epochs)")
    print(f"Gap (not logged anywhere): epoch {gap_start}-{gap_end} ({gap_end - gap_start + 1} epochs)")

    fig, (ax_sim, ax_var) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]}
    )

    for label, rows, color in runs:
        epochs = [r["epoch"] for r in rows]
        sim_raw = [r["sim"] for r in rows]
        var_raw = [r["var"] for r in rows]

        # Faint raw curve (keeps the real noise visible) + bold EMA on top.
        ax_sim.plot(epochs, sim_raw, color=color, linewidth=0.7, alpha=0.3, zorder=1)
        ax_sim.plot(epochs, ema(sim_raw), color=color, linewidth=1.8, label=label, zorder=2)
        ax_var.plot(epochs, var_raw, color=color, linewidth=0.7, alpha=0.3, zorder=1)
        ax_var.plot(epochs, ema(var_raw), color=color, linewidth=1.8, label=label, zorder=2)

    for ax in (ax_sim, ax_var):
        ax.axvspan(overlap[0], overlap[-1], color="0.5", alpha=0.12, zorder=0)
        ax.axvspan(gap_start, gap_end, color=GAP_COLOR, alpha=0.12, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_sim.annotate(
        f"Train lại {len(overlap)} epoch\n(checkpoint bị trễ)",
        xy=((overlap[0] + overlap[-1]) / 2, ax_sim.get_ylim()[1]),
        xytext=(0, -4),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.5,
        color="0.35",
    )
    ax_sim.annotate(
        f"Ngắt {gap_end - gap_start + 1} epoch\n(không có log)",
        xy=((gap_start + gap_end) / 2, ax_sim.get_ylim()[1]),
        xytext=(0, -4),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.5,
        color=GAP_COLOR,
    )

    ax_sim.set_ylabel(SIM_KEY)
    split_label = "tập train" if SPLIT == "train" else "tập validation"
    ax_sim.set_title(
        f"Độ ổn định huấn luyện SSL ({split_label}): chạy bị ngắt quãng ($\\lambda$=0.005) so với chạy liên tục"
    )
    ax_sim.legend(frameon=False, loc="lower right", fontsize=8)

    ax_var.set_ylabel(VAR_KEY)
    ax_var.set_xlabel("Epoch")
    ax_var.set_xlim(0, max(r["epoch"] for _, rows, _ in runs for r in rows))

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
