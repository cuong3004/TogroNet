"""Ve duong cong train SSL cua TogroNet (Barlow Twins), gop 3 lan chay ke tiep
nhau (version_31 -> version_33 -> version_34, cung 1 lan train bi ngat quang
va resume tu checkpoint) thanh 1 duong lien tuc theo global step.

2 subplot canh nhau: train_loss (trai) va train_pos_sim (phai). Khong title,
lam min bang EMA (cung phong cach voi plot_ssl_stability.py trong repo).

Usage:
    python plot_ssl_train_togronet.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
LOG_DIR = Path("/home/agi/thesis_code/ssl_project/lightning_logs")
VERSIONS = ["version_31", "version_33", "version_34"]

OUT_PNG = ROOT / "ssl_train_togronet.png"
OUT_PDF = ROOT / "ssl_train_togronet.pdf"

# ---------------------------------------------------------------------------
# Plot style (academic / publication-ready) -- dong bo voi plot_ssl_stability.py
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

LINE_COLOR = "#2a78d6"
SMOOTH_WEIGHT = 0.9


def ema(values: list[float], weight: float = SMOOTH_WEIGHT) -> list[float]:
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def load_scalar(tag: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for v in VERSIONS:
        ea = EventAccumulator(str(LOG_DIR / v))
        ea.Reload()
        for e in ea.Scalars(tag):
            steps.append(e.step)
            values.append(e.value)
    # 3 lan chay noi tiep co the chong step nhe o diem resume -> sap xep lai
    # theo step de duong ve luon don dieu tang, khong bi "giat lui".
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    values = [values[i] for i in order]
    return steps, values


def steps_to_epochs(query_steps: list[int]) -> list[float]:
    """Anh xa step -> epoch bang scalar 'epoch' cua chinh Lightning log (lien
    tuc xuyen suot 3 version: 0 -> 18 -> 317 -> 499), tra ve epoch ganh gan
    nhat truoc do cho moi step truy van."""
    epoch_steps, epoch_vals = load_scalar("epoch")
    import bisect

    out = []
    for s in query_steps:
        i = bisect.bisect_right(epoch_steps, s) - 1
        i = max(i, 0)
        out.append(epoch_vals[i])
    return out


def main() -> None:
    loss_steps, loss_vals = load_scalar("train_loss")
    sim_steps, sim_vals = load_scalar("train_pos_sim")

    loss_epochs = steps_to_epochs(loss_steps)
    sim_epochs = steps_to_epochs(sim_steps)

    print(f"train_loss: {len(loss_vals)} diem, epoch {loss_epochs[0]:.0f}-{loss_epochs[-1]:.0f}")
    print(f"train_pos_sim: {len(sim_vals)} diem, epoch {sim_epochs[0]:.0f}-{sim_epochs[-1]:.0f}")

    fig, (ax_loss, ax_sim) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(loss_epochs, ema(loss_vals), color=LINE_COLOR, linewidth=1.8, zorder=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")

    ax_sim.plot(sim_epochs, ema(sim_vals), color=LINE_COLOR, linewidth=1.8, zorder=2)
    ax_sim.set_xlabel("Epoch")
    ax_sim.set_ylabel("Train pos. similarity")

    for ax in (ax_loss, ax_sim):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 500)

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
