"""Gop confusion matrix cua TogroNet-Fruit (cm_cache) va TogroNet-Stage
(cm_cache_stage) vao chung 1 luoi duy nhat, thay vi 2 hinh rieng
(plot_confusion_matrices.py va plot_cm_stage.py van giu nguyen, khong doi).

Usage:
    python plot_confusion_matrices_combined.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FRUIT_CACHE_DIR = ROOT / "cm_cache"
STAGE_CACHE_DIR = ROOT / "cm_cache_stage"
OUT_PNG = ROOT / "confusion_matrices_combined.png"
OUT_PDF = ROOT / "confusion_matrices_combined.pdf"

FRUIT_SHORT_NAMES = {
    "green": "green",
    "half_ripened": "half",
    "fully_ripened": "full",
    "background": "bg",
}

FRUIT_LABEL_MAP = {
    "togrow0_coco_finetune": "TogroNet-Fruit-V0",
    "togrow1_coco_finetune": "TogroNet-Fruit-V1",
    "togrow2_coco_finetune": "TogroNet-Fruit-V2",
    "togrow3_coco_finetune": "TogroNet-Fruit-V3",
    "togrow4_coco_finetune": "TogroNet-Fruit-V4",
    "togrow5_coco_finetune": "TogroNet-Fruit-V5",
    "togrow6_coco_finetune": "TogroNet-Fruit-V6",
}
FRUIT_ORDER = list(FRUIT_LABEL_MAP.values())

STAGE_SHORT_NAMES = {
    "flowering": "flowering",
    "fruit_development": "fruit_dev",
    "fruit_ripening": "fruit_rip",
    "vegetative_growth": "veg_growth",
}
GROWTH_ORDER = ["vegetative_growth", "flowering", "fruit_development", "fruit_ripening"]
STAGE_LABEL_MAP = {
    "TogroNet-Stage": "TogroNet-Stage",
    "Mô hình đề xuất": "TogroNet-Stage\nSSL-BT",
}
STAGE_ORDER = list(STAGE_LABEL_MAP.values())

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_fruit():
    entries = []
    for f in sorted(FRUIT_CACHE_DIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        run_dir = str(d["run_dir"])
        if run_dir not in FRUIT_LABEL_MAP:
            continue
        entries.append(dict(
            matrix=d["matrix"],
            names=[FRUIT_SHORT_NAMES.get(n, n) for n in d["names"]],
            label=FRUIT_LABEL_MAP[run_dir],
        ))
    entries.sort(key=lambda e: FRUIT_ORDER.index(e["label"]))
    return entries


def load_stage():
    entries = []
    for f in sorted(STAGE_CACHE_DIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        raw_names = list(d["names"])
        k = len(raw_names)
        matrix = d["matrix"][:k, :k]
        perm = [raw_names.index(n) for n in GROWTH_ORDER]
        matrix = matrix[np.ix_(perm, perm)]
        names = [STAGE_SHORT_NAMES.get(n, n) for n in GROWTH_ORDER]

        raw_label = str(d["label"])
        entries.append(dict(
            matrix=matrix,
            names=names,
            label=STAGE_LABEL_MAP.get(raw_label, raw_label),
        ))
    entries.sort(key=lambda e: STAGE_ORDER.index(e["label"]))
    return entries


def draw_cm(ax, entry):
    mat = entry["matrix"].astype(float)
    col_sum = mat.sum(0, keepdims=True)
    norm = np.divide(mat, col_sum, out=np.zeros_like(mat), where=col_sum > 0)

    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    k = len(entry["names"])
    for i in range(k):
        for j in range(k):
            v = norm[i, j]
            ax.text(
                j, i, f"{v:.2f}" if v > 0 else "",
                ha="center", va="center",
                fontsize=8,
                color="white" if v > 0.6 else "black",
            )

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(entry["names"], rotation=45, ha="right")
    ax.set_yticklabels(entry["names"])
    ax.set_title(entry["label"], fontsize=10.5)
    ax.tick_params(length=0)


def main():
    entries = load_fruit() + load_stage()
    n = len(entries)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.95 * ncols, 2.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, e in zip(axes, entries):
        draw_cm(ax, e)

    for ax in axes[n:]:
        ax.axis("off")

    fig.text(0.5, -0.01, "Nhãn thật", ha="center", fontsize=11.5)
    fig.text(-0.01, 0.5, "Nhãn dự đoán", va="center", rotation=90, fontsize=11.5)

    fig.tight_layout(rect=[0.01, 0.01, 1, 1], pad=0.4, h_pad=0.3, w_pad=0.3)

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG} ({n} panel, {nrows}x{ncols})")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
