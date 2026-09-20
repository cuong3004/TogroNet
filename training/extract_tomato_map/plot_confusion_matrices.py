"""Ve luoi confusion matrix (chuan hoa theo cot, giong quy uoc mac dinh cua
ultralytics) cho toan bo cac phien ban model detect tren laboro_tomato_yolo,
gop trong 1 hinh duy nhat -- doc du lieu da cache tu compute_confusion_matrices.py.

Usage:
    python plot_confusion_matrices.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cm_cache"
OUT_PNG = ROOT / "confusion_matrices_togronet.png"
OUT_PDF = ROOT / "confusion_matrices_togronet.pdf"

SHORT_NAMES = {
    "green": "green",
    "half_ripened": "half",
    "fully_ripened": "full",
    "background": "bg",
}

# run_dir -> nhan hien thi cuoi cung. Run nao khong co trong map nay se bi
# loai khoi hinh (yolo26n_coco_finetune, togrow0005 SSL toan phan).
LABEL_MAP = {
    "togrow0_coco_finetune": "TogroNet-Fruit-V0",
    "togrow1_coco_finetune": "TogroNet-Fruit-V1",
    "togrow2_coco_finetune": "TogroNet-Fruit-V2",
    "togrow3_coco_finetune": "TogroNet-Fruit-V3",
    "togrow4_coco_finetune": "TogroNet-Fruit-V4",
    "togrow5_coco_finetune": "TogroNet-Fruit-V5",
    "togrow6_coco_finetune": "TogroNet-Fruit-V6",
    "togrow0005_backbone": "Mô hình đề xuất",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_all():
    files = sorted(CACHE_DIR.glob("*.npz"))
    entries = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        run_dir = str(d["run_dir"])
        if run_dir not in LABEL_MAP:
            continue
        entries.append(
            dict(
                matrix=d["matrix"],
                names=[SHORT_NAMES.get(n, n) for n in d["names"]],
                label=LABEL_MAP[run_dir],
                map=float(d["map50_95"]),
            )
        )
    order = list(LABEL_MAP.values())
    entries.sort(key=lambda e: order.index(e["label"]))
    return entries


def main():
    entries = load_all()
    n = len(entries)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.35 * nrows))
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, e in zip(axes, entries):
        mat = e["matrix"].astype(float)
        col_sum = mat.sum(0, keepdims=True)
        norm = np.divide(mat, col_sum, out=np.zeros_like(mat), where=col_sum > 0)

        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

        k = len(e["names"])
        for i in range(k):
            for j in range(k):
                v = norm[i, j]
                ax.text(
                    j, i, f"{v:.2f}" if v > 0 else "",
                    ha="center", va="center",
                    fontsize=6.5,
                    color="white" if v > 0.6 else "black",
                )

        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(e["names"], rotation=45, ha="right")
        ax.set_yticklabels(e["names"])
        ax.set_title(e["label"], fontsize=8.5)
        ax.tick_params(length=0)

    for ax in axes[n:]:
        ax.axis("off")

    fig.text(0.5, -0.02, "Nhãn thật", ha="center", fontsize=10)
    fig.text(-0.01, 0.5, "Nhãn dự đoán", va="center", rotation=90, fontsize=10)

    fig.tight_layout(rect=[0.01, 0.01, 1, 1])

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
