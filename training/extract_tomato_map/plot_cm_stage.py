"""Ve confusion matrix (chuan hoa theo cot) cho 2 model classify TogroNet-Stage,
cung phong cach voi plot_confusion_matrices.py (khong title phu, khong colorbar)."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cm_cache_stage"
OUT_PNG = ROOT / "confusion_matrices_togronet_stage.png"
OUT_PDF = ROOT / "confusion_matrices_togronet_stage.pdf"

SHORT_NAMES = {
    "flowering": "flowering",
    "fruit_development": "fruit_dev",
    "fruit_ripening": "fruit_rip",
    "vegetative_growth": "veg_growth",
}

# Thu tu chu ky sinh truong that (khong phai alphabet):
# sinh duong -> ra hoa -> hinh thanh qua -> qua chuyen chin
GROWTH_ORDER = ["vegetative_growth", "flowering", "fruit_development", "fruit_ripening"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

LABEL_ORDER = ["TogroNet-Stage", "Mô hình đề xuất"]


def main():
    entries = []
    for f in sorted(CACHE_DIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        raw_names = list(d["names"])
        k = len(raw_names)
        matrix = d["matrix"][:k, :k]  # cat bo hang/cot "ma" (5x5 -> 4x4)

        # sap xep lai theo GROWTH_ORDER (ca hang va cot, giu tinh nhat quan pred/true)
        perm = [raw_names.index(n) for n in GROWTH_ORDER]
        matrix = matrix[np.ix_(perm, perm)]
        names = [GROWTH_ORDER[i] for i in range(k)]

        entries.append(dict(
            matrix=matrix,
            names=[SHORT_NAMES.get(n, n) for n in names],
            label=str(d["label"]),
        ))
    entries.sort(key=lambda e: LABEL_ORDER.index(e["label"]))

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4))

    for ax, e in zip(axes, entries):
        mat = e["matrix"].astype(float)
        col_sum = mat.sum(0, keepdims=True)
        norm = np.divide(mat, col_sum, out=np.zeros_like(mat), where=col_sum > 0)

        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

        k = len(e["names"])
        for i in range(k):
            for j in range(k):
                v = norm[i, j]
                ax.text(
                    j, i, f"{v:.2f}" if v > 0 else "",
                    ha="center", va="center",
                    fontsize=6,
                    color="white" if v > 0.6 else "black",
                )

        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(e["names"], rotation=45, ha="right")
        ax.set_yticklabels(e["names"])
        ax.set_title(e["label"], fontsize=8)
        ax.tick_params(length=0)

    fig.text(0.5, -0.03, "Nhãn thật", ha="center", fontsize=7)
    fig.text(-0.01, 0.5, "Nhãn dự đoán", va="center", rotation=90, fontsize=7)

    fig.tight_layout(rect=[0.01, 0.01, 1, 1])
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
