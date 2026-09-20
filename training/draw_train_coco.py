
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


# ============================================================
# CẤU HÌNH DỮ LIỆU
# ============================================================

RUNS_DIR = Path("runs/detect")

RUNS = {
    "YOLO26": "train-3",
    "ToGrowth-V1": "train-5",
    "ToGrowth-V2": "train-6",
    "ToGrowth-V3": "train-7",
    "ToGrowth-V4": "train-8",
    "ToGrowth-V5": "train-10",
    "ToGrowth-V6": "train-11",
}

OUTPUT_PDF = "training_loss_coco_pretrained.pdf"
OUTPUT_PNG = "training_loss_coco_pretrained.png"

LOSS_COLUMNS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
]


# ============================================================
# ĐỊNH DẠNG ĐỒ THỊ HỌC THUẬT
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.7,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


COLORS = {
    "YOLO26": "#1F1F1F",
    "ToGrowth-V1": "#0072B2",
    "ToGrowth-V2": "#D55E00",
    "ToGrowth-V3": "#009E73",
    "ToGrowth-V4": "#CC79A7",
    "ToGrowth-V5": "#E69F00",
    "ToGrowth-V6": "#6A3D9A",
}


LINESTYLES = {
    "YOLO26": "-",
    "ToGrowth-V1": "--",
    "ToGrowth-V2": "-.",
    "ToGrowth-V3": ":",
    "ToGrowth-V4": (0, (5, 1)),
    "ToGrowth-V5": (0, (3, 1, 1, 1)),
    "ToGrowth-V6": (0, (7, 2)),
}


MARKERS = {
    "YOLO26": None,
    "ToGrowth-V1": "o",
    "ToGrowth-V2": "s",
    "ToGrowth-V3": "^",
    "ToGrowth-V4": "D",
    "ToGrowth-V5": "v",
    "ToGrowth-V6": "P",
}


# ============================================================
# HÀM ĐỌC FILE KẾT QUẢ
# ============================================================

def find_results_file(run_dir: Path) -> Path:
    """
    Tìm file results.csv hoặc result.csv trong thư mục run.
    """
    candidates = [
        run_dir / "results.csv",
        run_dir / "result.csv",
    ]

    for file_path in candidates:
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        f"Không tìm thấy results.csv hoặc result.csv trong: {run_dir}"
    )


def load_results(run_name: str) -> pd.DataFrame:
    """
    Đọc dữ liệu training loss và tính tổng loss tại mỗi epoch.
    """
    run_dir = RUNS_DIR / run_name
    csv_path = find_results_file(run_dir)

    df = pd.read_csv(csv_path)

    # Loại bỏ khoảng trắng thừa trong tên cột
    df.columns = df.columns.str.strip()

    required_columns = ["epoch", *LOSS_COLUMNS]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"File {csv_path} thiếu các cột: {missing_columns}"
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns
    ).sort_values("epoch")

    # Ultralytics thường lưu epoch bắt đầu từ 0
    df["epoch"] = df["epoch"].astype(int) + 1

    # Tổng training loss
    df["train/total_loss"] = (
        df["train/box_loss"]
        + df["train/cls_loss"]
        + df["train/dfl_loss"]
    )

    return df


# ============================================================
# VẼ ĐỒ THỊ
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.6),
    constrained_layout=True,
)

loaded_results = {}

for model_label, run_name in RUNS.items():
    try:
        results = load_results(run_name)
        loaded_results[model_label] = results

        marker_interval = max(
            len(results) // 12,
            1,
        )

        ax.plot(
            results["epoch"],
            results["train/total_loss"],
            label=model_label,
            color=COLORS[model_label],
            linestyle=LINESTYLES[model_label],
            linewidth=1.7,
            marker=MARKERS[model_label],
            markersize=3.4,
            markevery=marker_interval,
            markerfacecolor="white",
            markeredgewidth=0.8,
            alpha=0.96,
            zorder=3,
        )

        print(
            f"{model_label:14s} | "
            f"run={run_name:8s} | "
            f"final total loss="
            f"{results['train/total_loss'].iloc[-1]:.4f} | "
            f"minimum total loss="
            f"{results['train/total_loss'].min():.4f}"
        )

    except (FileNotFoundError, ValueError) as error:
        print(f"[Bỏ qua] {model_label}: {error}")


if not loaded_results:
    raise RuntimeError(
        "Không đọc được dữ liệu từ bất kỳ thư mục huấn luyện nào."
    )


# ============================================================
# ĐỊNH DẠNG TRỤC
# ============================================================

ax.set_xlabel("Epoch")
ax.set_ylabel("Total training loss")

max_epoch = max(
    df["epoch"].max()
    for df in loaded_results.values()
)

ax.set_xlim(
    left=1,
    right=max_epoch,
)

all_loss_values = pd.concat(
    [
        df["train/total_loss"]
        for df in loaded_results.values()
    ],
    ignore_index=True,
)

loss_min = all_loss_values.min()
loss_max = all_loss_values.max()
loss_range = loss_max - loss_min

ax.set_ylim(
    bottom=max(0, loss_min - 0.05 * loss_range),
    top=loss_max + 0.05 * loss_range,
)


if max_epoch <= 50:
    epoch_step = 5
elif max_epoch <= 150:
    epoch_step = 10
elif max_epoch <= 300:
    epoch_step = 25
else:
    epoch_step = 50

ax.xaxis.set_major_locator(
    MultipleLocator(epoch_step)
)


# ============================================================
# LƯỚI VÀ KHUNG ĐỒ THỊ
# ============================================================

ax.grid(
    True,
    which="major",
    axis="both",
    linestyle="--",
    linewidth=0.55,
    color="#B8B8B8",
    alpha=0.65,
    zorder=0,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#4A4A4A")
ax.spines["bottom"].set_color("#4A4A4A")

ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=4,
    width=0.8,
    colors="#2F2F2F",
)


# ============================================================
# CHÚ GIẢI
# ============================================================

legend = ax.legend(
    loc="upper right",
    ncol=2,
    frameon=True,
    fancybox=False,
    framealpha=0.95,
    edgecolor="#808080",
    borderpad=0.6,
    labelspacing=0.45,
    columnspacing=1.2,
    handlelength=2.7,
)

legend.get_frame().set_linewidth(0.6)
legend.get_frame().set_facecolor("white")


# ============================================================
# LƯU HÌNH
# ============================================================

fig.savefig(
    OUTPUT_PDF,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    OUTPUT_PNG,
    format="png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()

print(f"\nĐã lưu file vector: {OUTPUT_PDF}")
print(f"Đã lưu file raster: {OUTPUT_PNG}")
