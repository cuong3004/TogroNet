from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


# ============================================================
# CẤU HÌNH DỮ LIỆU
# ============================================================

RUNS = {
    r"$\lambda = 0.001$": Path("version_9.csv"),
    r"$\lambda = 0.005$": Path("version_10.csv"),
    r"$\lambda = 0.01$": Path("version_11.csv"),
}

OUTPUT_PDF = "ssl_validation_positive_similarity.pdf"
OUTPUT_PNG = "ssl_validation_positive_similarity.png"

STEP_COLUMN = "Step"
METRIC_COLUMN = "Value"

# Làm mượt bằng trung bình trượt.
# Có thể tăng lên 15 hoặc 21 nếu đường vẫn còn nhiễu.
SMOOTH_WINDOW = 5

# Giới hạn trục tung cố định để tránh phóng đại khác biệt.
Y_MIN = 0.55
Y_MAX = 0.75
Y_MAJOR_STEP = 0.05

# Khoảng chia trục step.
X_MAJOR_STEP = 50_000


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
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


COLORS = {
    r"$\lambda = 0.001$": "#0072B2",
    r"$\lambda = 0.005$": "#D55E00",
    r"$\lambda = 0.01$": "#009E73",
}

LINESTYLES = {
    r"$\lambda = 0.001$": "-",
    r"$\lambda = 0.005$": "--",
    r"$\lambda = 0.01$": "-.",
}


# ============================================================
# ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ============================================================

def load_ssl_results(csv_path: Path) -> pd.DataFrame:
    """
    Đọc dữ liệu SSL từ CSV và tạo đường làm mượt.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Loại bỏ khoảng trắng trong tên cột.
    df.columns = df.columns.str.strip()

    required_columns = [
        STEP_COLUMN,
        METRIC_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"File {csv_path} thiếu các cột: {missing_columns}\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    df[STEP_COLUMN] = pd.to_numeric(
        df[STEP_COLUMN],
        errors="coerce",
    )

    df[METRIC_COLUMN] = pd.to_numeric(
        df[METRIC_COLUMN],
        errors="coerce",
    )

    df = (
        df.dropna(subset=required_columns)
        .sort_values(STEP_COLUMN)
    )

    # Gộp các step trùng nhau nếu có.
    df = (
        df.groupby(STEP_COLUMN, as_index=False)[METRIC_COLUMN]
        .mean()
    )

    if df.empty:
        raise ValueError(
            f"File {csv_path} không có dữ liệu hợp lệ."
        )

    # Làm mượt bằng centered rolling mean.
    df["smoothed_metric"] = (
        df[METRIC_COLUMN]
        .rolling(
            window=max(SMOOTH_WINDOW, 1),
            min_periods=1,
            center=True,
        )
        .mean()
    )

    return df


# ============================================================
# KHỞI TẠO HÌNH
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.0),
    constrained_layout=True,
)

loaded_results = {}


# ============================================================
# VẼ DỮ LIỆU
# ============================================================

for run_label, csv_path in RUNS.items():
    try:
        results = load_ssl_results(csv_path)
        loaded_results[run_label] = results

        # Dữ liệu gốc: vẽ mờ để thể hiện độ dao động.
        ax.plot(
            results[STEP_COLUMN],
            results[METRIC_COLUMN],
            color=COLORS[run_label],
            linestyle="-",
            linewidth=0.7,
            alpha=0.18,
            zorder=1,
        )

        # Đường làm mượt: dùng làm đường so sánh chính.
        ax.plot(
            results[STEP_COLUMN],
            results["smoothed_metric"],
            label=run_label,
            color=COLORS[run_label],
            linestyle=LINESTYLES[run_label],
            linewidth=2.0,
            alpha=1.0,
            zorder=3,
        )

        print(
            f"{run_label:20s} | "
            f"file={str(csv_path):30s} | "
            f"final={results[METRIC_COLUMN].iloc[-1]:.4f} | "
            f"mean={results[METRIC_COLUMN].mean():.4f} | "
            f"maximum={results[METRIC_COLUMN].max():.4f}"
        )

    except (FileNotFoundError, ValueError) as error:
        print(f"[Bỏ qua] {run_label}: {error}")


if not loaded_results:
    raise RuntimeError(
        "Không đọc được dữ liệu hợp lệ từ bất kỳ file CSV nào."
    )


# ============================================================
# ĐỊNH DẠNG TRỤC HOÀNH
# ============================================================

max_step = max(
    results[STEP_COLUMN].max()
    for results in loaded_results.values()
)

# Làm tròn giới hạn trên theo bội số của X_MAJOR_STEP.
x_upper = (
    math.ceil(max_step / X_MAJOR_STEP)
    * X_MAJOR_STEP
)

ax.set_xlim(
    left=0,
    right=x_upper,
)

ax.xaxis.set_major_locator(
    MultipleLocator(X_MAJOR_STEP)
)

# Hiển thị step theo dạng 0, 50, 100, ... ×10^3.
ax.ticklabel_format(
    axis="x",
    style="sci",
    scilimits=(3, 3),
    useMathText=True,
)

ax.xaxis.get_offset_text().set_fontsize(9.5)


# ============================================================
# ĐỊNH DẠNG TRỤC TUNG
# ============================================================

ax.set_ylim(
    bottom=Y_MIN,
    top=Y_MAX,
)

ax.yaxis.set_major_locator(
    MultipleLocator(Y_MAJOR_STEP)
)

ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.2f")
)

ax.set_xlabel("Training step")
ax.set_ylabel("Validation positive-pair similarity")


# ============================================================
# LƯỚI VÀ KHUNG
# ============================================================

ax.grid(
    True,
    which="major",
    axis="both",
    linestyle="--",
    linewidth=0.55,
    color="#B8B8B8",
    alpha=0.55,
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
    loc="lower right",
    ncol=1,
    frameon=True,
    fancybox=False,
    framealpha=0.96,
    edgecolor="#808080",
    borderpad=0.65,
    labelspacing=0.55,
    handlelength=3.2,
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
    pad_inches=0.04,
)

fig.savefig(
    OUTPUT_PNG,
    format="png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.show()

print(f"\nĐã lưu file vector: {OUTPUT_PDF}")
print(f"Đã lưu file raster: {OUTPUT_PNG}")