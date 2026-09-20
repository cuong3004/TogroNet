"""Ve bang cau hinh (STT, ten khoi, so kenh CNN ghi trong file .yaml goc) cho
8 phien ban model TogroNet, moi phien ban la mot cot. Du lieu duoc doc thu
cong tu cac file cau hinh that:
    togrow0.yaml, togrow1.yaml, togrow2.yaml, togrow3.yaml
    yolo26_versions/togroth_m2_width_0225.yaml  (= TogroNet-Fruit-V4)
    yolo26_versions/togroth_m3_width_0200.yaml  (= TogroNet-Fruit-V5)
    yolo26_versions/togroth_m4_width_0175.yaml  (= TogroNet-Fruit-V6)
"Mo hinh de xuat" (togrow0005_backbone) dung CHUNG file cau hinh voi V4
(togroth_m2_width_0225.yaml) -- khac nhau duy nhat o trong so khoi tao
(pretrain SSL backbone thay vi pretrain COCO), khong khac ve kien truc/so kenh.

Usage:
    python plot_togronet_config_table.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "togronet_versions_config.png"
OUT_PDF = ROOT / "togronet_versions_config.pdf"

MODEL_COLS = [
    "V0", "V1", "V2", "V3", "V4", "V5", "V6", "Đề xuất",
]

WIDTH_SCALE = {
    "V0": 0.25, "V1": 0.25, "V2": 0.25, "V3": 0.25,
    "V4": 0.225, "V5": 0.200, "V6": 0.175, "Đề xuất": 0.225,
}

# attn=True cho khoi C3k2 o cac vi tri (STT 13, 16, 19); STT 22 luon attn=True
ATTN13 = {"V0": False, "V1": True, "V2": False, "V3": False, "V4": False, "V5": False, "V6": False, "Đề xuất": False}
ATTN16 = {"V0": False, "V1": False, "V2": True, "V3": True, "V4": False, "V5": False, "V6": False, "Đề xuất": False}
ATTN19 = {"V0": False, "V1": True, "V2": False, "V3": True, "V4": True, "V5": True, "V6": True, "Đề xuất": True}
CH16 = {"V0": 256, "V1": 256, "V2": 512, "V3": 512, "V4": 256, "V5": 256, "V6": 256, "Đề xuất": 256}

# (STT, ten khoi, so kenh co dinh hoac None neu phu thuoc model, ghi chu)
ROWS = [
    (0, "Conv", 64),
    (1, "Conv", 128),
    (2, "C3k2", 256),
    (3, "Conv", 256),
    (4, "C3k2", 512),
    (5, "Conv", 512),
    (6, "C3k2", 512),
    (7, "Conv", 1024),
    (8, "C3k2", 1024),
    (9, "SPPF", 1024),
    (10, "C2PSA", 1024),
    (11, "Upsample", None),
    (12, "Concat", None),
    (13, "C3k2", 512),
    (14, "Upsample", None),
    (15, "Concat", None),
    (16, "C3k2", "CH16"),
    (17, "Conv", 256),
    (18, "Concat", None),
    (19, "C3k2", 512),
    (20, "Conv", 512),
    (21, "Concat", None),
    (22, "C3k2", 1024),
    (23, "Detect", "nc=80"),
]


def cell(stt, base_ch, model):
    if base_ch is None:
        return "–"
    if base_ch == "nc=80":
        return "nc=80"
    ch = CH16[model] if base_ch == "CH16" else base_ch
    attn = False
    if stt == 13:
        attn = ATTN13[model]
    elif stt == 16:
        attn = ATTN16[model]
    elif stt == 19:
        attn = ATTN19[model]
    elif stt == 22:
        attn = True
    return f"{ch} (A)" if attn else f"{ch}"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "font.size": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

col_labels = ["STT", "Khối"] + MODEL_COLS
table_data = []
for stt, name, base_ch in ROWS:
    row = [str(stt), name] + [cell(stt, base_ch, m) for m in MODEL_COLS]
    table_data.append(row)

# hang cuoi: he so scale [depth, width, max_channels] ghi trong cfg
scale_row = ["", "scale [d, w, c_max]"] + [f"0.5, {WIDTH_SCALE[m]}, 1024" for m in MODEL_COLS]
table_data.append(scale_row)

n_rows = len(table_data) + 1  # +1 cho header
fig_h = 0.235 * n_rows + 0.4
fig, ax = plt.subplots(figsize=(11.5, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.35)

col_widths = [0.035, 0.11] + [0.083] * len(MODEL_COLS)
for (r, c), w in ((k, col_widths[k[1]]) for k in tbl.get_celld().keys()):
    tbl[(r, c)].set_width(w)

for (r, c), cell_obj in tbl.get_celld().items():
    cell_obj.set_edgecolor("#999999")
    cell_obj.set_linewidth(0.4)
    if r == 0:
        cell_obj.set_facecolor("#dbe6f4")
        cell_obj.set_text_props(weight="bold")
    elif r == len(table_data):  # hang scale
        cell_obj.set_facecolor("#f2f2f2")
        cell_obj.set_text_props(style="italic")
    elif c == 1:
        cell_obj.set_text_props(ha="left")

fig.text(
    0.01, 0.005,
    "(A) = khối C3k2 có attn=True (attention). Số kênh là giá trị gốc ghi trong file cấu hình .yaml (chưa nhân hệ số width).\n"
    "\"Đề xuất\" (togrow0005_backbone) dùng chung file cấu hình với V4 (togroth_m2_width_0225.yaml); chỉ khác trọng số khởi tạo (pretrain SSL backbone thay vì COCO).",
    fontsize=6.5, va="bottom",
)

fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
