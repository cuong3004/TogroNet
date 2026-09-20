#!/usr/bin/env python3
"""
Hien thi tom tat luot survey (tomato_survey) + suy luan (batch_infer.py)
gan nhat, chia 4 vung:
  1. Sinh truong sinh duong
  2. Ra hoa
  3. Co qua (gom hinh_thanh_qua + qua_chuyen_chin)
  4. Thong ke so qua (tong so qua, so qua chin)

Tu dong lam moi dinh ky de bat kip luot survey moi neu dang chay nhu mot
service dai han.
"""

import csv
import glob
import os
import tkinter as tk

from PIL import Image, ImageTk

GUI_WIDTH = 684
GUI_HEIGHT = 600

TOMATO_RUNS_DIR = os.path.expanduser("~/tomato_survey_runs")

# Neu dat, GUI se "ghim" vao dung luot nay thay vi luot moi nhat (dung de
# xem lai 1 luot cu cu the). Bo trong (mac dinh) -> luon lay luot moi nhat.
FORCE_RUN_DIR = os.environ.get("GROWTH_FORCE_RUN_DIR") or None

STAGE_VEG_GROWTH = "sinh_truong_sinh_duong"
STAGE_FLOWERING = "ra_hoa"
STAGE_FRUIT_STAGES = ("hinh_thanh_qua", "qua_chuyen_chin")

REFRESH_INTERVAL_MS = 10_000
CELL_IMG_SIZE = (300, 240)


def find_latest_run_with_results():
    if FORCE_RUN_DIR:
        if os.path.isfile(os.path.join(FORCE_RUN_DIR, "results_table.csv")):
            return FORCE_RUN_DIR
        return None
    if not os.path.isdir(TOMATO_RUNS_DIR):
        return None
    candidates = sorted(
        glob.glob(os.path.join(TOMATO_RUNS_DIR, "run_*")),
        reverse=True,
    )
    for run_dir in candidates:
        if os.path.isfile(os.path.join(run_dir, "results_table.csv")):
            return run_dir
    return None


def read_results_csv(csv_path):
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Cannot read {csv_path}: {e}")
    return rows


def pick_best_row(rows, stage_names):
    best = None
    best_conf = -1.0
    for row in rows:
        if row.get("stage") not in stage_names:
            continue
        try:
            conf = float(row.get("stage_confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf > best_conf:
            best_conf = conf
            best = row
    return best


def pick_most_fruit_row(rows, stage_names):
    """
    Trong cac dong co stage nam trong stage_names, tra ve dong co
    fruit_total (so qua phat hien) cao nhat. None neu khong co dong nao
    khop hoac khong dong nao co fruit_total hop le.
    """
    best = None
    best_count = -1
    for row in rows:
        if row.get("stage") not in stage_names:
            continue
        fruit_total = row.get("fruit_total")
        if fruit_total in (None, "NA", ""):
            continue
        try:
            count = int(fruit_total)
        except (TypeError, ValueError):
            continue
        if count > best_count:
            best_count = count
            best = row
    return best


def categorize_positions(rows):
    """
    Gan MOI vi tri (vd vi_tri_1..4) vao DUNG 1 trong 3 nhom: fruit,
    flowering, vegetative, de tranh 1 vi tri bi hien lap o nhieu vung
    khac nhau tren GUI.

    Dung DA SO PHIEU: nhom nao co nhieu anh nhat trong so cac anh hop le
    (khong tinh khong_ro/bo_qua_trung_lap) cua vi tri do thi thang - tranh
    truong hop 1 anh nhan dang SAI (vd chup trung cua, khong phai cay,
    nhung model tu tin nham) lan at ca loat anh khac dong thuan. Chi khi
    HOA PHIEU moi dung thu tu uu tien fruit > flowering > vegetative.

    Luu y: anh stage=hinh_thanh_qua/qua_chuyen_chin nhung Fruit KHONG dem
    duoc qua nao (fruit_total=0) thi KHONG tinh la phieu "fruit" - vi
    stage chi la du doan tho, khong co qua thuc su duoc dem thay thi
    khong nen coi la bang chung manh cho "vi tri nay co qua" (da kiem
    chung: gay hoa phieu sai, uu tien nham "fruit" du fruit_total=0 het).

    Tra ve dict {position: "fruit" | "flowering" | "vegetative"}.
    """
    position_stage_counts = {}
    for row in rows:
        stage = row.get("stage")
        position = row.get("position")
        if not position or stage in (None, "khong_ro", "bo_qua_trung_lap"):
            continue
        group = None
        if stage in STAGE_FRUIT_STAGES:
            try:
                fruit_total = int(row.get("fruit_total"))
            except (TypeError, ValueError):
                fruit_total = 0
            group = "fruit" if fruit_total > 0 else None
        elif stage == STAGE_FLOWERING:
            group = "flowering"
        elif stage == STAGE_VEG_GROWTH:
            group = "vegetative"
        if group is None:
            continue
        counts = position_stage_counts.setdefault(
            position, {"fruit": 0, "flowering": 0, "vegetative": 0}
        )
        counts[group] += 1

    priority = {"fruit": 0, "flowering": 1, "vegetative": 2}
    categories = {}
    for position, counts in position_stage_counts.items():
        if sum(counts.values()) == 0:
            continue
        # Sap xep theo (so phieu giam dan, uu tien tang dan) - da so
        # thang, hoa thi dung thu tu uu tien fruit > flowering > vegetative.
        best_group = min(
            counts, key=lambda g: (-counts[g], priority[g])
        )
        if counts[best_group] > 0:
            categories[position] = best_group
    return categories


def rows_for_category(rows, categories, category_name):
    positions = {p for p, c in categories.items() if c == category_name}
    return [r for r in rows if r.get("position") in positions]


def collect_history_stats(max_runs=8):
    """
    Duyet cac thu muc run_* co results_table.csv, tinh (tong so qua, so
    qua chin) cho tung luot, sap xep theo thoi gian (cu -> moi), chi lay
    toi da max_runs luot GAN NHAT.

    Tra ve list[(nhan_luot, tong_qua, qua_chin)].
    """
    if not os.path.isdir(TOMATO_RUNS_DIR):
        return []

    run_dirs = sorted(glob.glob(os.path.join(TOMATO_RUNS_DIR, "run_*")))
    if FORCE_RUN_DIR:
        # Dang ghim vao 1 luot cu cu the - chi lay lich su TINH DEN luot
        # do, khong tinh cac luot xay ra SAU do (chua ton tai "tai thoi
        # diem" dang xem).
        run_dirs = [d for d in run_dirs if d <= FORCE_RUN_DIR]
    history = []
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "results_table.csv")
        if not os.path.isfile(csv_path):
            continue
        rows = read_results_csv(csv_path)
        total_fruit, total_ripe = compute_fruit_stats(rows)
        name = os.path.basename(run_dir)
        # run_YYYYMMDD_HHMMSS -> lay HH:MM lam nhan ngan gon.
        label = name[-6:-2] if len(name) >= 15 else name
        history.append((label, total_fruit, total_ripe))

    return history[-max_runs:]


def compute_fruit_stats(rows):
    total_fruit = 0
    total_ripe = 0
    for row in rows:
        fruit_total = row.get("fruit_total")
        if fruit_total in (None, "NA", ""):
            continue
        try:
            total_fruit += int(fruit_total)
            total_ripe += int(row.get("qua_chin") or 0)
        except (TypeError, ValueError):
            continue
    return total_fruit, total_ripe


class GrowthSummaryGUI:

    def __init__(self, root):
        self.root = root
        self.current_run_dir = None
        self.photo_refs = []

        root.title("Growth - Tom tat luot khao sat gan nhat")
        root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}+340+0")
        root.resizable(False, False)
        root.configure(bg="#eef2f5")

        self.status_var = tk.StringVar(value="Dang tim luot gan nhat...")
        status_bar = tk.Label(
            root, textvariable=self.status_var, font=("Arial", 9),
            bg="#eef2f5", fg="#555555", anchor="w",
        )
        status_bar.pack(fill="x", padx=6, pady=(4, 4))

        self.last_run_dir = None
        self.last_csv_mtime = None
        self.grid_built = False

        self.grid_frame = tk.Frame(root, bg="#eef2f5")
        self.grid_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.grid_frame.grid_rowconfigure(0, weight=1)
        self.grid_frame.grid_rowconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)

        self.refresh()
        self.root.after(REFRESH_INTERVAL_MS, self.auto_refresh_loop)

    def auto_refresh_loop(self):
        self.refresh()
        self.root.after(REFRESH_INTERVAL_MS, self.auto_refresh_loop)

    def load_image(self, path, size):
        try:
            img = Image.open(path)
            img = img.resize(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Cannot load image {path}: {e}")
            return None

    def refresh(self):
        run_dir = find_latest_run_with_results()

        if run_dir is None:
            # Ve 1 lan du chua co du lieu, de GUI khong "treo" o trang thai
            # ban dau chua bao gio duoc ve (vd service vua khoi dong ma
            # chua co luot nao xong ca).
            if not self.grid_built or self.last_run_dir is not None:
                self.status_var.set(
                    "Chua co luot survey nao da chay suy luan (batch_infer.py)."
                )
                try:
                    self.build_grid(None, [])
                except Exception as e:
                    print(f"build_grid failed, will retry next poll: {e}")
                    return
                self.grid_built = True
                self.last_run_dir = None
                self.last_csv_mtime = None
            return

        csv_path = os.path.join(run_dir, "results_table.csv")
        try:
            csv_mtime = os.path.getmtime(csv_path)
        except OSError:
            csv_mtime = -1

        # Chi ve lai neu doi sang luot khac HOAC results_table.csv cua
        # luot hien tai vua duoc ghi lai (vd chay lai batch_infer.py voi
        # cach loc/chon anh moi tren cung 1 luot cu - so anh trong
        # images/ khong doi nhung NOI DUNG bang ket qua da doi).
        if (
            self.grid_built
            and run_dir == self.last_run_dir
            and csv_mtime == self.last_csv_mtime
        ):
            return

        self.status_var.set(f"Luot gan nhat: {os.path.basename(run_dir)}")
        rows = read_results_csv(
            os.path.join(run_dir, "results_table.csv")
        )
        try:
            self.build_grid(run_dir, rows)
        except Exception as e:
            # Khong danh dau da cap nhat neu ve loi giua chung (vd doc
            # anh dung luc batch_infer.py dang ghi file) - de lan poll
            # sau thu lai, tranh bi "ket" o trang thai trong/loi mai mai.
            print(f"build_grid failed, will retry next poll: {e}")
            return

        self.grid_built = True
        self.last_run_dir = run_dir
        self.last_csv_mtime = csv_mtime

    def build_grid(self, run_dir, rows):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.photo_refs = []

        images_dir = os.path.join(run_dir, "images") if run_dir else None

        def build_image_cell(title, row_data):
            frame = tk.Frame(
                self.grid_frame, bg="#ffffff", relief="solid", bd=1
            )
            tk.Label(
                frame, text=title, font=("Arial", 11, "bold"),
                bg="#ffffff", fg="#1f2933"
            ).pack(pady=(6, 2))

            if row_data is None:
                tk.Label(
                    frame, text="Khong co du lieu",
                    font=("Arial", 9), bg="#ffffff", fg="#888888"
                ).pack(expand=True)
                return frame

            image_path = os.path.join(images_dir, row_data["image_id"])
            photo = self.load_image(image_path, CELL_IMG_SIZE)
            if photo is not None:
                self.photo_refs.append(photo)
                tk.Label(frame, image=photo, bg="#ffffff").pack(pady=(0, 6))
            else:
                tk.Label(
                    frame, text="(khong doc duoc anh)",
                    font=("Arial", 9), bg="#ffffff", fg="#888888"
                ).pack(expand=True)
            return frame

        # Moi vi tri chi thuoc DUNG 1 nhom (fruit > flowering > vegetative)
        # de tranh 1 vi tri hien lap o nhieu vung khac nhau.
        categories = categorize_positions(rows)
        veg_row = pick_best_row(
            rows_for_category(rows, categories, "vegetative"), (STAGE_VEG_GROWTH,)
        )
        flower_row = pick_best_row(
            rows_for_category(rows, categories, "flowering"), (STAGE_FLOWERING,)
        )
        fruit_row = pick_most_fruit_row(
            rows_for_category(rows, categories, "fruit"), STAGE_FRUIT_STAGES
        )

        cell1 = build_image_cell("Sinh truong sinh duong", veg_row)
        cell1.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        cell2 = build_image_cell("Ra hoa", flower_row)
        cell2.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        cell3 = build_image_cell("Co qua", fruit_row)
        cell3.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        stats_frame = tk.Frame(
            self.grid_frame, bg="#ffffff", relief="solid", bd=1
        )
        stats_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        tk.Label(
            stats_frame, text="Thong ke so qua", font=("Arial", 11, "bold"),
            bg="#ffffff", fg="#1f2933"
        ).pack(pady=(6, 10))

        total_fruit, total_ripe = compute_fruit_stats(rows)

        tk.Label(
            stats_frame, text=f"Tong so qua: {total_fruit}",
            font=("Arial", 14), bg="#ffffff", fg="#1f2933"
        ).pack(pady=4)

        tk.Label(
            stats_frame, text=f"So qua chin: {total_ripe}",
            font=("Arial", 14), bg="#ffffff", fg="#c0392b"
        ).pack(pady=4)

        self.draw_history_chart(stats_frame)

    def draw_history_chart(self, parent):
        history = collect_history_stats(max_runs=8)
        if len(history) < 2:
            # Chua du du lieu de ve xu huong (can >= 2 luot).
            return

        canvas_w, canvas_h = 300, 130
        canvas = tk.Canvas(
            parent, width=canvas_w, height=canvas_h,
            bg="#ffffff", highlightthickness=0,
        )
        canvas.pack(pady=(6, 4))

        max_val = max(
            (max(fruit, ripe) for _, fruit, ripe in history), default=0
        )
        max_val = max(max_val, 1)

        n = len(history)
        margin_bottom = 18
        chart_h = canvas_h - margin_bottom
        group_w = canvas_w / n
        bar_w = group_w * 0.32

        for idx, (label, fruit, ripe) in enumerate(history):
            group_x = idx * group_w
            fruit_h = (fruit / max_val) * (chart_h - 8)
            ripe_h = (ripe / max_val) * (chart_h - 8)

            x1 = group_x + group_w * 0.15
            canvas.create_rectangle(
                x1, chart_h - fruit_h, x1 + bar_w, chart_h,
                fill="#2e8b57", outline="",
            )
            x2 = group_x + group_w * 0.53
            canvas.create_rectangle(
                x2, chart_h - ripe_h, x2 + bar_w, chart_h,
                fill="#c0392b", outline="",
            )
            canvas.create_text(
                group_x + group_w / 2, chart_h + 9,
                text=label, font=("Arial", 7), fill="#555555",
            )

        canvas.create_line(0, chart_h, canvas_w, chart_h, fill="#cccccc")

        legend = tk.Frame(parent, bg="#ffffff")
        legend.pack()
        tk.Label(
            legend, text="■ Tong qua", font=("Arial", 8),
            bg="#ffffff", fg="#2e8b57",
        ).pack(side="left", padx=4)
        tk.Label(
            legend, text="■ Qua chin", font=("Arial", 8),
            bg="#ffffff", fg="#c0392b",
        ).pack(side="left", padx=4)


def main():
    root = tk.Tk()
    GrowthSummaryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
