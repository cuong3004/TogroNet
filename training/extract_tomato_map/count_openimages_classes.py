#!/usr/bin/env python3
"""
Phan tich thong ke lop doi tuong (Tomato, Plant, Houseplant, ...) tu du lieu
annotation (bounding box) cua Open Images V7 - KHONG can tai anh that.

Chay truoc: bash download_openimages_annotations.sh
(tai + loc cac file bbox_filtered_{train,validation,test}.csv vao
openimages_stats_data/, chi giu lai cac dong LabelName thuoc 10 lop ben duoi)

Dinh nghia:
  - "object"  = 1 bounding box duoc gan nhan cho 1 lop trong 1 anh
                (1 anh co the co nhieu object cung lop, vd 5 qua Tomato)
  - "image"   = 1 anh (ImageID) co it nhat 1 bounding box cua lop do
  - "Tomato+Plant" (giao/AND) = so anh vua co it nhat 1 box Tomato,
                vua co it nhat 1 box Plant (khong nhat thiet cung box)
"""

import csv
import sys
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "openimages_stats_data"

# Thu tu nay QUYET DINH thu tu cong don giao (AND) cua bao cao phan 2
list_classes = """
Tomato
Plant
Houseplant
Flower
Flowerpot
Tree
Vegetable
Fruit
Bell pepper
Cucumber
""".strip().split("\n")

NAME_TO_MID = {
    "Tomato": "/m/07j87",
    "Plant": "/m/05s2s",
    "Houseplant": "/m/03fp41",
    "Flower": "/m/0c9ph5",
    "Flowerpot": "/m/0fm3zh",
    "Tree": "/m/07j7r",
    "Vegetable": "/m/0f4s2w",
    "Fruit": "/m/02xwb",
    "Bell pepper": "/m/0jg57",
    "Cucumber": "/m/015x4r",
}
MID_TO_NAME = {v: k for k, v in NAME_TO_MID.items()}

BBOX_FILES = [
    DATA_DIR / "bbox_filtered_train.csv",
    DATA_DIR / "bbox_filtered_validation.csv",
    DATA_DIR / "bbox_filtered_test.csv",
]


def aggregate():
    """Doc tung dong CSV streaming (khong giu ca file trong RAM), chi trich
    3 cot can dung (Split, LabelName, ImageID) roi cap nhat thang vao
    set/counter. Moi dong bi bo sau khi xu ly xong nen RAM chi ti le voi
    so ImageID/object DUY NHAT, khong ti le voi tong so dong file (~1.87M)."""
    image_sets = {c: {"train": set(), "validation": set(), "test": set()} for c in list_classes}
    object_counts = {c: {"train": 0, "validation": 0, "test": 0} for c in list_classes}

    missing = []
    found_any = False
    for fp in BBOX_FILES:
        if not fp.exists():
            missing.append(fp)
            continue
        found_any = True
        with open(fp, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_split = header.index("Split")
            idx_label = header.index("LabelName")
            idx_image = header.index("ImageID")
            for row in reader:
                cname = MID_TO_NAME.get(row[idx_label])
                if cname is None:
                    continue
                split = row[idx_split]
                image_sets[cname][split].add(row[idx_image])
                object_counts[cname][split] += 1

    if missing:
        print("[CANH BAO] Thieu cac file sau (chay 'bash download_openimages_annotations.sh' truoc):", file=sys.stderr)
        for fp in missing:
            print(f"  - {fp}", file=sys.stderr)
    if not found_any:
        print("Khong doc duoc du lieu nao. Hay chay: bash download_openimages_annotations.sh", file=sys.stderr)
        sys.exit(1)

    return image_sets, object_counts


def main():
    image_sets, object_counts = aggregate()

    # ---- Phan 1: thong ke tung lop rieng le (object count + image count) ----
    per_class_report = []
    for c in list_classes:
        n_train_img = len(image_sets[c]["train"])
        n_val_img = len(image_sets[c]["validation"])
        n_test_img = len(image_sets[c]["test"])
        n_train_obj = object_counts[c]["train"]
        n_val_obj = object_counts[c]["validation"]
        n_test_obj = object_counts[c]["test"]
        per_class_report.append({
            "class": c,
            "mid": NAME_TO_MID[c],
            "num_images_train": n_train_img,
            "num_images_validation": n_val_img,
            "num_images_test": n_test_img,
            "num_images_total": n_train_img + n_val_img + n_test_img,
            "num_objects_train": n_train_obj,
            "num_objects_validation": n_val_obj,
            "num_objects_test": n_test_obj,
            "num_objects_total": n_train_obj + n_val_obj + n_test_obj,
        })

    out1 = BASE / "report_per_class.csv"
    with open(out1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_report[0].keys()))
        w.writeheader()
        w.writerows(per_class_report)
    print(f"[OK] Da ghi {out1}")

    # ---- Phan 2: giao (AND) cong don theo thu tu list_classes, tung split + tong ----
    def images_for_class_split(cname, split):
        return image_sets[cname][split]

    def images_for_class_all_splits(cname):
        return (
            image_sets[cname]["train"]
            | image_sets[cname]["validation"]
            | image_sets[cname]["test"]
        )

    combo_report = []
    running_sets = {"train": None, "validation": None, "test": None, "total": None}
    combo_names = []
    for c in list_classes:
        combo_names.append(c)
        combo_label = "+".join(combo_names)

        row = {"combo": combo_label, "n_classes": len(combo_names)}
        for split in ["train", "validation", "test"]:
            cur = images_for_class_split(c, split)
            running_sets[split] = cur if running_sets[split] is None else (running_sets[split] & cur)
            row[f"num_images_{split}"] = len(running_sets[split])

        cur_total = images_for_class_all_splits(c)
        running_sets["total"] = cur_total if running_sets["total"] is None else (running_sets["total"] & cur_total)
        row["num_images_total"] = len(running_sets["total"])

        combo_report.append(row)

    out2 = BASE / "report_cumulative_intersection.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combo_report[0].keys()))
        w.writeheader()
        w.writerows(combo_report)
    print(f"[OK] Da ghi {out2}")

    print("\n=== THONG KE TUNG LOP (tong 3 split) ===")
    print(f"{'Class':<12} {'#Images':>10} {'#Objects':>10}")
    for r in per_class_report:
        print(f"{r['class']:<12} {r['num_images_total']:>10} {r['num_objects_total']:>10}")

    print("\n=== GIAO CONG DON (AND) THEO THU TU list_classes, tong 3 split ===")
    for r in combo_report:
        print(f"{r['combo']:<70} #images = {r['num_images_total']}")


if __name__ == "__main__":
    main()
