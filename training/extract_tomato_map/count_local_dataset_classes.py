#!/usr/bin/env python3
"""
Thong ke object/image cho 10 lop CHI TREN TAP ANH DA TAI VE THUC TE qua
FiftyOne tai ~/fiftyone/open-images-v7/{train,validation}/data/ (vd:
300000/~1.7M anh train, 5000/41620 anh validation da chon), KHONG PHAI
toan bo split goc (nen so se nho hon report_per_class.csv truoc do).

Chay truoc (neu chua co): bash download_openimages_annotations.sh

Thiet ke gioi han bo nho (RAM thap):
  - Danh sach anh cuc bo: chi doc TEN FILE trong thu muc data/ (scandir),
    KHONG mo/doc noi dung anh -> 1 set ID/split (~300k + 5k chuoi).
  - Annotation: doc bbox_filtered_{train,validation}.csv theo TUNG DONG
    (csv.reader streaming), khong load ca file / khong dung DictReader
    giu list - moi dong duoc doi chieu voi set ID cuc bo roi vut ngay.
  - Xu ly xong split nao thi giai phong (del) set ID cuc bo cua split do
    truoc khi sang split ke tiep -> peak RAM ~ max(1 split), khong phai
    tong ca 2 split cong don.
"""

import csv
import os
import sys
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "openimages_stats_data"
FIFTYONE_DIR = Path.home() / "fiftyone" / "open-images-v7"

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

SPLITS = ["train", "validation"]
BBOX_FILES = {
    "train": DATA_DIR / "bbox_filtered_train.csv",
    "validation": DATA_DIR / "bbox_filtered_validation.csv",
}


def load_local_image_ids(split):
    """Chi doc TEN FILE (khong doc noi dung anh) trong <split>/data/."""
    data_dir = FIFTYONE_DIR / split / "data"
    if not data_dir.is_dir():
        print(f"[LOI] Khong thay thu muc: {data_dir}", file=sys.stderr)
        sys.exit(1)
    ids = set()
    with os.scandir(data_dir) as it:
        for entry in it:
            if entry.is_file():
                ids.add(os.path.splitext(entry.name)[0])
    return ids


def aggregate():
    image_sets = {c: {s: set() for s in SPLITS} for c in list_classes}
    object_counts = {c: {s: 0 for s in SPLITS} for c in list_classes}
    local_total = {}

    for split in SPLITS:
        local_ids = load_local_image_ids(split)
        local_total[split] = len(local_ids)
        print(f"[{split}] So anh cuc bo tren dia: {len(local_ids)}", file=sys.stderr)

        fp = BBOX_FILES[split]
        if not fp.exists():
            print(f"[CANH BAO] Thieu {fp} - chay download_openimages_annotations.sh truoc", file=sys.stderr)
            del local_ids
            continue

        with open(fp, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_label = header.index("LabelName")
            idx_image = header.index("ImageID")
            for row in reader:
                image_id = row[idx_image]
                if image_id not in local_ids:
                    continue
                cname = MID_TO_NAME.get(row[idx_label])
                if cname is None:
                    continue
                image_sets[cname][split].add(image_id)
                object_counts[cname][split] += 1

        del local_ids  # giai phong RAM truoc khi sang split ke tiep

    return image_sets, object_counts, local_total


def main():
    image_sets, object_counts, local_total = aggregate()

    report = []
    for c in list_classes:
        n_tr_img = len(image_sets[c]["train"])
        n_va_img = len(image_sets[c]["validation"])
        n_tr_obj = object_counts[c]["train"]
        n_va_obj = object_counts[c]["validation"]
        report.append({
            "class": c,
            "mid": NAME_TO_MID[c],
            "local_images_train": local_total["train"],
            "num_images_train": n_tr_img,
            "num_objects_train": n_tr_obj,
            "local_images_validation": local_total["validation"],
            "num_images_validation": n_va_img,
            "num_objects_validation": n_va_obj,
            "num_images_total": n_tr_img + n_va_img,
            "num_objects_total": n_tr_obj + n_va_obj,
        })

    out = BASE / "report_per_class_local_dataset.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"[OK] Da ghi {out}")

    print(f"\n=== THONG KE TREN TAP ANH CUC BO (train={local_total['train']}, validation={local_total['validation']}) ===")
    print(f"{'Class':<12} {'#Img train':>11} {'#Obj train':>11} {'#Img val':>9} {'#Obj val':>9} {'#Img total':>11}")
    for r in report:
        print(f"{r['class']:<12} {r['num_images_train']:>11} {r['num_objects_train']:>11} "
              f"{r['num_images_validation']:>9} {r['num_objects_validation']:>9} {r['num_images_total']:>11}")


if __name__ == "__main__":
    main()
