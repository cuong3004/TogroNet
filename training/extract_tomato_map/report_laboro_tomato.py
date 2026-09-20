"""Per-class CSV report for the converted Laboro Tomato YOLO dataset (built by
build_laboro_tomato_yolo_dataset.py), reading manifest.csv.

Usage:
    python3 report_laboro_tomato.py --dataset /mnt/data/laboro_tomato_yolo
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

NEW_NAMES = ["green", "half_ripened", "fully_ripened"]
ORIG_BY_MERGED = {
    "green": ["b_green", "l_green"],
    "half_ripened": ["b_half_ripened", "l_half_ripened"],
    "fully_ripened": ["b_fully_ripened", "l_fully_ripened"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="/mnt/data/laboro_tomato_yolo")
    args = ap.parse_args()
    root = Path(args.dataset)

    with open(root / "manifest.csv") as f:
        rows = list(csv.DictReader(f))

    images_by_split_class = defaultdict(set)
    objects = Counter()
    for r in rows:
        images_by_split_class[(r["split"], r["merged_class"])].add(r["filename"])
        objects[(r["split"], r["merged_class"])] += 1

    all_images = defaultdict(set)
    for r in rows:
        all_images[r["split"]].add(r["filename"])

    report = []
    for cname in NEW_NAMES:
        n_tr_img = len(images_by_split_class[("train", cname)])
        n_te_img = len(images_by_split_class[("test", cname)])
        n_tr_obj = objects[("train", cname)]
        n_te_obj = objects[("test", cname)]
        report.append({
            "class": cname,
            "merged_from": "+".join(ORIG_BY_MERGED[cname]),
            "num_images_train": n_tr_img,
            "num_objects_train": n_tr_obj,
            "num_images_test": n_te_img,
            "num_objects_test": n_te_obj,
            "num_images_total": n_tr_img + n_te_img,
            "num_objects_total": n_tr_obj + n_te_obj,
            "avg_objects_per_image_train": round(n_tr_obj / n_tr_img, 2) if n_tr_img else 0,
            "avg_objects_per_image_test": round(n_te_obj / n_te_img, 2) if n_te_img else 0,
        })

    total_row = {
        "class": "ALL",
        "merged_from": "",
        "num_images_train": len(all_images["train"]),
        "num_objects_train": sum(objects[("train", c)] for c in NEW_NAMES),
        "num_images_test": len(all_images["test"]),
        "num_objects_test": sum(objects[("test", c)] for c in NEW_NAMES),
        "num_images_total": len(all_images["train"]) + len(all_images["test"]),
        "num_objects_total": sum(objects.values()),
        "avg_objects_per_image_train": round(sum(objects[("train", c)] for c in NEW_NAMES) / len(all_images["train"]), 2),
        "avg_objects_per_image_test": round(sum(objects[("test", c)] for c in NEW_NAMES) / len(all_images["test"]), 2),
    }
    report.append(total_row)

    out = Path(__file__).parent / "report_laboro_tomato.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"[OK] wrote {out}")

    print(f"\n{'Class':<15} {'#Img train':>10} {'#Obj train':>10} {'#Img test':>9} {'#Obj test':>9} {'#Img total':>10} {'#Obj total':>10}")
    for r in report:
        print(f"{r['class']:<15} {r['num_images_train']:>10} {r['num_objects_train']:>10} "
              f"{r['num_images_test']:>9} {r['num_objects_test']:>9} {r['num_images_total']:>10} {r['num_objects_total']:>10}")


if __name__ == "__main__":
    main()
