"""Build a YOLO-format detection dataset from the Laboro Tomato dataset.

Source: laboro_tomato/{train,test} images + annotations/{train,test}.json
(COCO instance-segmentation format, 6 original classes: b_/l_ x
fully_ripened/half_ripened/green -- "b" = normal-size tomato, "l" = cherry
tomato). Steps:

  1. Convert each instance's polygon segmentation to a bounding box (min/max
     of all polygon points) -- the dataset's own "bbox" field already equals
     this exactly (verified), but we derive it from the polygon ourselves
     to make the conversion explicit rather than trusting a precomputed field.
  2. Merge the 6 original classes into 3 by ripeness (drop the b_/l_ size
     split): green, half_ripened, fully_ripened.
  3. Write YOLO-format labels (one .txt per image, normalized cx,cy,w,h),
     symlink images into images/{train,test}/ (no copy -- saves ~1.6GB), and
     emit data.yaml + a per-instance manifest.csv.

The dataset ships with exactly 2 splits (train=643 img, test=161 img) -- kept
as-is as train/test, with data.yaml's `val:` pointing at the test split so
Ultralytics can train/validate directly without a 3rd split.

Usage:
    python3 build_laboro_tomato_yolo_dataset.py \
        --src /mnt/data/laboro_tomato_raw/laboro_tomato \
        --dst /mnt/data/laboro_tomato_yolo
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SPLITS = ["train", "test"]

# 6 original classes -> 3 merged classes (ripeness only, size dropped).
MERGE = {
    "b_green": "green",
    "l_green": "green",
    "b_half_ripened": "half_ripened",
    "l_half_ripened": "half_ripened",
    "b_fully_ripened": "fully_ripened",
    "l_fully_ripened": "fully_ripened",
}
NEW_NAMES = ["green", "half_ripened", "fully_ripened"]
NEW_ID = {name: i for i, name in enumerate(NEW_NAMES)}


def polygon_to_bbox(segmentation: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for part in segmentation:
        xs.extend(part[0::2])
        ys.extend(part[1::2])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def build_split(src: Path, dst: Path, split: str, manifest_rows: list[dict], class_counts: Counter):
    coco = json.loads((src / "annotations" / f"{split}.json").read_text())
    old_cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {im["id"]: im for im in coco["images"]}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_dir = dst / "images" / split
    lbl_dir = dst / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images_with_instances = 0
    for image_id, im in images_by_id.items():
        anns = anns_by_image.get(image_id, [])
        if not anns:
            continue  # skip images with no annotated tomato at all
        n_images_with_instances += 1

        stem = Path(im["file_name"]).stem
        src_img = src / split / im["file_name"]
        dst_img = img_dir / im["file_name"]
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

        w, h = im["width"], im["height"]
        lines = []
        for ann in anns:
            x, y, bw, bh = polygon_to_bbox(ann["segmentation"])
            old_name = old_cat_name[ann["category_id"]]
            new_name = MERGE[old_name]
            cls_id = NEW_ID[new_name]

            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nbw, nbh = bw / w, bh / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f}")

            class_counts[(split, new_name, "objects")] += 1
            manifest_rows.append({
                "split": split,
                "filename": im["file_name"],
                "image_width": w,
                "image_height": h,
                "orig_class": old_name,
                "merged_class": new_name,
                "class_id": cls_id,
                "bbox_x": round(x, 2),
                "bbox_y": round(y, 2),
                "bbox_w": round(bw, 2),
                "bbox_h": round(bh, 2),
            })
        class_counts[(split, "__images__", "images")] += 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    return n_images_with_instances, len(coco["images"]), len(coco["annotations"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/mnt/data/laboro_tomato_raw/laboro_tomato")
    ap.add_argument("--dst", default="/mnt/data/laboro_tomato_yolo")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    class_counts: Counter = Counter()
    split_stats = {}
    for split in SPLITS:
        n_used, n_total, n_ann = build_split(src, dst, split, manifest_rows, class_counts)
        split_stats[split] = {"images_total": n_total, "images_with_instances": n_used, "instances": n_ann}
        print(f"[{split}] {n_used}/{n_total} images with >=1 instance, {n_ann} instances -> "
              f"{dst/'images'/split} + {dst/'labels'/split}")

    manifest_path = dst / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"wrote {manifest_path} ({len(manifest_rows)} instance rows)")

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(NEW_NAMES))
    (dst / "data.yaml").write_text(
        "# Laboro Tomato, converted: polygon segmentation -> bbox, 6 classes -> 3 (ripeness only)\n"
        "# source: https://github.com/laboroai/LaboroTomato (CC BY-NC-SA 4.0)\n"
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/test\n"
        "\n"
        f"nc: {len(NEW_NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    print(f"wrote {dst/'data.yaml'}")

    report = {
        "source": "https://github.com/laboroai/LaboroTomato",
        "original_classes": list(MERGE.keys()),
        "merged_classes": NEW_NAMES,
        "merge_map": MERGE,
        "splits": split_stats,
    }
    (dst / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {dst/'build_report.json'}")


if __name__ == "__main__":
    main()
