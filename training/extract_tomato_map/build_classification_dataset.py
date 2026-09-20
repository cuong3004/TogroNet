"""Reorganize a YOLO-detection-format TomatoMAP-derived dataset (built by
build_tomatomap_detection_dataset.py, manifest.csv has a 'state' column) into
Ultralytics' classification folder layout:

    <output>/train/<class_name>/*.jpg
    <output>/val/<class_name>/*.jpg

No bbox/label files are needed for classification -- the class is the parent
folder name. Images are hard-linked (not copied) from the source dataset, so
this is fast and uses no extra disk space as long as --dataset and --output
are on the same filesystem; falls back to copying otherwise.

Usage:
    python3 build_classification_dataset.py \
        --dataset /mnt/data/tomatoMAP_det_derived_robot \
        --output /mnt/data/tomatoMAP_cls_robot
"""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="source detection-format dataset (has manifest.csv)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src_root = Path(args.dataset)
    out_root = Path(args.output)

    with open(src_root / "manifest.csv") as f:
        rows = list(csv.DictReader(f))

    counts = Counter()
    for row in rows:
        split = row["split"]
        state = row["state"]
        filename = row["filename"]
        src_img = src_root / "images" / split / filename
        dst_dir = out_root / split / state
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_img = dst_dir / filename
        if not dst_img.exists():
            link_or_copy(src_img, dst_img)
        counts[(split, state)] += 1

    report = {
        "source_dataset": str(src_root.resolve()),
        "total_images": len(rows),
        "class_names": sorted({r["state"] for r in rows}),
        "counts": {f"{split}/{state}": n for (split, state), n in sorted(counts.items())},
    }
    (out_root / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
