"""Build the labeled detection dataset D_L described in thesis section 3.2
from the raw TomatoMAP FiftyOne export.

For each `det`-tagged sample in TomatoMAP:
  - keep the largest "whole plant" bounding box b_j
  - map its BBCH code z_j to a growth-stage label s_j (Table 3.2)
  - resize the image (short side -> --min-side, aspect ratio preserved)
  - write a YOLO-style (images/, labels/) dataset split by plant_id

Usage:
    python3 build_tomatomap_detection_dataset.py \
        --source /mnt/data/tomatoMAP \
        --output /mnt/data/tomatoMAP_det_derived
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

STATE_NAMES = ["vegetative_growth", "flowering", "fruit_development", "fruit_ripening"]
STATE_TO_ID = {name: i for i, name in enumerate(STATE_NAMES)}

# BBCH principal growth stages 1+2 (10-29, leaf development + side-shoot
# formation) are merged into one "vegetative_growth" class. Kept separate
# they gave "vegetative" (20-29) only 432 train images from 18/101 plants,
# which a detector essentially never learned (near-zero recall). Merged with
# "seedling" (13-19) it draws from 60/101 plants and lands within ~2.8x of
# the largest class instead of ~28x -- a real fix, not just relabeling.
BBCH_RANGES = [
    (13, 29, "vegetative_growth"),
    (51, 69, "flowering"),
    (70, 79, "fruit_development"),
    (80, 89, "fruit_ripening"),
]


def map_bbch_to_state(code: int) -> str | None:
    for lo, hi, state in BBCH_RANGES:
        if lo <= code <= hi:
            return state
    return None


def select_whole_plant_box(sample: dict) -> dict | None:
    gt = sample.get("ground_truth")
    if not gt:
        return None
    boxes = [d for d in gt.get("detections", []) if d.get("label") == "whole plant"]
    if not boxes:
        return None
    return max(boxes, key=lambda d: d["bounding_box"][2] * d["bounding_box"][3])


def to_center_bbox(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    xc = x + w / 2
    yc = y + h / 2
    return (
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def load_det_records(source: Path, keep_piids: set[int] | None = None):
    with open(source / "samples.json") as f:
        data = json.load(f)

    records = []
    excluded = []
    for s in data["samples"]:
        if "det" not in s.get("tags", []):
            continue

        filepath = s["filepath"]
        plant_id = s.get("plant_id")
        image_set_id = s.get("image_set_id")
        piid = s.get("piid")
        pose_id = s.get("pose_id")
        meta = s.get("metadata") or {}
        orig_w, orig_h = meta.get("width"), meta.get("height")

        if keep_piids is not None and piid not in keep_piids:
            continue

        box = select_whole_plant_box(s)
        if box is None:
            excluded.append((filepath, "no_whole_plant_box"))
            continue

        cls = s.get("classification") or {}
        label = cls.get("label")
        if not label or not label.startswith("bbch_"):
            excluded.append((filepath, "no_bbch_label"))
            continue
        bbch_code = int(label.split("_")[1])
        state = map_bbch_to_state(bbch_code)
        if state is None:
            excluded.append((filepath, "bbch_out_of_range"))
            continue

        xc, yc, w, h = to_center_bbox(box["bounding_box"])
        if w <= 0 or h <= 0:
            excluded.append((filepath, "invalid_bbox"))
            continue

        records.append(
            {
                "filepath": filepath,
                "plant_id": plant_id,
                "image_set_id": image_set_id,
                "piid": piid,
                "pose_id": pose_id,
                "bbch_stage": bbch_code,
                "state": state,
                "bbox": (xc, yc, w, h),
                "orig_w": orig_w,
                "orig_h": orig_h,
            }
        )
    return records, excluded


def resize_and_save(src_path: Path, dst_path: Path, min_side: int, quality: int):
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min_side / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        im.save(dst_path, "JPEG", quality=quality)
        return new_w, new_h


def split_by_plant(records, val_ratio: float, seed: int):
    """Greedy multi-label stratified split of plant_ids into train/val.

    A plain "dominant state per plant" stratification fails here: 94/101
    plants are dominated by the "fruiting" state (each plant is imaged
    over 163 days and spends most sessions in that stage), so sklearn's
    per-class stratify can't even guarantee 2 plants per class. Instead we
    balance per-STATE image counts directly: process states from rarest to
    most common, and for each state assign the plants that contribute the
    most images of it to the val split first (until its target quota is
    reached), then the rest to train. This keeps every one of the 5 states
    represented in both splits close to `val_ratio`, and naturally lands
    the overall image-level split close to `val_ratio` too.
    """
    plant_state_counts: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        plant_state_counts[r["plant_id"]][r["state"]] += 1

    plant_ids = list(plant_state_counts.keys())
    rng = random.Random(seed)
    rng.shuffle(plant_ids)

    total_state_counts = Counter()
    for c in plant_state_counts.values():
        total_state_counts.update(c)
    target_val = {st: cnt * val_ratio for st, cnt in total_state_counts.items()}

    val_state_counts = Counter()
    train_ids, val_ids = set(), set()
    assigned = set()
    for state in sorted(total_state_counts, key=lambda s: total_state_counts[s]):
        candidates = [pid for pid in plant_ids if plant_state_counts[pid][state] > 0 and pid not in assigned]
        candidates.sort(key=lambda pid: -plant_state_counts[pid][state])
        for pid in candidates:
            if pid in assigned:
                continue
            if val_state_counts[state] < target_val[state]:
                val_ids.add(pid)
                val_state_counts.update(plant_state_counts[pid])
            else:
                train_ids.add(pid)
            assigned.add(pid)

    for pid in plant_ids:
        if pid not in assigned:
            train_ids.add(pid)

    return train_ids, val_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="/mnt/data/tomatoMAP")
    ap.add_argument("--output", default="/mnt/data/tomatoMAP_det_derived")
    ap.add_argument("--min-side", type=int, default=640)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="process only first N records (debug)")
    ap.add_argument(
        "--keep-piids",
        default=None,
        help="comma-separated list of rig camera ids (piid, 1-4) to keep; default keeps all 4. "
        "1=bottom(45deg) 2=horizontal(90deg,fisheye) 3=tilt-top(135deg) 4=vertical(180deg,top-down)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    source = Path(args.source)
    output = Path(args.output)
    keep_piids = {int(x) for x in args.keep_piids.split(",")} if args.keep_piids else None

    print("Loading and filtering samples.json ...")
    records, excluded = load_det_records(source, keep_piids)
    print(f"  kept {len(records)} records, excluded {len(excluded)}")

    if args.limit:
        records = records[: args.limit]

    train_plants, val_plants = split_by_plant(records, args.val_ratio, args.seed)
    print(f"  {len(train_plants)} plants -> train, {len(val_plants)} plants -> val")

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    state_split_counts = Counter()
    corrupt = []

    for r in tqdm(records, desc="processing images"):
        split = "train" if r["plant_id"] in train_plants else "val"
        src = source / r["filepath"]
        stem = src.stem
        dst_img = output / "images" / split / f"{stem}.jpg"
        dst_lbl = output / "labels" / split / f"{stem}.txt"

        try:
            new_w, new_h = resize_and_save(src, dst_img, args.min_side, args.jpeg_quality)
        except Exception as e:
            corrupt.append((r["filepath"], str(e)))
            continue

        xc, yc, w, h = r["bbox"]
        class_id = STATE_TO_ID[r["state"]]
        dst_lbl.write_text(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        state_split_counts[(split, r["state"])] += 1
        manifest_rows.append(
            {
                "filename": f"{stem}.jpg",
                "plant_id": r["plant_id"],
                "image_set_id": r["image_set_id"],
                "piid": r["piid"],
                "pose_id": r["pose_id"],
                "bbch_stage": r["bbch_stage"],
                "state": r["state"],
                "split": split,
                "orig_w": r["orig_w"],
                "orig_h": r["orig_h"],
                "new_w": new_w,
                "new_h": new_h,
            }
        )

    manifest_path = output / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    excluded_path = output / "excluded.csv"
    with open(excluded_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "reason"])
        writer.writerows(excluded)
        writer.writerows((fp, f"corrupt_image: {err}") for fp, err in corrupt)

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(STATE_NAMES))
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "# TomatoMAP-derived growth-stage detection dataset (D_L, thesis section 3.2)\n"
        "# Single-class-per-image \"whole plant\" bbox + BBCH-derived growth stage.\n"
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "{}\n".format(output.resolve(), names_block)
    )

    report = {
        "total_det_samples": len(records) + len(excluded),
        "excluded_before_processing": len(excluded),
        "excluded_reasons": dict(Counter(reason for _, reason in excluded)),
        "corrupt_images": len(corrupt),
        "final_kept": len(manifest_rows),
        "train_images": sum(v for (s, _), v in state_split_counts.items() if s == "train"),
        "val_images": sum(v for (s, _), v in state_split_counts.items() if s == "val"),
        "train_plants": len(train_plants),
        "val_plants": len(val_plants),
        "state_distribution": {
            f"{split}/{state}": count for (split, state), count in sorted(state_split_counts.items())
        },
        "keep_piids": sorted(keep_piids) if keep_piids else "all",
        "min_side": args.min_side,
        "jpeg_quality": args.jpeg_quality,
        "seed": args.seed,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nDone.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
