"""In-place fix for an existing YOLO-format TomatoMAP-derived dataset (built by
build_tomatomap_detection_dataset.py): merge the "seedling" (BBCH 13-19) and
"vegetative" (BBCH 20-29) classes into one "vegetative_growth" class.

Rationale: standalone "vegetative" was only 18/101 plants' worth of images
(~1.6% of train), which a YOLO detector trained on it got ~0 recall on.
Merging it into "vegetative_growth" (BBCH 13-29) draws from 60/101 plants and
brings the smallest class within ~2.8x of the largest instead of ~28x.

Only rewrites labels/*.txt (class id remap), manifest.csv, data.yaml, and
build_report.json -- images are untouched (bbox geometry doesn't change).

Usage:
    python3 merge_vegetative_classes.py --dataset /mnt/data/tomatoMAP_det_derived_robot
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

OLD_NAMES = ["seedling", "vegetative", "flowering", "fruit_development", "fruit_ripening"]
NEW_NAMES = ["vegetative_growth", "flowering", "fruit_development", "fruit_ripening"]
STATE_REMAP = {
    "seedling": "vegetative_growth",
    "vegetative": "vegetative_growth",
    "flowering": "flowering",
    "fruit_development": "fruit_development",
    "fruit_ripening": "fruit_ripening",
}
NEW_ID = {name: i for i, name in enumerate(NEW_NAMES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    root = Path(args.dataset)

    manifest_path = root / "manifest.csv"
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    state_split_counts = Counter()
    for row in rows:
        old_state = row["state"]
        new_state = STATE_REMAP[old_state]
        row["state"] = new_state
        state_split_counts[(row["split"], new_state)] += 1

        stem = Path(row["filename"]).stem
        lbl_path = root / "labels" / row["split"] / f"{stem}.txt"
        class_id, xc, yc, w, h = lbl_path.read_text().split()
        lbl_path.write_text(f"{NEW_ID[new_state]} {xc} {yc} {w} {h}\n")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rewrote {len(rows)} label files + manifest.csv")

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(NEW_NAMES))
    yaml_path = root / "data.yaml"
    old_yaml = yaml_path.read_text()
    header_lines = [l for l in old_yaml.splitlines() if l.startswith("#")]
    yaml_path.write_text(
        "\n".join(header_lines)
        + "\n"
        + "# Growth-stage classes merged: seedling+vegetative -> vegetative_growth\n"
        + "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "{}\n".format(root.resolve(), names_block)
    )
    print("rewrote data.yaml")

    report_path = root / "build_report.json"
    report = json.loads(report_path.read_text())
    report["train_images"] = sum(v for (s, _), v in state_split_counts.items() if s == "train")
    report["val_images"] = sum(v for (s, _), v in state_split_counts.items() if s == "val")
    report["state_distribution"] = {
        f"{split}/{state}": count for (split, state), count in sorted(state_split_counts.items())
    }
    report["classes_merged"] = "seedling+vegetative -> vegetative_growth (BBCH 13-29)"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("rewrote build_report.json")
    print(json.dumps(report["state_distribution"], indent=2))


if __name__ == "__main__":
    main()
