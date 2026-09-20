"""Sample gallery for the classification-format tomatoMAP dataset
(/mnt/data/tomatoMAP_cls_robot, train/<class>/*.jpg + val/<class>/*.jpg --
same images/classes as tomatoMAP_det_derived_robot, just re-sorted into
class folders for classification training, no bbox to draw).

Mirrors make_sample_gallery.py's per-state grid layout, minus the bounding
box (there isn't one here) and any text overlay -- plain images, center-
cropped to fill each cell edge-to-edge (no letterbox padding).

Usage:
    python3 make_cls_gallery.py \
        --dataset /mnt/data/tomatoMAP_cls_robot \
        --output ./samples_cls_4x3 \
        --per-state 3
"""

import argparse
import random
from pathlib import Path

from PIL import Image

STATE_ORDER = ["vegetative_growth", "flowering", "fruit_development", "fruit_ripening"]


def load_sample(img_path: Path) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def center_crop(im: Image.Image, aspect: float) -> Image.Image:
    """Crop to the given width/height aspect ratio, centered, so the grid
    cell is filled edge-to-edge with no letterbox padding."""
    w, h = im.size
    target_w = min(w, round(h * aspect))
    target_h = min(h, round(w / aspect))
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return im.crop((left, top, left + target_w, top + target_h))


def make_contact_sheet(images: list[Image.Image], cols: int, cell_w: int, cell_h: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h))
    for i, im in enumerate(images):
        thumb = center_crop(im, cell_w / cell_h).resize((cell_w, cell_h), Image.LANCZOS)
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="/mnt/data/tomatoMAP_cls_robot")
    ap.add_argument("--output", default="./samples_cls_4x3")
    ap.add_argument("--per-state", type=int, default=3)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    picked_images = []
    for state in STATE_ORDER:
        class_dir = dataset / args.split / state
        candidates = sorted(class_dir.glob("*.jpg"))
        picks = rng.sample(candidates, min(args.per_state, len(candidates)))
        for i, img_path in enumerate(picks):
            im = load_sample(img_path)
            out_path = output / f"{state}_{i+1}.jpg"
            im.save(out_path, "JPEG", quality=92)
            picked_images.append(im)
            print("saved", out_path)

    sheet = make_contact_sheet(picked_images, cols=args.per_state, cell_w=340, cell_h=440)
    sheet_path = output / "gallery_overview.jpg"
    sheet.save(sheet_path, "JPEG", quality=90)
    print("saved", sheet_path)


if __name__ == "__main__":
    main()
