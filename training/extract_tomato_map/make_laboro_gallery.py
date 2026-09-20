"""Draw all fruit bboxes on a handful of sample images from the converted
Laboro Tomato YOLO dataset (built by build_laboro_tomato_yolo_dataset.py)
and save a 3x3 contact-sheet gallery, for a visual sanity check.

Usage:
    python3 make_laboro_gallery.py --dataset /mnt/data/laboro_tomato_yolo --grid 3
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

CLASS_COLOR = {
    "green": (60, 180, 75),
    "half_ripened": (255, 165, 0),
    "fully_ripened": (220, 20, 60),
}
CLASS_ORDER = ["green", "half_ripened", "fully_ripened"]


def box_aware_square_crop(im: Image.Image, boxes: list[dict]) -> tuple[Image.Image, float, float]:
    """Crop the largest square that fits in `im`, positioned to contain the
    union of all `boxes` as fully as possible (instead of a blind center
    crop, which truncates any box lying in the trimmed top/bottom strip of
    these portrait photos). Returns (cropped_image, x_offset, y_offset)."""
    w, h = im.size
    side = min(w, h)

    xs0 = [b["bbox_x"] for b in boxes]
    ys0 = [b["bbox_y"] for b in boxes]
    xs1 = [b["bbox_x"] + b["bbox_w"] for b in boxes]
    ys1 = [b["bbox_y"] + b["bbox_h"] for b in boxes]
    union_cx = (min(xs0) + max(xs1)) / 2
    union_cy = (min(ys0) + max(ys1)) / 2

    left = min(max(union_cx - side / 2, 0), w - side)
    top = min(max(union_cy - side / 2, 0), h - side)
    return im.crop((left, top, left + side, top + side)), left, top


def draw_sample(dataset: Path, split: str, filename: str, boxes: list[dict], cell: int) -> Image.Image:
    img_path = dataset / "images" / split / filename
    # exif_transpose is required: ~38% of the raw Laboro photos carry a
    # non-1 EXIF Orientation tag (portrait shots stored as rotated landscape
    # buffers). The manifest's bbox_x/y/w/h are in the *displayed* (rotated)
    # frame -- same convention Ultralytics' cv2.imread-based loader uses at
    # train time -- so skipping this step draws boxes on the wrong frame.
    im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    crop_side = min(im.size)

    im, off_x, off_y = box_aware_square_crop(im, boxes)
    im = im.resize((cell, cell), Image.LANCZOS)
    scale = cell / crop_side
    draw = ImageDraw.Draw(im)

    for b in boxes:
        x = (b["bbox_x"] - off_x) * scale
        y = (b["bbox_y"] - off_y) * scale
        bw, bh = b["bbox_w"] * scale, b["bbox_h"] * scale
        color = CLASS_COLOR[b["merged_class"]]
        draw.rectangle([x, y, x + bw, y + bh], outline=color, width=4)

    return im


def make_contact_sheet(images: list[Image.Image], cols: int, cell: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (25, 25, 25))
    for i, im in enumerate(images):
        x = (i % cols) * cell
        y = (i // cols) * cell
        sheet.paste(im, (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="/mnt/data/laboro_tomato_yolo")
    ap.add_argument("--output", default="./laboro_gallery_3x3.jpg")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--cell-size", type=int, default=520)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    rng = random.Random(args.seed)

    with open(dataset / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
        for r in rows:
            r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"] = (
                float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])
            )

    boxes_by_image = defaultdict(list)
    for r in rows:
        boxes_by_image[(r["split"], r["filename"])].append(r)

    def fits_in_square_crop(boxes: list[dict]) -> bool:
        # True if a single square crop of side min(img_w, img_h) can contain
        # every box for this image at once (so box_aware_square_crop below
        # never has to clip one) -- these are portrait photos, so a cluster
        # of fruit spread far down the vine can span more than the width.
        img_w, img_h = float(boxes[0]["image_width"]), float(boxes[0]["image_height"])
        side = min(img_w, img_h)
        x0 = min(b["bbox_x"] for b in boxes)
        y0 = min(b["bbox_y"] for b in boxes)
        x1 = max(b["bbox_x"] + b["bbox_w"] for b in boxes)
        y1 = max(b["bbox_y"] + b["bbox_h"] for b in boxes)
        return (x1 - x0) <= side and (y1 - y0) <= side

    # Prefer images where (a) every box fits in one square crop with no
    # clipping and (b) at least 2 of the 3 classes appear (more informative
    # for a dataset-overview gallery), relaxing (b) then (a) if too few match.
    fits = [k for k, v in boxes_by_image.items() if fits_in_square_crop(v)]
    mixed_fits = [k for k in fits if len({b["merged_class"] for b in boxes_by_image[k]}) >= 2]
    if len(mixed_fits) >= args.grid * args.grid:
        pool = mixed_fits
    elif len(fits) >= args.grid * args.grid:
        pool = fits
    else:
        pool = list(boxes_by_image.keys())
    rng.shuffle(pool)
    picks = pool[: args.grid * args.grid]

    images = []
    for split, filename in picks:
        im = draw_sample(dataset, split, filename, boxes_by_image[(split, filename)], args.cell_size)
        images.append(im)

    sheet = make_contact_sheet(images, cols=args.grid, cell=args.cell_size)
    sheet.save(args.output, "JPEG", quality=90)
    print(f"saved {args.output} ({len(images)} images, {args.grid}x{args.grid})")
    for split, filename in picks:
        classes = defaultdict(int)
        for b in boxes_by_image[(split, filename)]:
            classes[b["merged_class"]] += 1
        print(f"  {split}/{filename}: {dict(classes)}")


if __name__ == "__main__":
    main()
