"""Draw the whole-plant bbox + growth-stage label on a handful of samples
from the derived D_L dataset (built by build_tomatomap_detection_dataset.py)
and save them as an annotated image gallery, for a quick visual sanity check
of the dataset.

Usage:
    python3 make_sample_gallery.py \
        --dataset /mnt/data/tomatoMAP_det_derived \
        --output ./samples \
        --per-state 4
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

STATE_VN = {
    "vegetative_growth": "Sinh truong dinh duong",
    "flowering": "Ra hoa",
    "fruit_development": "Hinh thanh qua",
    "fruit_ripening": "Qua chuyen chin",
}
STATE_COLOR = {
    "vegetative_growth": (0, 200, 0),
    "flowering": (255, 0, 200),
    "fruit_development": (255, 120, 0),
    "fruit_ripening": (255, 40, 40),
}
STATE_ORDER = ["vegetative_growth", "flowering", "fruit_development", "fruit_ripening"]


def load_manifest(dataset: Path):
    rows = []
    with open(dataset / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_sample(dataset: Path, row: dict, font) -> Image.Image:
    split = row["split"]
    filename = row["filename"]
    img_path = dataset / "images" / split / filename
    lbl_path = dataset / "labels" / split / (Path(filename).stem + ".txt")

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    class_id, xc, yc, bw, bh = lbl_path.read_text().split()
    xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
    x0, y0, x1, y1 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2

    state = row["state"]
    color = STATE_COLOR[state]
    draw = ImageDraw.Draw(im)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

    lines = [
        f"{STATE_VN[state]} (BBCH {row['bbch_stage']})",
        f"plant #{row['plant_id']} [{split}]",
    ]
    pad = 6
    line_heights = []
    max_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w, line_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        max_w = max(max_w, line_w)
        line_heights.append(line_h)
    bar_h = sum(line_heights) + pad * (len(lines) + 1)
    draw.rectangle([0, 0, min(max_w + 2 * pad, w), bar_h], fill=color)
    y = pad
    for line, lh in zip(lines, line_heights):
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += lh + pad

    return im


def make_contact_sheet(images: list[Image.Image], cols: int, cell_w: int, cell_h: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    for i, im in enumerate(images):
        thumb = im.copy()
        thumb.thumbnail((cell_w - 8, cell_h - 8))
        x = (i % cols) * cell_w + (cell_w - thumb.width) // 2
        y = (i // cols) * cell_h + (cell_h - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="/mnt/data/tomatoMAP_det_derived")
    ap.add_argument("--output", default="./samples")
    ap.add_argument("--per-state", type=int, default=4, help="ignored if --grid is set")
    ap.add_argument("--grid", type=int, default=None, help="NxN grid of randomly mixed samples (overrides --per-state)")
    ap.add_argument("--cell-size", type=int, default=340, help="square cell size in px, used with --grid")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(dataset)
    rng = random.Random(args.seed)
    font = load_font(24)

    picked_images = []
    if args.grid:
        n = args.grid * args.grid
        # Round-robin across states so the grid stays diverse instead of being
        # dominated by whichever state has the most images (e.g. fruit_development).
        by_state = defaultdict(list)
        for r in rows:
            by_state[r["state"]].append(r)
        for pool in by_state.values():
            rng.shuffle(pool)
        cursors = {state: 0 for state in STATE_ORDER}
        picks = []
        while len(picks) < n:
            for state in STATE_ORDER:
                if len(picks) >= n:
                    break
                pool = by_state[state]
                c = cursors[state]
                if c < len(pool):
                    picks.append(pool[c])
                    cursors[state] += 1
        rng.shuffle(picks)
        for i, row in enumerate(picks):
            im = draw_sample(dataset, row, font)
            out_path = output / f"sample_{i+1}.jpg"
            im.save(out_path, "JPEG", quality=92)
            picked_images.append(im)
            print("saved", out_path)
        cols = args.grid
        cell_w = cell_h = args.cell_size
    else:
        by_state = defaultdict(list)
        for r in rows:
            by_state[r["state"]].append(r)
        for state in STATE_ORDER:
            candidates = by_state[state]
            picks = rng.sample(candidates, min(args.per_state, len(candidates)))
            for i, row in enumerate(picks):
                im = draw_sample(dataset, row, font)
                out_path = output / f"{state}_{i+1}.jpg"
                im.save(out_path, "JPEG", quality=92)
                picked_images.append(im)
                print("saved", out_path)
        cols = args.per_state
        cell_w, cell_h = 340, 440

    sheet = make_contact_sheet(picked_images, cols=cols, cell_w=cell_w, cell_h=cell_h)
    sheet_path = output / "gallery_overview.jpg"
    sheet.save(sheet_path, "JPEG", quality=90)
    print("saved", sheet_path)


if __name__ == "__main__":
    main()
