"""Build an NxN illustrative grid of Open Images V7 samples: each image is
center-cropped to a square (min(w,h) side) then resized to a fixed cell size.

Usage:
    python3 make_openimages_grid.py \
        --source ~/fiftyone/open-images-v7/train/data \
        --output ./openimages_grid.jpg \
        --grid 8 --cell 220
"""

import argparse
import random
import subprocess
from pathlib import Path

from PIL import Image


def sample_filenames(data_dir: Path, n: int, seed: int) -> list[str]:
    # Avoid materializing the full (hundreds-of-thousands entry) directory
    # listing in Python; let `shuf` do reservoir sampling and only return N names.
    out = subprocess.run(
        f"ls {data_dir} | shuf -n {n} --random-source=<(yes {seed})",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def center_crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="~/fiftyone/open-images-v7/train/data")
    ap.add_argument("--output", default="./openimages_grid.jpg")
    ap.add_argument("--grid", type=int, default=8, help="grid is NxN")
    ap.add_argument("--cell", type=int, default=220, help="cell size in px (square)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data_dir = Path(args.source).expanduser()
    n = args.grid * args.grid
    names = sample_filenames(data_dir, n, args.seed)
    print(f"sampled {len(names)} images")

    sheet = Image.new("RGB", (args.grid * args.cell, args.grid * args.cell), (20, 20, 20))
    for i, name in enumerate(names):
        with Image.open(data_dir / name) as im:
            im = im.convert("RGB")
            im = center_crop_square(im)
            im = im.resize((args.cell, args.cell), Image.Resampling.LANCZOS)
        x = (i % args.grid) * args.cell
        y = (i // args.grid) * args.cell
        sheet.paste(im, (x, y))

    out_path = Path(args.output)
    sheet.save(out_path, "JPEG", quality=92)
    print("saved", out_path, sheet.size)


if __name__ == "__main__":
    main()
