"""Doc 3 file results.csv va in ra dong cuoi cung (epoch cuoi) cua moi file,
ghep ten cot voi gia tri de kiem tra doi chieu duoc, khong doan/go tay."""

import csv
from pathlib import Path

FILES = [
    "/home/agi/thesis_code/togro0005_cls.csv",
    "/home/agi/thesis_code/togro_coco_cls.csv",
    "/home/agi/thesis_code/togro0005.csv",
]

for f in FILES:
    path = Path(f)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    n = len(rows)
    last = rows[-1]

    print("=" * 80)
    print(f"{path.name}  ({n} epoch, dong cuoi = epoch {last['epoch']})")
    print("=" * 80)
    for k, v in last.items():
        print(f"  {k:30s} = {v}")
    print()
