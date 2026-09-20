"""Parse the 3 fine-tuning logs from 2026-08-22 (log_finetuning_v{1,2,3}_1_22_8.txt)
-- the togrow0-6 / yolo26n / togrow0005 model sweep re-run on the new Laboro
Tomato YOLO dataset (3 ripeness classes: green/half_ripened/fully_ripened)
built by build_laboro_tomato_yolo_dataset.py, instead of the old tomatoMAP
growth-stage dataset -- and write a per-class CSV summary (same schema as
the earlier finetuning_results_summary.csv, for direct comparability).

Also dumps a JSON with the full per-epoch mAP series for each run, used by
plot_map_curves_22_8.py.

Usage:
    python3 report_finetuning_22_8.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FILES = {
    "v1": ROOT / "log_finetuning_v1_1_22_8.txt",
    "v2": ROOT / "log_finetuning_v2_1_22_8.txt",
    "v3": ROOT / "log_finetuning_v3_1_22_8.txt",
}
# v3_2 is a later, corrected retrain of the togrow0005/yolo_ssl_version_2_2.pt
# run (cfg fixed from togroth_m3_width_0200 -> togroth_m3_width_0225, matching
# the width the SSL checkpoint was actually pretrained on -- see report from
# 2026-08-22). Its run replaces the matching (name, pretrain) run from v3 below.
PATCH_FILES = {
    "v3": ROOT / "log_finetuning_v3_2_22_8.txt",
}
OUT_CSV = ROOT / "finetuning_results_summary_22_8.csv"
OUT_EPOCHS_JSON = ROOT / "finetuning_epochs_22_8.json"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Accepts both "N.Nit/s" and "N.Ns/it" (Rich switches units when an epoch is
# slow, e.g. the very first, warmup, epoch) -- missing this dropped ~half the
# epochs silently in an earlier version of this parser.
EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+([\d.]+G)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+):"
    r"\s*100%.*?(\d+)/(\d+)\s+[\d.]+(?:it/s|s/it)"
)
# Metric fields are normally plain decimals, but a very bad cold-start epoch
# (near-zero mAP) can print them in scientific notation (e.g. "2.07e-05") --
# without the exponent alternative those rows silently fail to match, which
# desyncs every later epoch's index-based P/R/mAP50/mAP50-95 alignment.
NUM = r"[\d.]+(?:e[-+]?\d+)?"
ALL_ROW_RE = re.compile(rf"^\s*all\s+(\d+)\s+(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$")
CLASS_ROW_RE = re.compile(rf"^\s{{5,}}(\S+)\s+(\d+)\s+(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$")
SUMMARY_RE = re.compile(r"summary \(fused\): (\d+) layers, ([\d,]+) parameters, \d+ gradients, ([\d.]+) GFLOPs")
TIME_RE = re.compile(r"(\d+) epochs completed in ([\d.]+) hours")


def load(path: Path) -> list[str]:
    with open(path, "r", errors="replace", newline=None) as f:
        return [ANSI_RE.sub("", line).rstrip("\n") for line in f]


def split_blocks(lines: list[str]) -> list[list[str]]:
    idxs = [i for i, l in enumerate(lines) if l.startswith("TRAIN ")]
    idxs.append(len(lines))
    return [lines[a:b] for a, b in zip(idxs, idxs[1:])]


def parse_block(block: list[str]) -> dict:
    name = block[0].replace("TRAIN ", "").strip()
    cfg = pretrain = attention = ""
    for l in block[:6]:
        if l.strip().startswith("CFG"):
            cfg = l.split(":", 1)[1].strip()
        elif l.strip().startswith("PRETRAIN"):
            pretrain = l.split(":", 1)[1].strip()
        elif l.strip().startswith("ATTENTION"):
            attention = l.split(":", 1)[1].strip()

    epoch_train: dict[int, dict] = {}
    for l in block:
        m = EPOCH_RE.match(l)
        if m:
            e, total, mem, box, cls, dfl, inst, size, cur, tot = m.groups()
            if cur == tot:
                epoch_train[int(e)] = dict(
                    total_epochs=int(total), box_loss=float(box), cls_loss=float(cls), dfl_loss=float(dfl)
                )

    all_rows = []
    for l in block:
        m = ALL_ROW_RE.match(l)
        if m:
            images, inst, p, r, map50, map50_95 = m.groups()
            all_rows.append(dict(images=int(images), instances=int(inst), P=float(p), R=float(r),
                                  mAP50=float(map50), mAP50_95=float(map50_95)))

    n_epochs = len(epoch_train)
    per_epoch_val = all_rows[:n_epochs]
    final_val = all_rows[n_epochs] if len(all_rows) > n_epochs else (all_rows[-1] if all_rows else None)

    epoch_series = []
    for e in sorted(epoch_train.keys()):
        row = dict(epoch=e, **epoch_train[e])
        idx = e - 1
        if idx < len(per_epoch_val):
            row.update(per_epoch_val[idx])
        epoch_series.append(row)

    per_class = []
    if final_val is not None:
        val_idx = next((i for i, l in enumerate(block) if "Validating" in l and "best.pt" in l), None)
        if val_idx is not None:
            all_i = next((i for i in range(val_idx, len(block)) if ALL_ROW_RE.match(block[i])), None)
            if all_i is not None:
                for l in block[all_i + 1: all_i + 20]:
                    m = CLASS_ROW_RE.match(l)
                    if m:
                        cname, images, inst, p, r, map50, map50_95 = m.groups()
                        per_class.append(dict(cls=cname, images=int(images), instances=int(inst),
                                               P=float(p), R=float(r), mAP50=float(map50), mAP50_95=float(map50_95)))
                    elif l.strip() == "" or "Speed:" in l:
                        break

    summary = None
    m_time = None
    for l in block:
        m = SUMMARY_RE.search(l)
        if m:
            summary = dict(layers=int(m.group(1)), params=m.group(2).replace(",", ""), gflops=float(m.group(3)))
        m2 = TIME_RE.search(l)
        if m2:
            m_time = dict(epochs=int(m2.group(1)), hours=float(m2.group(2)))

    return dict(name=name, cfg=cfg, pretrain=pretrain, attention=attention,
                epoch_series=epoch_series, final_val=final_val, per_class=per_class,
                summary=summary, time=m_time)


def main():
    csv_rows = []
    epochs_dump = {}
    all_runs_by_vkey = {}

    for vkey, path in LOG_FILES.items():
        blocks = split_blocks(load(path))
        runs = [parse_block(b) for b in blocks]
        for r in runs:
            r["source_file"] = path.name
        all_runs_by_vkey[vkey] = runs

    # Apply patches: a later retrain of one specific (name, pretrain) run
    # replaces the original entry in-place, keeping its position in the list.
    for vkey, path in PATCH_FILES.items():
        patch_blocks = split_blocks(load(path))
        patch_runs = [parse_block(b) for b in patch_blocks]
        for pr in patch_runs:
            pr["source_file"] = path.name
            key = (pr["name"], pr["pretrain"])
            runs = all_runs_by_vkey[vkey]
            idx = next((i for i, r in enumerate(runs) if (r["name"], r["pretrain"]) == key), None)
            if idx is not None:
                print(f"[{vkey}] patching '{pr['name']}' (pretrain={pr['pretrain']}) "
                      f"from {path.name}: cfg {runs[idx]['cfg']} -> {pr['cfg']}")
                runs[idx] = pr
            else:
                runs.append(pr)

    for vkey, runs in all_runs_by_vkey.items():
        epochs_dump[vkey] = {"runs": []}

        for r in runs:
            params = r["summary"]["params"] if r["summary"] else ""
            gflops = r["summary"]["gflops"] if r["summary"] else ""
            hours = f"{r['time']['hours']:.3f}" if r["time"] else ""

            if r["final_val"]:
                csv_rows.append({
                    "log_file": r["source_file"], "run_name": r["name"], "cfg": r["cfg"], "pretrain": r["pretrain"],
                    "attention": r["attention"], "params": params, "gflops": gflops, "train_time_h": hours,
                    "precision": r["final_val"]["P"], "recall": r["final_val"]["R"],
                    "mAP50": r["final_val"]["mAP50"], "mAP50_95": r["final_val"]["mAP50_95"],
                })

            epochs_dump[vkey]["runs"].append({
                "name": r["name"], "cfg": r["cfg"],
                "epochs": [e["epoch"] for e in r["epoch_series"]],
                "mAP50_95": [e.get("mAP50_95") for e in r["epoch_series"]],
                "mAP50": [e.get("mAP50") for e in r["epoch_series"]],
            })
            print(f"[{vkey}] {r['name']} ({r['source_file']}): {len(r['epoch_series'])} epochs logged, "
                  f"final mAP50-95={r['final_val']['mAP50_95'] if r['final_val'] else '?'}")

    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwrote {OUT_CSV} ({len(csv_rows)} rows)")

    OUT_EPOCHS_JSON.write_text(json.dumps(epochs_dump, indent=2))
    print(f"wrote {OUT_EPOCHS_JSON}")


if __name__ == "__main__":
    main()
