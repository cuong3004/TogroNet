"""Chay validation cho 1 model detect (theo index) tren tap laboro_tomato_yolo,
luu confusion matrix ra .npz de script ve hinh doc lai sau. Chay rieng tung
process vi cac model can trang thai patch_attention khac nhau (class Attention
da import vao bo nho thi khong doi lai duoc giua chung, phai tach process)."""

import sys
import numpy as np
from ultralytics import YOLO

DATA = "/mnt/data/laboro_tomato_yolo/data.yaml"
RUNS_DIR = "/home/agi/thesis_code/runs/detect/runs/tomato_finetune_coco_pretrained"
OUT_DIR = "/home/agi/thesis_code/extract_tomato_map/cm_cache"

# (label ngan de ve hinh, ten thu muc run, attention mode)
MODELS = [
    ("yolo26n\n(COCO)",              "yolo26n_coco_finetune", "original"),
    ("togrow0",                      "togrow0_coco_finetune", "replace"),
    ("togrow1",                      "togrow1_coco_finetune", "replace"),
    ("togrow2",                      "togrow2_coco_finetune", "replace"),
    ("togrow3",                      "togrow3_coco_finetune", "replace"),
    ("togrow4\n(w=0.225)",           "togrow4_coco_finetune", "replace"),
    ("togrow5\n(w=0.200)",           "togrow5_coco_finetune", "replace"),
    ("togrow6\n(w=0.175)",           "togrow6_coco_finetune", "replace"),
    ("togrow0005\n(SSL)",            "togrow0005", "replace"),
    ("togrow0005\n(SSL, backbone)",  "togrow0005_backbone", "replace"),
]

if __name__ == "__main__":
    idx = int(sys.argv[1])
    label, run_dir, attn = MODELS[idx]
    weights = f"{RUNS_DIR}/{run_dir}/weights/best.pt"

    print(f"[{idx}] {run_dir} (attention={attn}) -> {weights}")

    model = YOLO(weights)
    results = model.val(data=DATA, imgsz=640, split="val", verbose=False, plots=True)

    cm = results.confusion_matrix
    names = list(cm.names.values()) + ["background"]

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        f"{OUT_DIR}/{idx:02d}_{run_dir}.npz",
        matrix=cm.matrix,
        names=np.array(names),
        label=label,
        run_dir=run_dir,
        map50_95=results.box.map,
    )
    print(f"Saved cm_cache/{idx:02d}_{run_dir}.npz  (mAP50-95={results.box.map:.4f})")
