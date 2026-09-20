import sys
import numpy as np
from ultralytics import YOLO

DATA = "/mnt/data/tomatoMAP_cls_robot"

MODELS = [
    ("TogroNet-Stage", "/home/agi/thesis_code/extract_tomato_map/togro_tomatomap_cls_best.pt"),
    ("Mô hình đề xuất", "/home/agi/thesis_code/extract_tomato_map/togrow0005_cls_local_best.pt"),
]

idx = int(sys.argv[1])
label, weights = MODELS[idx]

model = YOLO(weights)
results = model.val(data=DATA, imgsz=320, split="val", verbose=False, plots=True)

cm = results.confusion_matrix
names = list(cm.names.values())

import os
os.makedirs("cm_cache_stage", exist_ok=True)
np.savez(f"cm_cache_stage/{idx:02d}.npz", matrix=cm.matrix, names=np.array(names), label=label, acc=results.top1)
print(f"Saved cm_cache_stage/{idx:02d}.npz  acc={results.top1:.4f}")
