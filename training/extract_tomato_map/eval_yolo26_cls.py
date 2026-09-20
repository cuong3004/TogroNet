import sys
import numpy as np
from ultralytics import YOLO

DATA = "/mnt/data/tomatoMAP_cls_robot"

MODELS = [
    ("yolo26_cls_finetune (COCO)",
     "/home/agi/thesis_code/runs/classify/runs/tomato_finetune_coco_pretrained/yolo26_cls_finetune/weights/best.pt"),
    ("yolo26_cls_ssl_finetune (SSL)",
     "/home/agi/thesis_code/runs/classify/runs/tomato_finetune_coco_pretrained/yolo26_cls_ssl_finetune/weights/best.pt"),
]

idx = int(sys.argv[1])
label, weights = MODELS[idx]

model = YOLO(weights)
model.fuse()  # fold Conv+BN truoc khi do params/GFLOPs, dung quy uoc "(fused)"
info = model.info(detailed=False, verbose=True, imgsz=320)
n_params = info[1]
gflops = info[3]

results = model.val(data=DATA, imgsz=320, split="val", verbose=False, plots=True)

cm = results.confusion_matrix
names = cm.names
matrix = cm.matrix[: len(names), : len(names)]

tp = matrix.diagonal()
fp = matrix.sum(1) - tp
fn = matrix.sum(0) - tp
support = matrix.sum(0)

precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
f1 = np.divide(2 * precision * recall, precision + recall,
                out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)

w = support / support.sum()
weighted_p = (precision * w).sum()
weighted_r = (recall * w).sum()
weighted_f1 = (f1 * w).sum()

print(
    f"RESULT {label} acc={results.top1*100:.2f}% f1={weighted_f1*100:.2f}% "
    f"p={weighted_p*100:.2f}% r={weighted_r*100:.2f}% "
    f"params={n_params/1e6:.3f}M gflops={gflops:.3f}"
)
