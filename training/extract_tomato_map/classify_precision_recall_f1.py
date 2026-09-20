"""Chay validation cho 2 checkpoint classify tren cung 1 tap du lieu, roi tinh
precision/recall/F1 tung lop + macro/weighted average tu confusion matrix cua
ultralytics (ultralytics classify chi bao cao accuracy_top1/top5 mac dinh,
khong co san precision/recall/F1)."""

import sys
import numpy as np
from ultralytics import YOLO

DATA = "/mnt/data/tomatoMAP_cls_robot"

MODELS = [
    ("local togrow0005_cls (COCO+SSL, backbone+neck+head)",
     "/home/agi/thesis_code/extract_tomato_map/togrow0005_cls_local_best.pt", 320),
    ("kaggle togro_tomatomap_cls (COCO-only, no SSL)",
     "/home/agi/thesis_code/extract_tomato_map/togro_tomatomap_cls_best.pt", 320),
]

idx = int(sys.argv[1])
label, weights, imgsz = MODELS[idx]
if True:
    print("=" * 90)
    print(label, "->", weights, f"(imgsz={imgsz})")
    print("=" * 90)

    model = YOLO(weights)
    results = model.val(data=DATA, imgsz=imgsz, split="val", verbose=False)

    print(f"\n[ultralytics tu bao cao] top1={results.top1:.4f}  top5={results.top5:.4f}")

    cm = results.confusion_matrix
    names = cm.names
    # cm.matrix co the la (nc+1)x(nc+1) (hang/cot cuoi la slot noi bo, toan
    # so 0 voi task classify) -- cat dung ve nc x nc de khong lam sai macro avg
    matrix = cm.matrix[: len(names), : len(names)]  # matrix[pred][true]

    tp = matrix.diagonal()
    fp = matrix.sum(1) - tp          # predicted as class i, actually khac
    fn = matrix.sum(0) - tp          # thuc te la class i, du doan sai
    support = matrix.sum(0)          # so luong mau thuc te moi lop

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)

    accuracy = tp.sum() / matrix.sum()

    print(f"\n{'class':22s} {'support':>8s} {'precision':>10s} {'recall':>10s} {'f1':>10s}")
    for i in range(len(names)):
        print(f"{names[i]:22s} {int(support[i]):8d} {precision[i]:10.4f} {recall[i]:10.4f} {f1[i]:10.4f}")

    macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()
    w = support / support.sum()
    weighted_p = (precision * w).sum()
    weighted_r = (recall * w).sum()
    weighted_f1 = (f1 * w).sum()

    print(f"\n{'accuracy':22s} {'':8s} {'':10s} {'':10s} {accuracy:10.4f}")
    print(f"{'macro avg':22s} {int(support.sum()):8d} {macro_p:10.4f} {macro_r:10.4f} {macro_f1:10.4f}")
    print(f"{'weighted avg':22s} {int(support.sum()):8d} {weighted_p:10.4f} {weighted_r:10.4f} {weighted_f1:10.4f}")
    print()
