"""Danh gia chat luong (Acc/F1 cho TogroNet-Stage, mAP50/mAP50-95 cho
TogroNet-Fruit) cua tung file .tflite da export, tren dung tap du lieu goc.
Chay theo index (1 process/model) vi can dung trang thai patch_attention
khac nhau va tranh chong RAM giua cac lan chay."""

import sys
import numpy as np
from ultralytics import YOLO

CLS_DATA = "/mnt/data/tomatoMAP_cls_robot"
DET_DATA = "/mnt/data/laboro_tomato_yolo/data.yaml"

JOBS = [
    ("TogroNet-Stage", "FP32", "tflite_export/togronet_stage_float32.tflite", "classify", 320, False),
    ("TogroNet-Stage", "FP16", "tflite_export/togronet_stage_float16.tflite", "classify", 320, False),
    ("TogroNet-Stage", "INT8", "tflite_export/togronet_stage_int8.tflite",    "classify", 320, True),
    ("TogroNet-Fruit", "FP32", "tflite_export/togronet_fruit_float32.tflite", "detect",   640, False),
    ("TogroNet-Fruit", "FP16", "tflite_export/togronet_fruit_float16.tflite", "detect",   640, False),
    ("TogroNet-Fruit", "INT8", "tflite_export/togronet_fruit_int8.tflite",    "detect",   640, True),
]


def eval_classify(weights, imgsz, is_int8):
    model = YOLO(weights, task="classify")
    kwargs = dict(data=CLS_DATA, imgsz=imgsz, split="val", verbose=False, plots=True)
    if is_int8:
        kwargs["int8"] = True
    results = model.val(**kwargs)

    cm = results.confusion_matrix
    names = cm.names
    matrix = cm.matrix[: len(names), : len(names)]

    tp = matrix.diagonal()
    fp = matrix.sum(1) - tp
    fn = matrix.sum(0) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)
    macro_f1 = f1.mean()

    return dict(acc=results.top1, f1=macro_f1)


def eval_detect(weights, imgsz, is_int8):
    model = YOLO(weights, task="detect")
    kwargs = dict(data=DET_DATA, imgsz=imgsz, split="val", verbose=False, plots=False)
    if is_int8:
        kwargs["int8"] = True
    results = model.val(**kwargs)
    return dict(map50=results.box.map50, map5095=results.box.map)


if __name__ == "__main__":
    idx = int(sys.argv[1])
    model_name, precision, weights, task, imgsz, is_int8 = JOBS[idx]
    print(f"[{idx}] {model_name} [{precision}] -> {weights}")

    if task == "classify":
        m = eval_classify(weights, imgsz, is_int8)
        print(f"RESULT {model_name} {precision} acc={m['acc']:.4f} f1={m['f1']:.4f}")
    else:
        m = eval_detect(weights, imgsz, is_int8)
        print(f"RESULT {model_name} {precision} mAP50={m['map50']:.4f} mAP50-95={m['map5095']:.4f}")
