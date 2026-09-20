"""Doc truc tiep cac file best.zip trong runs/kaggle (thuc chat la best.pt cua
ultralytics, chi doi ten .pt -> .zip khi tai ve tu Kaggle -- dinh dang zip cua
torch.save() ben trong khong doi) va in ra epoch/best_fitness/train_metrics/
train_args de biet ket qua cuoi cung cua moi lan train tren Kaggle."""

import torch
from pathlib import Path

CKPTS = [
    "/home/agi/thesis_code/runs/kaggle/classify/togro_0005_tomatomap_cls/best.zip",
    "/home/agi/thesis_code/runs/kaggle/classify/togro_tomatomap_cls/best.zip",
    "/home/agi/thesis_code/runs/kaggle/detect/togro_0005_labro/best.zip",
]

for p in CKPTS:
    path = Path(p)
    print("=" * 90)
    print(path)
    print("=" * 90)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("LOI khi load:", e)
        print()
        continue

    print("top-level keys:", list(ckpt.keys()))
    print("epoch:", ckpt.get("epoch"))
    print("best_fitness:", ckpt.get("best_fitness"))
    print("date:", ckpt.get("date"))
    print("version:", ckpt.get("version"))

    train_args = ckpt.get("train_args")
    if train_args:
        for k in ("model", "data", "epochs", "imgsz", "task", "name"):
            if k in train_args:
                print(f"train_args.{k}:", train_args[k])

    train_metrics = ckpt.get("train_metrics")
    print("train_metrics:", train_metrics)

    train_results = ckpt.get("train_results")
    if train_results is not None:
        print("train_results keys:", list(train_results.keys()) if isinstance(train_results, dict) else type(train_results))
        if isinstance(train_results, dict):
            for k, v in train_results.items():
                last_val = v[-1] if isinstance(v, list) and v else v
                print(f"  {k}: last={last_val}")

    print()
