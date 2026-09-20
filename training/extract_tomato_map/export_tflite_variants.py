"""Export togrow0005 (classify) va togrow4_coco_finetune (detect) sang TFLite
float32 / float16 / int8. int8 can dung calibration data dung dataset da train
tuong ung (thu muc classify hoac data.yaml detect)."""

from ultralytics import YOLO

JOBS = [
    dict(
        name="classify-togrow0005",
        weights="/home/agi/thesis_code/runs/classify/runs/tomato_finetune_coco_pretrained/togrow0005/weights/last.pt",
        data="/mnt/data/tomatoMAP_cls_robot",
        imgsz=640,
    ),
    dict(
        name="detect-togrow4",
        weights="/home/agi/thesis_code/runs/detect/runs/tomato_finetune_coco_pretrained/togrow4_coco_finetune/weights/last.pt",
        data="/mnt/data/laboro_tomato_yolo/data.yaml",
        imgsz=640,
    ),
]

for job in JOBS:
    print("=" * 80)
    print("JOB:", job["name"], "->", job["weights"])
    print("=" * 80)

    model = YOLO(job["weights"])

    print(f"--- {job['name']}: float32 ---")
    model.export(format="tflite", imgsz=job["imgsz"])

    print(f"--- {job['name']}: float16 ---")
    model.export(format="tflite", imgsz=job["imgsz"], half=True)

    print(f"--- {job['name']}: int8 ---")
    model.export(format="tflite", imgsz=job["imgsz"], int8=True, data=job["data"])

print("\nALL EXPORTS DONE")
