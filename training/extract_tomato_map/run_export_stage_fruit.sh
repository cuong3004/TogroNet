#!/bin/bash
set -uo pipefail
source /home/agi/miniconda3/etc/profile.d/conda.sh
conda activate ai
cd /home/agi/thesis_code

CLS_W="/home/agi/thesis_code/runs/classify/runs/tomato_finetune_coco_pretrained/togrow0005_cls/weights/best.pt"
CLS_DATA="/mnt/data/tomatoMAP_cls_robot"
DET_W="/home/agi/thesis_code/runs/detect/runs/tomato_finetune_coco_pretrained/togrow0005_backbone/weights/best.pt"
DET_DATA="/mnt/data/laboro_tomato_yolo/data.yaml"

run_one () {
    local label=$1 weights=$2 precision=$3 data=$4 imgsz=$5 fraction=$6
    echo "### $label [$precision]"
    if [ "$precision" = "int8" ]; then
        python3 extract_tomato_map/export_tflite_one.py --weights "$weights" --precision "$precision" --data "$data" --imgsz "$imgsz" --fraction "$fraction"
    else
        python3 extract_tomato_map/export_tflite_one.py --weights "$weights" --precision "$precision" --imgsz "$imgsz"
    fi
    echo "exit code: $?"
}

run_one "TogroNet-Stage" "$CLS_W" "float32" "$CLS_DATA" 320 1.0
run_one "TogroNet-Stage" "$CLS_W" "float16" "$CLS_DATA" 320 1.0
run_one "TogroNet-Stage" "$CLS_W" "int8"    "$CLS_DATA" 320 0.05
run_one "TogroNet-Fruit" "$DET_W" "float32" "$DET_DATA" 640 1.0
run_one "TogroNet-Fruit" "$DET_W" "float16" "$DET_DATA" 640 1.0
run_one "TogroNet-Fruit" "$DET_W" "int8"    "$DET_DATA" 640 1.0

echo "ALL_EXPORTS_DONE"
