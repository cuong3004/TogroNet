#!/bin/bash
set -uo pipefail

source /home/agi/miniconda3/etc/profile.d/conda.sh
conda activate ai
cd /home/agi/thesis_code

CLS_W="/home/agi/thesis_code/runs/classify/runs/tomato_finetune_coco_pretrained/togrow0005/weights/last.pt"
CLS_DATA="/mnt/data/tomatoMAP_cls_robot"
DET_W="/home/agi/thesis_code/runs/detect/runs/tomato_finetune_coco_pretrained/togrow4_coco_finetune/weights/last.pt"
DET_DATA="/mnt/data/laboro_tomato_yolo/data.yaml"

run_one () {
    local label=$1
    local weights=$2
    local precision=$3
    local data=$4

    echo "########################################"
    echo "### $label [$precision]"
    echo "########################################"

    if [ "$precision" = "int8" ]; then
        python3 extract_tomato_map/export_tflite_one.py --weights "$weights" --precision "$precision" --data "$data"
    else
        python3 extract_tomato_map/export_tflite_one.py --weights "$weights" --precision "$precision"
    fi

    echo "exit code: $?"
}

run_one "classify-togrow0005" "$CLS_W" "float32" "$CLS_DATA"
run_one "classify-togrow0005" "$CLS_W" "float16" "$CLS_DATA"
run_one "classify-togrow0005" "$CLS_W" "int8"    "$CLS_DATA"
run_one "detect-togrow4"      "$DET_W" "float32" "$DET_DATA"
run_one "detect-togrow4"      "$DET_W" "float16" "$DET_DATA"
run_one "detect-togrow4"      "$DET_W" "int8"    "$DET_DATA"

echo "ALL EXPORTS DONE"
