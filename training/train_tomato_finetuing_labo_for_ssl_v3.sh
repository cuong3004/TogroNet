#!/bin/bash

set -e

DATA="/mnt/data/laboro_tomato_yolo/data.yaml"

EPOCHS=100
IMGSZ=640
BATCH=32
WORKERS=4
DEVICE=0

PROJECT="runs/tomato_finetune_coco_pretrained"

# Checkpoint đã train COCO trước đó
TOGROW4_PRETRAIN="/home/agi/thesis_code/runs/detect/train-14/weights/last.pt"
TOGROW5_PRETRAIN="/home/agi/thesis_code/extract_tomato_map/last_ssl_v4.pt"
TOGROW6_PRETRAIN="/home/agi/thesis_code/extract_tomato_map/last_yolo_ssl_e88.pt"



train_model () {
    MODEL_NAME=$1
    MODEL_CFG=$2
    PRETRAIN=$3
    ATTENTION_MODE=$4

    echo "========================================"
    echo "TRAIN $MODEL_NAME"
    echo "CFG       : $MODEL_CFG"
    echo "PRETRAIN  : $PRETRAIN"
    echo "ATTENTION : $ATTENTION_MODE"
    echo "========================================"

    if [ "$ATTENTION_MODE" = "original" ]; then
        python patch_attention.py restore
    else
        python patch_attention.py replace
    fi

    yolo detect train \
        model="$MODEL_CFG" \
        pretrained="$PRETRAIN" \
        data="$DATA" \
        epochs="$EPOCHS" \
        imgsz="$IMGSZ" \
        batch="$BATCH" \
        workers="$WORKERS" \
        device="$DEVICE" \
        val=True \
        project="$PROJECT" \
        name="$MODEL_NAME" \
        exist_ok=True
}


echo "========================================"
echo "START TOMATO FINETUNING FROM COCO PRETRAINED"
echo "========================================"

train_model "yolotogrow0005_88" "/home/agi/thesis_code/yolo26.yaml" "$TOGROW6_PRETRAIN" "original"
# train_model "togrow0005_9" "/home/agi/thesis_code/yolo26_versions/togroth_m2_width_0225.yaml" "$TOGROW5_PRETRAIN" "replace"

# train_model "togrow_coco" "/home/agi/thesis_code/yolo26_versions/togroth_m2_width_0225.yaml" "$TOGROW4_PRETRAIN" "replace"

# train_model "togrow0005" "/home/agi/thesis_code/yolo26_versions/togroth_m3_width_0200.yaml" "$TOGROW5_PRETRAIN" "replace"


echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"

python patch_attention.py restore

echo "========================================"
echo "ALL FINETUNING COMPLETED"
echo "RESULTS SAVED TO: $PROJECT"
echo "========================================"
