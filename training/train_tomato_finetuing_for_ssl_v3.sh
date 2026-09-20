#!/bin/bash

set -e

DATA="/home/agi/thesis_code/tomato_yolo_dataset"

EPOCHS=200
IMGSZ=640
BATCH=32
WORKERS=4
DEVICE=0

PROJECT="runs/tomato_finetune_coco_pretrained"

# Checkpoint đã train COCO trước đó
# TOGROW4_PRETRAIN="/home/agi/thesis_code/yolo_ssl_version_1.pt"
TOGROW5_PRETRAIN="/home/agi/thesis_code/yolo_ssl_version_2_2.pt"
# TOGROW6_PRETRAIN="/home/agi/thesis_code/yolo_ssl_version_3.pt"


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
        cache=disk \
        project="$PROJECT" \
        name="$MODEL_NAME" \
        exist_ok=True
}


echo "========================================"
echo "START TOMATO FINETUNING FROM COCO PRETRAINED"
echo "========================================"

# train_model "togrow0001" "/home/agi/thesis_code/yolo26_versions/togroth_m3_width_0200.yaml" "$TOGROW4_PRETRAIN" "replace"

train_model "togrow0005" "/home/agi/thesis_code/yolo26_versions/togroth_m3_width_0200.yaml" "$TOGROW5_PRETRAIN" "replace"

# train_model "togrow001" "/home/agi/thesis_code/yolo26_versions/togroth_m3_width_0200.yaml" "$TOGROW6_PRETRAIN" "replace"


echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"

python patch_attention.py restore

echo "========================================"
echo "ALL FINETUNING COMPLETED"
echo "RESULTS SAVED TO: $PROJECT"
echo "========================================"
