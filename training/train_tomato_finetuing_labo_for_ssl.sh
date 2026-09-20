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
YOLO26_PRETRAIN="/home/agi/thesis_code/runs/detect/train-3/weights/last.pt"
TOGROW0_PRETRAIN="/home/agi/thesis_code/runs/detect/train-3/weights/last.pt"
TOGROW1_PRETRAIN="/home/agi/thesis_code/runs/detect/train-5/weights/last.pt"
TOGROW2_PRETRAIN="//home/agi/thesis_code/runs/detect/train-6/weights/last.pt"
TOGROW3_PRETRAIN="/home/agi/thesis_code/runs/detect/train-7/weights/last.pt"


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

train_model "yolo26n_coco_finetune" "yolo26n.yaml" "$YOLO26_PRETRAIN" "original"

train_model "togrow0_coco_finetune" "togrow0.yaml" "$TOGROW0_PRETRAIN" "replace"

train_model "togrow1_coco_finetune" "togrow1.yaml" "$TOGROW1_PRETRAIN" "replace"

train_model "togrow2_coco_finetune" "togrow2.yaml" "$TOGROW2_PRETRAIN" "replace"

train_model "togrow3_coco_finetune" "togrow3.yaml" "$TOGROW3_PRETRAIN" "replace"


echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"

python patch_attention.py restore

echo "========================================"
echo "ALL FINETUNING COMPLETED"
echo "RESULTS SAVED TO: $PROJECT"
echo "========================================"
