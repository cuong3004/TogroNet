#!/bin/bash

set -e

DATA="/home/agi/thesis_code/tomato_yolo_dataset"

EPOCHS=100
IMGSZ=640
BATCH=32
WORKERS=4
DEVICE=0

PROJECT="runs/tomato_finetune_ssl_pretrained"

# Checkpoint đã train COCO trước đó
YOLO26_PRETRAIN="/home/agi/thesis_code/yolo26n_ssl_init.pt"


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

train_model "yolo26n_coco_finetune" "yolo26n.yaml" "$YOLO26_PRETRAIN" "original"
