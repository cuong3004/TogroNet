#!/bin/bash

set -e

DATA="/home/agi/thesis_code/COCO Dataset.v43i.yolo26/data.yaml"

echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"

python patch_attention.py restore

echo "========================================"
echo "TRAIN YOLO26 BASELINE"
echo "========================================"

#yolo detect train \
#    model=yolo26n.yaml \
#    data="$DATA" \
#    epochs=50 \
#    imgsz=640 \
#    batch=32 \
#    val=False \
#    workers=4
    # cache=disk

echo "========================================"
echo "REPLACE ATTENTION"
echo "========================================"

python patch_attention.py replace

echo "========================================"
echo "TRAIN TOGROW1"
echo "========================================"

yolo detect train \
    model=togrow1.yaml \
    data="$DATA" \
    epochs=50 \
    imgsz=640 \
    batch=32 \
    val=False \
    workers=4
    # cache=disk

echo "========================================"
echo "TRAIN TOGROW2"
echo "========================================"

yolo detect train \
    model=togrow2.yaml \
    data="$DATA" \
    epochs=50 \
    imgsz=640 \
    batch=32 \
    val=False \
    workers=4
    # cache=disk

echo "========================================"
echo "TRAIN TOGROW3"
echo "========================================"

yolo detect train \
    model=togrow3.yaml \
    data="$DATA" \
    epochs=50 \
    imgsz=640 \
    batch=32 \
    val=False \
    workers=4
    # cache=disk

echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"

python patch_attention.py restore

echo "========================================"
echo "ALL TRAINING COMPLETED"
echo "========================================"
