#!/bin/bash

set -e

DATA="/home/agi/thesis_code/COCO Dataset.v43i.yolo26/data.yaml"

echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"

python patch_attention.py restore

echo "========================================"
echo "REPLACE ATTENTION"
echo "========================================"

python patch_attention.py replace

echo "========================================"
echo "TRAIN TOGROW1"
echo "========================================"

# yolo detect train \
#     model=/home/agi/thesis_code/yolo26_versions/togroth_m2_width_0225.yaml \
#     data="$DATA" \
#     epochs=50 \
#     imgsz=640 \
#     batch=32 \
#     val=False \
#     workers=4
#     # cache=disk

echo "========================================"
echo "TRAIN TOGROW2"
echo "========================================"

#yolo detect train \
#    model=/home/agi/thesis_code/yolo26_versions/togroth_m3_width_0200.yaml \
#    data="$DATA" \
#    epochs=50 \
#    imgsz=640 \
#    batch=32 \
#    val=False \
#    workers=4
    # cache=disk

echo "========================================"
echo "TRAIN TOGROW3"
echo "========================================"

#yolo detect train \
#    model=/home/agi/thesis_code/yolo26_versions/togroth_m4_width_0175.yaml \
#    data="$DATA" \
#    epochs=50 \
#    imgsz=640 \
#    batch=32 \
#    val=False \
#    workers=4
    # cache=disk

echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"


#yolo detect train \
#    model=/home/agi/thesis_code/yolo26_versions/togroth_m4_width_0175.yaml \
#    data="$DATA" \
#    epochs=50 \
#    imgsz=640 \
#    batch=32 \
#    val=False \
#    workers=4
    # cache=disk

echo "========================================"
echo "RESTORE ORIGINAL ATTENTION"
echo "========================================"


yolo detect train \
    model=/home/agi/thesis_code/yolo26_versions/togroth_m5_depth_045_width_0200.yaml \
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

yolo detect train \
    model=/home/agi/thesis_code/yolo26_versions/togroth_m6_depth_040_width_0200.yaml \
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
