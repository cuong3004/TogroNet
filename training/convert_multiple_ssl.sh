#!/bin/bash
set -e

cd /home/agi/thesis_code

# Bước 1: patch attention trước khi load checkpoint YOLO
python3 patch_attention.py replace

# Checkpoint YOLO gốc dùng cho tất cả các lần chuyển đổi
BASE_YOLO="/home/agi/thesis_code/runs/detect/train-14/weights/best.pt"

# Bước 2: chuyển từng SSL checkpoint sang trọng số YOLO
# python3 convert_ssl_ckpt_to_yolo.py \
#     --ssl_ckpt /home/agi/thesis_code/ssl_project/lightning_logs/version_0/checkpoints/last.ckpt \
#     --base_yolo "$BASE_YOLO" \
#     --output /home/agi/thesis_code/yolo_ssl_version_0.pt

# python3 convert_ssl_ckpt_to_yolo.py \
#     --ssl_ckpt /home/agi/thesis_code/ssl_project/checkpoints/ssl_lastpretrain_lambda0.001/last.ckpt \
#     --base_yolo "$BASE_YOLO" \
#     --output /home/agi/thesis_code/yolo_ssl_version_1.pt

python3 convert_ssl_ckpt_to_yolo.py \
    --ssl_ckpt ssl_project/checkpoints/ssl_last_lambda0.005/epoch=499-step=585500.ckpt \
    --base_yolo "$BASE_YOLO" \
    --output /home/agi/thesis_code/yolo_ssl_version_2_2.pt

# python3 convert_ssl_ckpt_to_yolo.py \
#     --ssl_ckpt /home/agi/thesis_code/ssl_project/checkpoints/ssl_lastpretrain_lambda0.01/last.ckpt \
#     --base_yolo "$BASE_YOLO" \
#     --output /home/agi/thesis_code/yolo_ssl_version_3.pt