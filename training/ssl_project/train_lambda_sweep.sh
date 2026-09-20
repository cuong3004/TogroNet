#!/bin/bash

set -e

WEIGHT="last_200.pt"

LAMBDAS=(
  0.005
)

LOG_DIR="logs/ssl_lambda_sweep_09_yolo"
mkdir -p "$LOG_DIR"

for LAMBDA in "${LAMBDAS[@]}"; do

  EXP_NAME="ssl_pretrain_200_${WEIGHT%.yaml}_lambda${LAMBDA}"

  echo "=========================================="
  echo "Training experiment: $EXP_NAME"
  echo "Weight/YAML: $WEIGHT"
  echo "Lambda: $LAMBDA"
  echo "=========================================="

  python train.py \
    --weight "$WEIGHT" \
    --lambda-coeff "$LAMBDA" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}.log"

done

echo "All lambda experiments finished."
