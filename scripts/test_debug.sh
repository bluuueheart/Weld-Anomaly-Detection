#!/bin/bash
# Quick debug test for training with dummy data

echo "======================================================================"
echo "Testing training with debug output (dummy data)"
echo "======================================================================"

python src/train.py \
    --use-dummy \
    --debug \
    --num-epochs 2 \
    --batch-size 4 \
    --device cpu

echo ""
echo "======================================================================"
echo "Debug test completed"
echo "======================================================================"
