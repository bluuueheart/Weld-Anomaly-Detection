#!/bin/bash

# Evaluation script for quad-modal welding anomaly detection

echo "=========================================="
echo "Evaluating Quad-Modal SOTA Model"
echo "=========================================="

# Default parameters
CHECKPOINT="/root/autodl-tmp/outputs/checkpoints/best_model.pth"
K=5
METRIC="cosine"
BATCH_SIZE=16

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dummy)
            DUMMY_FLAG="--dummy"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run evaluation
python src/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --k "$K" \
    --metric "$METRIC" \
    --batch-size "$BATCH_SIZE" \
    $DUMMY_FLAG

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
