#!/bin/bash
# Training script for Late Fusion baseline

echo "=========================================="
echo "Late Fusion Baseline Training"
echo "=========================================="

# Configuration
DATA_ROOT="/root/autodl-tmp/Intel_Robotic_Welding_Multimodal_Dataset/raid/intel_robotic_welding_dataset"
MANIFEST="configs/manifest.csv"
SAVE_DIR="baselines/Late_Fusion/checkpoints"
DEVICE="cuda"

# Parse arguments
MODALITY="both"  # audio, video, or both
DUMMY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --dummy)
            DUMMY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python baselines/Late_Fusion/train.py \
    --modality $MODALITY \
    --data_root $DATA_ROOT \
    --manifest $MANIFEST \
    --save_dir $SAVE_DIR \
    --device $DEVICE"

if [ "$DUMMY" = true ]; then
    CMD="$CMD --dummy"
fi

# Run training
echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training completed"
echo "=========================================="
