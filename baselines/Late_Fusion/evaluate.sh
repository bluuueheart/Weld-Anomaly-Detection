#!/bin/bash
# Evaluation script for Late Fusion baseline

echo "=========================================="
echo "Late Fusion Baseline Evaluation"
echo "=========================================="

# Configuration
DATA_ROOT="/root/autodl-tmp/Intel_Robotic_Welding_Multimodal_Dataset/raid/intel_robotic_welding_dataset"
MANIFEST="configs/manifest.csv"
AUDIO_CHECKPOINT="baselines/Late_Fusion/checkpoints/audio_autoencoder_best.pth"
VIDEO_CHECKPOINT="baselines/Late_Fusion/checkpoints/video_autoencoder_best.pth"
OUTPUT_DIR="baselines/Late_Fusion/results"
DEVICE="cuda"

# Parse arguments
DUMMY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --audio_checkpoint)
            AUDIO_CHECKPOINT="$2"
            shift 2
            ;;
        --video_checkpoint)
            VIDEO_CHECKPOINT="$2"
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
CMD="python baselines/Late_Fusion/evaluate.py \
    --audio_checkpoint $AUDIO_CHECKPOINT \
    --video_checkpoint $VIDEO_CHECKPOINT \
    --data_root $DATA_ROOT \
    --manifest $MANIFEST \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE"

if [ "$DUMMY" = true ]; then
    CMD="$CMD --dummy"
fi

# Run evaluation
echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation completed"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
