#!/bin/bash
# Evaluation script for Causal-FiLM model

echo "==============================================="
echo "Evaluating Causal-FiLM Model"
echo "==============================================="
echo ""

# Activate environment (if needed)
# source /path/to/your/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 /root/autodl-tmp/outputs/checkpoints/best_model.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

# Evaluation command
python src/evaluate_causal_film.py --checkpoint "$CHECKPOINT_PATH" --split test --device cuda

echo ""
echo "==============================================="
echo "Evaluation Complete"
echo "==============================================="
