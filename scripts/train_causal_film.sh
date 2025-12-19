#!/bin/bash
# Training script for Causal-FiLM model

echo "==============================================="
echo "Training Causal-FiLM Model"
echo "==============================================="
echo ""

# Activate environment (if needed)
# source /path/to/your/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if arguments are provided
if [ $# -eq 0 ]; then
    # No arguments: use default training command
    python src/train_causal_film.py --config configs/train_config.py --wandb --wandb_project "weld-anomaly-detection" --wandb_name "causal_film_run_$(date +%Y%m%d_%H%M%S)"
else
    # Arguments provided: forward all arguments to the training script
    python src/train_causal_film.py "$@"
fi

echo ""
echo "==============================================="
echo "Training Complete"
echo "==============================================="

