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

# Training command
python src/train_causal_film.py \
    --config configs/train_config.py

echo ""
echo "==============================================="
echo "Training Complete"
echo "==============================================="
