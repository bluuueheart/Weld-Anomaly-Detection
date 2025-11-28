#!/bin/bash
# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python src/evaluate_fusion.py
