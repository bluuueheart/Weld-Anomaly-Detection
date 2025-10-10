#!/bin/bash
# Training script for quad-modal model

cd "$(dirname "$0")/.."
python src/train.py "$@"
