#!/bin/bash
# Plot training loss and accuracy (loss_and_accuracy.png)
# Runs the Python plotting utility. No args required.

python scripts/plot_loss.py "$@"

echo "Plotting complete."
