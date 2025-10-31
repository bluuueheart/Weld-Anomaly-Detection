#!/bin/bash
# 绘制训练损失曲线

cd "$(dirname "$0")/.."

# 基础用法: 绘制损失曲线
python scripts/plot_loss.py "$@"
