#!/bin/bash

# 检查 StratifiedBatchSampler 是否正确工作

echo "=================================================="
echo "检查 StratifiedBatchSampler"
echo "=================================================="

python scripts/check_sampler.py

echo ""
echo "如果看到 ❌ 错误，说明采样器没有正确混合类别"
echo "请检查 src/samplers.py 中的实现"
