#!/bin/bash
# Complete test suite for Quad-Modal SOTA Model
# 瀹屾暣娴嬭瘯濂椾欢 - 閫愪釜娴嬭瘯鎵€鏈夋ā鍧?

echo "======================================================================"
echo "RUNNING COMPLETE TEST SUITE"
echo "======================================================================"
echo ""

# Get project root directory
cd "$(dirname "$0")/.."

# Test 1: Dataset
echo ">>> Test 1/5: Dataset"
bash scripts/test_dataset.sh
if [ $? -ne 0 ]; then
    echo "鉂?Dataset test failed"
    exit 1
fi
echo ""

# Test 2: Encoders
echo ">>> Test 2/5: Encoders"
bash scripts/test_encoders.sh
if [ $? -ne 0 ]; then
    echo "鉂?Encoder test failed"
    exit 1
fi
echo ""

# Test 3: Fusion
echo ">>> Test 3/5: Fusion"
bash scripts/test_fusion.sh
if [ $? -ne 0 ]; then
    echo "鉂?Fusion test failed"
    exit 1
fi
echo ""

# Test 4: Model
echo ">>> Test 4/5: Model"
bash scripts/test_model.sh
if [ $? -ne 0 ]; then
    echo "鉂?Model test failed"
    exit 1
fi
echo ""

# Test 5: Losses
echo ">>> Test 5/5: Losses"
bash scripts/test_losses.sh
if [ $? -ne 0 ]; then
    echo "鉂?Loss test failed"
    exit 1
fi
echo ""

echo "======================================================================"
echo "鉁?ALL TESTS PASSED!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review test outputs above"
echo "  2. Configure training parameters in configs/train_config.py"
echo "  3. Start training with: bash scripts/train.sh"
echo ""
