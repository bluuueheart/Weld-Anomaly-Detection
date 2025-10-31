#!/bin/bash
# Quick test script for server environment (CUDA enabled)
# Tests loss computation and label distribution

echo "======================================================================"
echo "Step 1: Check Real Dataset Label Distribution"
echo "======================================================================"
echo ""

python tests/test_dataset_labels.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dataset label check failed! Fix data loading before training."
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 2: Test Loss Functions (Dummy Data)"
echo "======================================================================"
echo ""

python tests/test_loss_and_labels.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Step 3: Quick Training Test (3 epochs, debug mode, CUDA)"
    echo "======================================================================"
    echo ""
    
    # Run quick training test with debug output
    # --quick-test: 3 epochs, batch_size=4, dummy data
    # --debug: print first batch statistics
    # CUDA will be used automatically if available
    python src/train.py --quick-test --debug
    
    echo ""
    echo "✅ All tests completed successfully!"
    echo ""
    echo "To run full training on server (real data):"
    echo "  python src/train.py --batch-size 16 --num-epochs 100"
else
    echo ""
    echo "❌ Loss and label tests failed!"
    exit 1
fi
