#!/bin/bash
# 娴嬭瘯鏃ュ織鏌ユ壘鍔熻兘

cd "$(dirname "$0")/.."

echo "=========================================="
echo "娴嬭瘯鏃ュ織鑷姩鏌ユ壘鍔熻兘"
echo "=========================================="
echo ""

# 鍒涘缓娴嬭瘯鏃ュ織鏂囦欢
TEST_LOG_DIR="test_outputs/logs"
mkdir -p "$TEST_LOG_DIR"

cat > "$TEST_LOG_DIR/training_log.json" << 'EOF'
{
  "train": [
    {"loss": 2.95, "time": 45.2, "lr": 1e-6},
    {"loss": 2.45, "time": 42.1, "lr": 5e-5}
  ],
  "val": [
    {"loss": 3.40, "time": 12.3},
    {"loss": 3.05, "time": 11.8}
  ],
  "config": {"test": true},
  "best_metric": 3.05
}
EOF

echo "鉁?鍒涘缓娴嬭瘯鏃ュ織鏂囦欢: $TEST_LOG_DIR/training_log.json"
echo ""

# 娴嬭瘯鑷姩鏌ユ壘
echo "娴嬭瘯鑷姩鏌ユ壘鍔熻兘..."
python -c "
from scripts.plot_loss import find_log_file
log_path = find_log_file()
if log_path:
    print(f'鉁?鎵惧埌鏃ュ織鏂囦欢: {log_path}')
else:
    print('鉂?鏈壘鍒版棩蹇楁枃浠?)
"

echo ""
echo "娴嬭瘯缁樺浘鑴氭湰..."
python scripts/plot_loss.py --log "$TEST_LOG_DIR/training_log.json" --output test_outputs

echo ""
echo "娓呯悊娴嬭瘯鏂囦欢..."
rm -rf test_outputs

echo ""
echo "=========================================="
echo "娴嬭瘯瀹屾垚"
echo "=========================================="