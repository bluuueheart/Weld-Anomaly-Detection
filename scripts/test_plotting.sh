#!/bin/bash
# 测试日志查找功能

cd "$(dirname "$0")/.."

echo "=========================================="
echo "测试日志自动查找功能"
echo "=========================================="
echo ""

# 创建测试日志文件
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

echo "✅ 创建测试日志文件: $TEST_LOG_DIR/training_log.json"
echo ""

# 测试自动查找
echo "测试自动查找功能..."
python -c "
from scripts.plot_loss import find_log_file
log_path = find_log_file()
if log_path:
    print(f'✅ 找到日志文件: {log_path}')
else:
    print('❌ 未找到日志文件')
"

echo ""
echo "测试绘图脚本..."
python scripts/plot_loss.py --log "$TEST_LOG_DIR/training_log.json" --output test_outputs

echo ""
echo "清理测试文件..."
rm -rf test_outputs

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="