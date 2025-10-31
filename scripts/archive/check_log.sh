#!/bin/bash
# 快速检查训练日志状态

# 可能的日志文件路径
POSSIBLE_PATHS=(
    "/root/autodl-tmp/outputs/logs/training_log.json"
    "logs/training_log.json"
    "outputs/logs/training_log.json"
)

echo "=========================================="
echo "训练日志检查"
echo "=========================================="
echo ""

LOG_FILE=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        LOG_FILE="$path"
        break
    fi
done

if [ -z "$LOG_FILE" ]; then
    echo "❌ 找不到日志文件"
    echo "   尝试的路径:"
    for path in "${POSSIBLE_PATHS[@]}"; do
        echo "   - $path"
    done
    echo ""
    echo "💡 提示:"
    echo "   1. 确保已运行训练: bash scripts/train.sh"
    echo "   2. 检查训练配置中的 LOG_DIR 路径"
    echo "   3. 日志在每个epoch结束后自动保存"
    exit 1
fi

echo "✅ 找到日志文件: $LOG_FILE"
echo ""

# 使用Python解析JSON并提取关键信息
python -c "
import json
from pathlib import Path

log_file = Path('$LOG_FILE')
with open(log_file, 'r') as f:
    data = json.load(f)

train_log = data.get('train', [])
val_log = data.get('val', [])
best_metric = data.get('best_metric', None)

print('📊 训练概况:')
print(f'  总Epoch数: {len(train_log)}')

if train_log:
    print(f'  初始训练损失: {train_log[0][\"loss\"]:.4f}')
    print(f'  最终训练损失: {train_log[-1][\"loss\"]:.4f}')
    print(f'  训练损失降幅: {train_log[0][\"loss\"] - train_log[-1][\"loss\"]:.4f}')
    
    if 'lr' in train_log[-1]:
        print(f'  当前学习率: {train_log[-1][\"lr\"]:.2e}')

print()

if val_log:
    print('📈 验证概况:')
    print(f'  验证次数: {len(val_log)}')
    print(f'  初始验证损失: {val_log[0][\"loss\"]:.4f}')
    print(f'  最终验证损失: {val_log[-1][\"loss\"]:.4f}')
    
    if best_metric is not None:
        # 找到最佳epoch
        val_losses = [v['loss'] for v in val_log]
        if best_metric in val_losses:
            best_epoch = val_losses.index(best_metric) + 1
            print(f'  最佳验证损失: {best_metric:.4f} (Epoch {best_epoch})')
        else:
            print(f'  最佳验证损失: {best_metric:.4f}')
    
    # 检查是否过拟合
    if len(val_log) > 5:
        recent_val = [v['loss'] for v in val_log[-5:]]
        if all(recent_val[i] >= recent_val[i-1] for i in range(1, len(recent_val))):
            print('  ⚠️  警告: 最近5个epoch验证损失持续上升(可能过拟合)')
        elif recent_val[-1] < min(recent_val[:-1]):
            print('  ✅ 最近epoch验证损失有改善')

print()
print('💡 提示:')
print('  - 绘制损失曲线: python scripts/plot_loss.py')
print('  - 详细分析图表: python scripts/plot_loss.py --detailed')
" 2>/dev/null || echo "❌ Python解析失败,请检查JSON格式"

echo ""
echo "=========================================="
