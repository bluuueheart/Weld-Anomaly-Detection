#!/bin/bash
# 验证V4配置是否正确应用

echo "=========================================="
echo "验证 V4 配置"
echo "=========================================="
echo ""

echo "检查 train_config.py..."
python -c "
from configs.train_config import TRAIN_CONFIG as TC
checks = {
    'learning_rate': (TC.get('learning_rate'), 5e-5),
    'weight_decay': (TC.get('weight_decay'), 1e-2),
    'warmup_epochs': (TC.get('warmup_epochs'), 5),
    'early_stopping_patience': (TC.get('early_stopping_patience'), 8),
    'use_mixup': (TC.get('use_mixup'), True),
    'mixup_alpha': (TC.get('mixup_alpha'), 0.2),
}

all_ok = True
for key, (actual, expected) in checks.items():
    status = '✓' if actual == expected else '✗'
    print(f'{status} {key}: {actual} (期望: {expected})')
    if actual != expected:
        all_ok = False

if all_ok:
    print('\n✅ 所有配置正确!')
else:
    print('\n❌ 配置不匹配，请检查 configs/train_config.py')
"

echo ""
echo "检查 train.py (MixUp函数是否存在)..."
python -c "
import inspect
from src.train import Trainer

if hasattr(Trainer, '_mixup_features'):
    sig = inspect.signature(Trainer._mixup_features)
    params = list(sig.parameters.keys())
    print(f'✓ Trainer._mixup_features 存在，参数: {params}')
else:
    print('✗ Trainer._mixup_features 不存在')
"

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="
