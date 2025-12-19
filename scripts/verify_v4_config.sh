#!/bin/bash
# 楠岃瘉V4閰嶇疆鏄惁姝ｇ‘搴旂敤

echo "=========================================="
echo "楠岃瘉 V4 閰嶇疆"
echo "=========================================="
echo ""

echo "妫€鏌?train_config.py..."
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
    status = '鉁? if actual == expected else '鉁?
    print(f'{status} {key}: {actual} (鏈熸湜: {expected})')
    if actual != expected:
        all_ok = False

if all_ok:
    print('\n鉁?鎵€鏈夐厤缃纭?')
else:
    print('\n鉂?閰嶇疆涓嶅尮閰嶏紝璇锋鏌?configs/train_config.py')
"

echo ""
echo "妫€鏌?train.py (MixUp鍑芥暟鏄惁瀛樺湪)..."
python -c "
import inspect
from src.train import Trainer

if hasattr(Trainer, '_mixup_features'):
    sig = inspect.signature(Trainer._mixup_features)
    params = list(sig.parameters.keys())
    print(f'鉁?Trainer._mixup_features 瀛樺湪锛屽弬鏁? {params}')
else:
    print('鉁?Trainer._mixup_features 涓嶅瓨鍦?)
"

echo ""
echo "=========================================="
echo "楠岃瘉瀹屾垚"
echo "=========================================="
