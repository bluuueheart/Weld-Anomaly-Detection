#!/bin/bash
# 蹇€熸鏌ヨ缁冩棩蹇楃姸鎬?

# 鍙兘鐨勬棩蹇楁枃浠惰矾寰?
POSSIBLE_PATHS=(
    "/root/autodl-tmp/outputs/logs/training_log.json"
    "logs/training_log.json"
    "outputs/logs/training_log.json"
)

echo "=========================================="
echo "璁粌鏃ュ織妫€鏌?
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
    echo "鉂?鎵句笉鍒版棩蹇楁枃浠?
    echo "   灏濊瘯鐨勮矾寰?"
    for path in "${POSSIBLE_PATHS[@]}"; do
        echo "   - $path"
    done
    echo ""
    echo "馃挕 鎻愮ず:"
    echo "   1. 纭繚宸茶繍琛岃缁? bash scripts/train.sh"
    echo "   2. 妫€鏌ヨ缁冮厤缃腑鐨?LOG_DIR 璺緞"
    echo "   3. 鏃ュ織鍦ㄦ瘡涓猠poch缁撴潫鍚庤嚜鍔ㄤ繚瀛?
    exit 1
fi

echo "鉁?鎵惧埌鏃ュ織鏂囦欢: $LOG_FILE"
echo ""

# 浣跨敤Python瑙ｆ瀽JSON骞舵彁鍙栧叧閿俊鎭?
python -c "
import json
from pathlib import Path

log_file = Path('$LOG_FILE')
with open(log_file, 'r') as f:
    data = json.load(f)

train_log = data.get('train', [])
val_log = data.get('val', [])
best_metric = data.get('best_metric', None)

print('馃搳 璁粌姒傚喌:')
print(f'  鎬籈poch鏁? {len(train_log)}')

if train_log:
    print(f'  鍒濆璁粌鎹熷け: {train_log[0][\"loss\"]:.4f}')
    print(f'  鏈€缁堣缁冩崯澶? {train_log[-1][\"loss\"]:.4f}')
    print(f'  璁粌鎹熷け闄嶅箙: {train_log[0][\"loss\"] - train_log[-1][\"loss\"]:.4f}')
    
    if 'lr' in train_log[-1]:
        print(f'  褰撳墠瀛︿範鐜? {train_log[-1][\"lr\"]:.2e}')

print()

if val_log:
    print('馃搱 楠岃瘉姒傚喌:')
    print(f'  楠岃瘉娆℃暟: {len(val_log)}')
    print(f'  鍒濆楠岃瘉鎹熷け: {val_log[0][\"loss\"]:.4f}')
    print(f'  鏈€缁堥獙璇佹崯澶? {val_log[-1][\"loss\"]:.4f}')
    
    if best_metric is not None:
        # 鎵惧埌鏈€浣砮poch
        val_losses = [v['loss'] for v in val_log]
        if best_metric in val_losses:
            best_epoch = val_losses.index(best_metric) + 1
            print(f'  鏈€浣抽獙璇佹崯澶? {best_metric:.4f} (Epoch {best_epoch})')
        else:
            print(f'  鏈€浣抽獙璇佹崯澶? {best_metric:.4f}')
    
    # 妫€鏌ユ槸鍚﹁繃鎷熷悎
    if len(val_log) > 5:
        recent_val = [v['loss'] for v in val_log[-5:]]
        if all(recent_val[i] >= recent_val[i-1] for i in range(1, len(recent_val))):
            print('  鈿狅笍  璀﹀憡: 鏈€杩?涓猠poch楠岃瘉鎹熷け鎸佺画涓婂崌(鍙兘杩囨嫙鍚?')
        elif recent_val[-1] < min(recent_val[:-1]):
            print('  鉁?鏈€杩慹poch楠岃瘉鎹熷け鏈夋敼鍠?)

print()
print('馃挕 鎻愮ず:')
print('  - 缁樺埗鎹熷け鏇茬嚎: python scripts/plot_loss.py')
print('  - 璇︾粏鍒嗘瀽鍥捐〃: python scripts/plot_loss.py --detailed')
" 2>/dev/null || echo "鉂?Python瑙ｆ瀽澶辫触,璇锋鏌SON鏍煎紡"

echo ""
echo "=========================================="
