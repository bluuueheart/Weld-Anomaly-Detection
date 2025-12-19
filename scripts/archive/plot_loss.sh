#!/bin/bash
# 缁樺埗璁粌鎹熷け鏇茬嚎

cd "$(dirname "$0")/.."

# 鍩虹鐢ㄦ硶: 缁樺埗鎹熷け鏇茬嚎
python scripts/plot_loss.py "$@"
