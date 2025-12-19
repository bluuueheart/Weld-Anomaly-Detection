#!/bin/bash
# 楠岃瘉鎵€鏈変慨澶嶆槸鍚︽纭簲鐢?

echo "======================================================================"
echo "浠ｇ爜淇楠岃瘉"
echo "======================================================================"
echo ""

# 1. 妫€鏌ataset.py涓殑鏍囩鐢熸垚
echo "鉁?妫€鏌?dataset.py 鏍囩鐢熸垚閫昏緫..."
if grep -q "num_classes = 6" src/dataset.py && grep -q "random.shuffle" src/dataset.py; then
    echo "  鉁?鏍囩鐢熸垚宸蹭慨澶嶏紙6涓被鍒紝闅忔満鎵撲贡锛?
else
    echo "  鉂?鏍囩鐢熸垚鏈纭慨澶?
    exit 1
fi

# 2. 妫€鏌ユ病鏈夐噸澶嶇殑import random
echo "鉁?妫€鏌?dataset.py 涓病鏈夐噸澶?import random..."
count=$(grep -c "^import random" src/dataset.py)
if [ "$count" -eq 1 ]; then
    echo "  鉁?random鍙湪妯″潡椤堕儴瀵煎叆涓€娆?
else
    echo "  鉂?鍙戠幇閲嶅鐨?import random"
    exit 1
fi

# 3. 妫€鏌rain.py涓殑CUDA榛樿閰嶇疆
echo "鉁?妫€鏌?train.py CUDA榛樿閰嶇疆..."
if grep -q "config\['device'\] = 'cuda' if torch.cuda.is_available()" src/train.py; then
    echo "  鉁?榛樿浣跨敤CUDA锛堝鏋滃彲鐢級"
else
    echo "  鉂?CUDA閰嶇疆鏈纭缃?
    exit 1
fi

# 4. 妫€鏌ユ祴璇曟枃浠跺瓨鍦?
echo "鉁?妫€鏌ユ祴璇曟枃浠?.."
if [ -f "tests/test_loss_and_labels.py" ]; then
    echo "  鉁?娴嬭瘯鏂囦欢瀛樺湪"
else
    echo "  鉂?娴嬭瘯鏂囦欢缂哄け"
    exit 1
fi

# 5. 妫€鏌ユ枃妗ｆ枃浠?
echo "鉁?妫€鏌ユ枃妗ｆ枃浠?.."
docs_ok=true
for doc in "docs/LOSS_FIX.md" "docs/SERVER_TESTING.md" "docs/FIX_SUMMARY.md"; do
    if [ -f "$doc" ]; then
        echo "  鉁?$doc 瀛樺湪"
    else
        echo "  鉂?$doc 缂哄け"
        docs_ok=false
    fi
done

if [ "$docs_ok" = false ]; then
    exit 1
fi

echo ""
echo "======================================================================"
echo "鉁?鎵€鏈変慨澶嶉獙璇侀€氳繃锛?
echo "======================================================================"
echo ""
echo "涓嬩竴姝ワ細"
echo "  1. 鍦ㄦ湇鍔″櫒涓婅繍琛? bash scripts/test_server.sh"
echo "  2. 鎴栨墜鍔ㄦ祴璇? python tests/test_loss_and_labels.py"
echo "  3. 蹇€熻缁? python src/train.py --quick-test --debug"
echo ""
