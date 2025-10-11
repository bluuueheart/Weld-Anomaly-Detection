#!/bin/bash
# 验证所有修复是否正确应用

echo "======================================================================"
echo "代码修复验证"
echo "======================================================================"
echo ""

# 1. 检查dataset.py中的标签生成
echo "✓ 检查 dataset.py 标签生成逻辑..."
if grep -q "num_classes = 6" src/dataset.py && grep -q "random.shuffle" src/dataset.py; then
    echo "  ✅ 标签生成已修复（6个类别，随机打乱）"
else
    echo "  ❌ 标签生成未正确修复"
    exit 1
fi

# 2. 检查没有重复的import random
echo "✓ 检查 dataset.py 中没有重复 import random..."
count=$(grep -c "^import random" src/dataset.py)
if [ "$count" -eq 1 ]; then
    echo "  ✅ random只在模块顶部导入一次"
else
    echo "  ❌ 发现重复的 import random"
    exit 1
fi

# 3. 检查train.py中的CUDA默认配置
echo "✓ 检查 train.py CUDA默认配置..."
if grep -q "config\['device'\] = 'cuda' if torch.cuda.is_available()" src/train.py; then
    echo "  ✅ 默认使用CUDA（如果可用）"
else
    echo "  ❌ CUDA配置未正确设置"
    exit 1
fi

# 4. 检查测试文件存在
echo "✓ 检查测试文件..."
if [ -f "tests/test_loss_and_labels.py" ]; then
    echo "  ✅ 测试文件存在"
else
    echo "  ❌ 测试文件缺失"
    exit 1
fi

# 5. 检查文档文件
echo "✓ 检查文档文件..."
docs_ok=true
for doc in "docs/LOSS_FIX.md" "docs/SERVER_TESTING.md" "docs/FIX_SUMMARY.md"; do
    if [ -f "$doc" ]; then
        echo "  ✅ $doc 存在"
    else
        echo "  ❌ $doc 缺失"
        docs_ok=false
    fi
done

if [ "$docs_ok" = false ]; then
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ 所有修复验证通过！"
echo "======================================================================"
echo ""
echo "下一步："
echo "  1. 在服务器上运行: bash scripts/test_server.sh"
echo "  2. 或手动测试: python tests/test_loss_and_labels.py"
echo "  3. 快速训练: python src/train.py --quick-test --debug"
echo ""
