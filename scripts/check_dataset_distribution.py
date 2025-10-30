"""
检查数据集的类别分布
"""

import csv
import os
import sys
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.dataset_config import MANIFEST_PATH

# Category mapping (same as in dataset.py)
CATEGORY_MAP = {
    "good": 0,
    "excessive_convexity": 1,
    "undercut": 2,
    "lack_of_fusion": 3,
    "porosity": 5,
    "spatter": 6,
    "burnthrough": 7,
    "porosity_w_excessive_penetration": 4,
    "excessive_penetration": 8,
    "excessive penetration": 8,
    "crater_cracks": 9,
    "warping": 10,
    "overlap": 11,
}

def check_distribution():
    """检查训练集和测试集的类别分布"""
    
    print("=" * 70)
    print("检查数据集类别分布")
    print("=" * 70)
    print(f"Manifest 文件: {MANIFEST_PATH}\n")
    
    if not os.path.isfile(MANIFEST_PATH):
        print(f"❌ 错误: Manifest 文件不存在: {MANIFEST_PATH}")
        return
    
    # 读取 manifest
    train_categories = []
    test_categories = []
    
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get('CATEGORY', '').strip()
            split = row.get('SPLIT', '').strip().upper()
            
            if not category or not split:
                continue
            
            # 归一化类别名
            category_key = category.lower().replace(' ', '_')
            label = CATEGORY_MAP.get(category_key, 0)
            
            if split == 'TRAIN':
                train_categories.append((category, label))
            elif split == 'TEST':
                test_categories.append((category, label))
    
    # 统计训练集
    print("训练集 (TRAIN):")
    print("-" * 70)
    print(f"总样本数: {len(train_categories)}")
    
    train_labels = [label for _, label in train_categories]
    train_label_counts = Counter(train_labels)
    
    train_category_names = [cat for cat, _ in train_categories]
    train_category_counts = Counter(train_category_names)
    
    print(f"\n按类别名统计:")
    for cat, count in train_category_counts.most_common():
        pct = 100.0 * count / len(train_categories)
        print(f"  {cat:30s}: {count:4d} 样本 ({pct:5.1f}%)")
    
    print(f"\n按标签统计:")
    for label in sorted(train_label_counts.keys()):
        count = train_label_counts[label]
        pct = 100.0 * count / len(train_categories)
        print(f"  标签 {label:2d}: {count:4d} 样本 ({pct:5.1f}%)")
    
    # 统计测试集
    print(f"\n\n测试集 (TEST):")
    print("-" * 70)
    print(f"总样本数: {len(test_categories)}")
    
    test_labels = [label for _, label in test_categories]
    test_label_counts = Counter(test_labels)
    
    test_category_names = [cat for cat, _ in test_categories]
    test_category_counts = Counter(test_category_names)
    
    print(f"\n按类别名统计:")
    for cat, count in test_category_counts.most_common():
        pct = 100.0 * count / len(test_categories)
        print(f"  {cat:30s}: {count:4d} 样本 ({pct:5.1f}%)")
    
    print(f"\n按标签统计:")
    for label in sorted(test_label_counts.keys()):
        count = test_label_counts[label]
        pct = 100.0 * count / len(test_categories)
        print(f"  标签 {label:2d}: {count:4d} 样本 ({pct:5.1f}%)")
    
    # 诊断
    print("\n" + "=" * 70)
    print("诊断结果")
    print("=" * 70)
    
    # 检查训练集是否严重不平衡
    if len(train_label_counts) == 1:
        only_label = list(train_label_counts.keys())[0]
        print(f"❌ 严重问题: 训练集只有一个类别 (标签 {only_label})")
        print(f"   所有 {len(train_categories)} 个样本都是同一类!")
        print(f"\n   这就是为什么:")
        print(f"   - SupConLoss 计算的是 -log(batch_size-1) ≈ {-1 * __import__('math').log(31):.4f}")
        print(f"   - Loss 完全不下降 (恒定值)")
        print(f"   - StratifiedBatchSampler 只能采样到类别 {only_label}")
        print(f"\n   解决方案:")
        print(f"   1. 使用完整数据集（包含多个类别）")
        print(f"   2. 或切换到交叉熵损失 (不推荐 - 只有1类无法训练)")
        print(f"   3. 或使用无监督方法（自监督学习）")
    elif len(train_label_counts) < 6:
        print(f"⚠️  警告: 训练集只有 {len(train_label_counts)} 个类别")
        print(f"   期望: 6 个类别")
        print(f"   当前: {sorted(train_label_counts.keys())}")
    else:
        # 检查是否严重不平衡
        max_count = max(train_label_counts.values())
        min_count = min(train_label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 10:
            print(f"⚠️  警告: 数据集严重不平衡!")
            print(f"   最多类: {max_count} 样本")
            print(f"   最少类: {min_count} 样本")
            print(f"   不平衡比: {imbalance_ratio:.1f}:1")
            print(f"\n   建议:")
            print(f"   1. 使用加权采样 (WeightedRandomSampler)")
            print(f"   2. 或使用数据增强")
            print(f"   3. 或使用 Focal Loss")
        else:
            print(f"✅ 训练集类别分布相对均衡")
            print(f"   类别数: {len(train_label_counts)}")
            print(f"   不平衡比: {imbalance_ratio:.1f}:1")
    
    print("=" * 70)

if __name__ == "__main__":
    check_distribution()
