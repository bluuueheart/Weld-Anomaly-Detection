"""
重新划分数据集 - 训练/测试 80/20 分割
确保训练集包含所有类别
"""

import csv
import os
import sys
from collections import defaultdict
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.dataset_config import MANIFEST_PATH

def resplit_dataset(train_ratio=0.8, seed=42):
    """重新划分数据集为训练/测试集"""
    
    print("=" * 70)
    print("重新划分数据集")
    print("=" * 70)
    print(f"Manifest 文件: {MANIFEST_PATH}")
    print(f"训练比例: {train_ratio * 100:.0f}%")
    print(f"随机种子: {seed}\n")
    
    if not os.path.isfile(MANIFEST_PATH):
        print(f"❌ 错误: Manifest 文件不存在")
        return
    
    # 读取所有数据
    rows = []
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"总样本数: {len(rows)}")
    
    # 按类别分组
    category_groups = defaultdict(list)
    for row in rows:
        category = row.get('CATEGORY', '').strip()
        if category:
            category_groups[category].append(row)
    
    print(f"类别数: {len(category_groups)}")
    for cat, samples in sorted(category_groups.items()):
        print(f"  {cat:30s}: {len(samples):4d} 样本")
    
    # 设置随机种子
    random.seed(seed)
    
    # 对每个类别进行分层划分
    new_rows = []
    train_count = 0
    test_count = 0
    
    for category, samples in category_groups.items():
        # 打乱样本
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # 计算训练集大小
        n_train = max(1, int(len(shuffled) * train_ratio))  # 至少1个训练样本
        
        # 划分
        for i, row in enumerate(shuffled):
            if i < n_train:
                row['SPLIT'] = 'TRAIN'
                train_count += 1
            else:
                row['SPLIT'] = 'TEST'
                test_count += 1
            new_rows.append(row)
    
    # 保存新的 manifest
    backup_path = MANIFEST_PATH + '.backup'
    print(f"\n备份原文件到: {backup_path}")
    
    # 备份原文件
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        with open(backup_path, 'w', encoding='utf-8', newline='') as f_backup:
            f_backup.write(f.read())
    
    # 写入新文件
    with open(MANIFEST_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    
    print(f"✅ 已更新 manifest: {MANIFEST_PATH}")
    print(f"\n新的数据划分:")
    print(f"  训练集: {train_count} 样本 ({100.0 * train_count / len(rows):.1f}%)")
    print(f"  测试集: {test_count} 样本 ({100.0 * test_count / len(rows):.1f}%)")
    
    # 统计训练集类别分布
    train_categories = defaultdict(int)
    test_categories = defaultdict(int)
    
    for row in new_rows:
        cat = row.get('CATEGORY', '').strip()
        split = row.get('SPLIT', '').strip().upper()
        if split == 'TRAIN':
            train_categories[cat] += 1
        elif split == 'TEST':
            test_categories[cat] += 1
    
    print(f"\n训练集类别分布:")
    for cat in sorted(train_categories.keys()):
        count = train_categories[cat]
        print(f"  {cat:30s}: {count:4d} 样本")
    
    print(f"\n测试集类别分布:")
    for cat in sorted(test_categories.keys()):
        count = test_categories[cat]
        print(f"  {cat:30s}: {count:4d} 样本")
    
    print("\n" + "=" * 70)
    print("✅ 重新划分完成!")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 运行: python scripts/check_dataset_distribution.py  (验证划分)")
    print("  2. 运行: python scripts/check_sampler.py  (验证采样器)")
    print("  3. 运行: python src/train.py  (开始训练)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    args = parser.parse_args()
    
    resplit_dataset(train_ratio=args.train_ratio, seed=args.seed)
