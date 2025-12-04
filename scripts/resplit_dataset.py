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

def resplit_unsupervised_exact(seed=42):
    """
    重新划分数据集为无监督学习模式 (Train/Val/Test)
    
    严格按照以下数量划分:
    Train (Good): 576
    Val (Good): 122, Val (Defective): 1610
    Test (Good): 121, Test (Defective): 1611
    """
    print("=" * 70)
    print("重新划分数据集 (无监督学习模式 - 严格数量)")
    print("=" * 70)
    print(f"Manifest 文件: {MANIFEST_PATH}")
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
    
    # 分离 Good 和 Defective 样本
    good_samples = []
    defective_samples = []
    
    for row in rows:
        cat = row.get('CATEGORY', '').strip()
        if cat.lower() == 'good':
            good_samples.append(row)
        else:
            defective_samples.append(row)
            
    print(f"Good 样本数: {len(good_samples)}")
    print(f"Defective 样本数: {len(defective_samples)}")
    
    # 设置随机种子并打乱
    random.seed(seed)
    random.shuffle(good_samples)
    random.shuffle(defective_samples)
    
    # 划分 Good 样本
    # Train: 576, Val: 122, Test: 121
    n_train_good = 576
    n_val_good = 122
    n_test_good = 121
    
    if len(good_samples) != (n_train_good + n_val_good + n_test_good):
        print(f"⚠️ 警告: Good 样本总数 ({len(good_samples)}) 与预期 ({n_train_good + n_val_good + n_test_good}) 不符!")
    
    train_good = good_samples[:n_train_good]
    val_good = good_samples[n_train_good : n_train_good + n_val_good]
    test_good = good_samples[n_train_good + n_val_good :]
    
    # 划分 Defective 样本
    # Train: 0, Val: 1610, Test: 1611
    n_val_defective = 1610
    n_test_defective = 1611
    
    if len(defective_samples) != (n_val_defective + n_test_defective):
        print(f"⚠️ 警告: Defective 样本总数 ({len(defective_samples)}) 与预期 ({n_val_defective + n_test_defective}) 不符!")
        
    val_defective = defective_samples[:n_val_defective]
    test_defective = defective_samples[n_val_defective:]
    
    # 分配 SPLIT 标签
    new_rows = []
    
    for row in train_good:
        row['SPLIT'] = 'TRAIN'
        new_rows.append(row)
        
    for row in val_good:
        row['SPLIT'] = 'VAL'  # 使用 VAL
        new_rows.append(row)
        
    for row in val_defective:
        row['SPLIT'] = 'VAL'
        new_rows.append(row)
        
    for row in test_good:
        row['SPLIT'] = 'TEST'
        new_rows.append(row)
        
    for row in test_defective:
        row['SPLIT'] = 'TEST'
        new_rows.append(row)
        
    # 保存新的 manifest
    backup_path = MANIFEST_PATH + '.backup'
    print(f"\n备份原文件到: {backup_path}")
    
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        with open(backup_path, 'w', encoding='utf-8', newline='') as f_backup:
            f_backup.write(f.read())
            
    with open(MANIFEST_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
        
    print(f"✅ 已更新 manifest: {MANIFEST_PATH}")
    
    # 打印统计信息
    print(f"\n新的数据划分统计:")
    print(f"  Train (Good): {len(train_good)}")
    print(f"  Val   (Good): {len(val_good)}, (Defective): {len(val_defective)}, Total: {len(val_good) + len(val_defective)}")
    print(f"  Test  (Good): {len(test_good)}, (Defective): {len(test_defective)}, Total: {len(test_good) + len(test_defective)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--unsupervised', action='store_true', help='使用无监督学习模式划分 (Train/Val/Test)')
    args = parser.parse_args()
    
    if args.unsupervised:
        resplit_unsupervised_exact(seed=args.seed)
    else:
        resplit_dataset(train_ratio=args.train_ratio, seed=args.seed)
