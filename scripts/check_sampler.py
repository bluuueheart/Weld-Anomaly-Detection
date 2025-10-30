"""
检查 StratifiedBatchSampler 是否正确工作
验证每个 batch 是否包含多个类别
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.dataset import WeldingDataset
from src.samplers import StratifiedBatchSampler
from configs.dataset_config import *

def check_sampler():
    """检查采样器是否正常工作"""
    
    print("=" * 70)
    print("检查 StratifiedBatchSampler")
    print("=" * 70)
    
    # 创建数据集
    dataset = WeldingDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        split="train",
        video_length=VIDEO_LENGTH,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        sensor_length=SENSOR_LENGTH,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        dummy=False
    )
    
    print(f"数据集样本数: {len(dataset)}")
    
    # 统计类别分布
    all_labels = [dataset[i]['label'] for i in range(len(dataset))]
    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\n类别分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  类别 {label}: {count} 样本")
    
    # 创建采样器
    batch_size = 32
    sampler = StratifiedBatchSampler(
        labels=all_labels,
        batch_size=batch_size
    )
    
    print(f"\n批次大小: {batch_size}")
    print(f"批次数量: {len(sampler)}")
    
    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    
    # 检查前几个批次
    print(f"\n检查前 5 个批次:")
    print("-" * 70)
    
    all_good = True
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5:
            break
        
        labels = batch['label']
        unique, counts = torch.unique(labels, return_counts=True)
        
        print(f"\n批次 {batch_idx + 1}:")
        print(f"  样本数: {len(labels)}")
        print(f"  唯一类别: {unique.tolist()}")
        print(f"  每类数量: {counts.tolist()}")
        
        # 检查是否所有样本都是同一类别
        if len(unique) == 1:
            print(f"  ❌ 错误! 所有 {len(labels)} 个样本都是类别 {unique[0].item()}")
            print(f"  StratifiedBatchSampler 没有正确混合类别!")
            all_good = False
        else:
            print(f"  ✅ 正常! 批次包含 {len(unique)} 个不同类别")
    
    print("\n" + "=" * 70)
    if all_good:
        print("✅ 采样器工作正常 - 所有批次都包含多个类别")
    else:
        print("❌ 采样器有问题 - 某些批次只包含单一类别")
        print("\n建议:")
        print("1. 检查 src/samplers.py 中的 StratifiedBatchSampler.__iter__() 方法")
        print("2. 确保采样逻辑从所有类别中均匀采样")
        print("3. 验证 shuffle 操作是否正确")
    print("=" * 70)

if __name__ == "__main__":
    check_sampler()
