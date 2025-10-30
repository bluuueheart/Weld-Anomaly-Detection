"""
诊断训练问题：检查参数冻结状态和梯度流

用于排查训练损失不下降的问题
"""
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import create_quadmodal_model
from src.dataset import WeldingDataset
from src.losses import SupConLoss
from torch.utils.data import DataLoader
from configs.dataset_config import *
from configs.model_config import *

def diagnose_model_parameters(model):
    """诊断模型参数状态"""
    print("=" * 70)
    print("模型参数诊断")
    print("=" * 70)
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    print("\n各模块参数统计:")
    print("-" * 70)
    
    # 按模块统计
    modules = {
        "video_encoder": model.video_encoder,
        "image_encoder": model.image_encoder,
        "audio_encoder": model.audio_encoder,
        "sensor_encoder": model.sensor_encoder,
        "fusion": model.fusion,
    }
    
    for name, module in modules.items():
        module_total = 0
        module_trainable = 0
        
        for param in module.parameters():
            num_params = param.numel()
            module_total += num_params
            if param.requires_grad:
                module_trainable += num_params
        
        total_params += module_total
        trainable_params += module_trainable
        frozen_params += (module_total - module_trainable)
        
        status = "✅ 可训练" if module_trainable > 0 else "❄️  完全冻结"
        print(f"{name:20s}: {module_total:>12,} 总参数 | {module_trainable:>12,} 可训练 | {status}")
    
    print("-" * 70)
    print(f"{'总计':20s}: {total_params:>12,} 总参数 | {trainable_params:>12,} 可训练")
    print(f"可训练比例: {trainable_params/total_params*100:.2f}%")
    print()
    
    # 检查是否有可训练参数
    if trainable_params == 0:
        print("⚠️  警告: 模型没有任何可训练参数！")
        print("   → 所有参数都被冻结，模型无法学习")
        return False
    elif trainable_params < total_params * 0.01:
        print("⚠️  警告: 可训练参数太少 (<1%)")
        print("   → 检查是否意外冻结了关键模块")
    else:
        print("✅ 参数状态正常")
    
    return True

def diagnose_gradient_flow(model, dataloader, criterion, device):
    """诊断梯度流"""
    print("=" * 70)
    print("梯度流诊断")
    print("=" * 70)
    
    model.train()
    
    # 获取一个batch
    batch = next(iter(dataloader))
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # 前向传播
    print("\n执行前向传播...")
    features = model(batch)
    print(f"  输出特征形状: {features.shape}")
    print(f"  输出范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
    
    # 计算损失
    loss_output = criterion(features, batch["label"])
    if isinstance(loss_output, dict):
        loss = loss_output['total']
    else:
        loss = loss_output
    
    print(f"  损失值: {loss.item():.4f}")
    
    # 反向传播
    print("\n执行反向传播...")
    loss.backward()
    
    # 检查梯度
    print("\n各模块梯度统计:")
    print("-" * 70)
    
    modules = {
        "video_encoder": model.video_encoder,
        "image_encoder": model.image_encoder,
        "audio_encoder": model.audio_encoder,
        "sensor_encoder": model.sensor_encoder,
        "fusion": model.fusion,
    }
    
    has_gradient = False
    for name, module in modules.items():
        grad_norm = 0.0
        param_count = 0
        params_with_grad = 0
        
        for param in module.parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
                    params_with_grad += 1
        
        if param_count > 0:
            grad_norm = grad_norm ** 0.5
            status = "✅ 有梯度" if params_with_grad > 0 else "❌ 无梯度"
            print(f"{name:20s}: {param_count:>4} 可训练参数 | {params_with_grad:>4} 有梯度 | 梯度范数: {grad_norm:>10.4f} | {status}")
            if params_with_grad > 0:
                has_gradient = True
        else:
            print(f"{name:20s}: [完全冻结，跳过]")
    
    print("-" * 70)
    
    if not has_gradient:
        print("\n❌ 严重问题: 没有任何模块收到梯度！")
        print("   可能原因:")
        print("   1. 所有可训练参数被意外冻结")
        print("   2. 计算图断开（detach 操作）")
        print("   3. 损失函数有问题")
        return False
    else:
        print("\n✅ 梯度流正常")
        return True

def check_optimizer_state(model, learning_rate=1e-4):
    """检查优化器配置"""
    print("=" * 70)
    print("优化器诊断")
    print("=" * 70)
    
    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        print("❌ 错误: 没有可训练参数传递给优化器！")
        return None
    
    print(f"\n可训练参数组: {len(trainable_params)} 个张量")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    print(f"优化器类型: AdamW")
    print(f"学习率: {learning_rate}")
    print(f"参数组数量: {len(optimizer.param_groups)}")
    
    total_opt_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print(f"优化器管理的参数总数: {total_opt_params:,}")
    
    print("\n✅ 优化器配置正常")
    return optimizer

def main():
    """主诊断函数"""
    print("\n" + "=" * 70)
    print("训练问题诊断工具")
    print("=" * 70)
    print()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 1. 加载模型
    print("加载模型...")
    model_config = {
        "VIDEO_ENCODER": VIDEO_ENCODER,
        "IMAGE_ENCODER": IMAGE_ENCODER,
        "AUDIO_ENCODER": AUDIO_ENCODER,
        "SENSOR_ENCODER": SENSOR_ENCODER,
        "FUSION": FUSION,
    }
    
    model = create_quadmodal_model(model_config, use_dummy=False)
    model = model.to(device)
    print()
    
    # 2. 诊断参数状态
    params_ok = diagnose_model_parameters(model)
    
    if not params_ok:
        print("\n" + "=" * 70)
        print("❌ 诊断失败: 参数配置有严重问题")
        print("=" * 70)
        return
    
    # 3. 创建数据加载器
    print("=" * 70)
    print("加载数据...")
    print("=" * 70)
    
    dataset = WeldingDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        split='train',
        video_length=VIDEO_LENGTH,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        sensor_length=SENSOR_LENGTH,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        dummy=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    print(f"数据集样本数: {len(dataset)}")
    print()
    
    # 4. 诊断梯度流
    criterion = SupConLoss(temperature=0.07)
    grad_ok = diagnose_gradient_flow(model, dataloader, criterion, device)
    
    if not grad_ok:
        print("\n" + "=" * 70)
        print("❌ 诊断失败: 梯度流有问题")
        print("=" * 70)
        return
    
    # 5. 检查优化器
    optimizer = check_optimizer_state(model)
    
    if optimizer is None:
        print("\n" + "=" * 70)
        print("❌ 诊断失败: 优化器配置有问题")
        print("=" * 70)
        return
    
    # 6. 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    
    if params_ok and grad_ok and optimizer is not None:
        print("\n✅ 所有检查通过！模型配置正常。")
        print("\n如果训练损失仍不下降，可能原因:")
        print("  1. 学习率太小（当前默认 1e-4）")
        print("  2. 数据集问题（标签错误、数据质量）")
        print("  3. 损失函数不适合当前任务")
        print("  4. 需要更长的 warmup 时间")
        print("\n建议:")
        print("  - 尝试增大学习率到 1e-3")
        print("  - 检查数据集标签分布")
        print("  - 观察更多 epoch 是否有变化")
    else:
        print("\n❌ 发现配置问题，请根据上述诊断修复")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
