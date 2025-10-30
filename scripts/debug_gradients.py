"""
紧急调试脚本：检查模型参数更新

直接在训练循环中插入此代码段来诊断问题
"""
import torch

def debug_model_gradients_and_params(model, optimizer, step_name=""):
    """
    打印模型参数和梯度的详细信息
    
    Args:
        model: PyTorch 模型
        optimizer: 优化器
        step_name: 步骤名称（before_backward, after_backward, after_step等）
    """
    print(f"\n{'='*70}")
    print(f"调试检查点: {step_name}")
    print(f"{'='*70}")
    
    total_params = 0
    trainable_params = 0
    params_with_grad = 0
    total_grad_norm = 0.0
    
    # 按模块统计
    if hasattr(model, 'video_encoder'):
        modules = {
            "video_encoder": model.video_encoder,
            "image_encoder": model.image_encoder,
            "audio_encoder": model.audio_encoder,
            "sensor_encoder": model.sensor_encoder,
            "fusion": model.fusion,
        }
    else:
        modules = {"model": model}
    
    for module_name, module in modules.items():
        module_total = 0
        module_trainable = 0
        module_with_grad = 0
        module_grad_norm = 0.0
        
        for name, param in module.named_parameters():
            num_params = param.numel()
            module_total += num_params
            total_params += num_params
            
            if param.requires_grad:
                module_trainable += num_params
                trainable_params += num_params
                
                if param.grad is not None:
                    module_with_grad += num_params
                    params_with_grad += num_params
                    grad_norm = param.grad.norm().item()
                    module_grad_norm += grad_norm ** 2
                    total_grad_norm += grad_norm ** 2
        
        module_grad_norm = module_grad_norm ** 0.5
        
        status = "✅" if module_with_grad > 0 else "❌"
        print(f"\n{module_name}:")
        print(f"  总参数: {module_total:,}")
        print(f"  可训练: {module_trainable:,}")
        print(f"  有梯度: {module_with_grad:,} {status}")
        if module_with_grad > 0:
            print(f"  梯度范数: {module_grad_norm:.6f}")
    
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"\n{'='*70}")
    print(f"总计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  有梯度参数: {params_with_grad:,} ({params_with_grad/trainable_params*100:.2f}% of trainable)")
    print(f"  总梯度范数: {total_grad_norm:.6f}")
    
    # 检查优化器状态
    print(f"\n优化器状态:")
    print(f"  参数组数: {len(optimizer.param_groups)}")
    print(f"  学习率: {optimizer.param_groups[0]['lr']:.6e}")
    
    # 检查参数是否在更新
    if step_name == "after_step":
        # 随机抽样几个参数检查值
        sample_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'fusion' in name:
                sample_params.append((name, param))
                if len(sample_params) >= 3:
                    break
        
        if sample_params:
            print(f"\n样本参数值（fusion 模块）:")
            for name, param in sample_params:
                print(f"  {name[:50]}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
    
    print(f"{'='*70}\n")


def debug_loss_computation(features, labels, criterion, batch_idx=0):
    """
    调试损失计算过程
    
    Args:
        features: 模型输出特征 (B, D)
        labels: 标签 (B,)
        criterion: 损失函数
        batch_idx: batch 索引
    """
    print(f"\n{'='*70}")
    print(f"损失计算调试 (Batch {batch_idx})")
    print(f"{'='*70}")
    
    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"特征 requires_grad: {features.requires_grad}")
    print(f"特征范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
    print(f"特征均值: {features.mean().item():.4f}, 标准差: {features.std().item():.4f}")
    
    # 检查标签分布
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"\n标签分布:")
    for label, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
        print(f"  类别 {label}: {count} 个样本")
    
    # 检查是否有重复标签（正样本对）
    num_pos_pairs = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                num_pos_pairs += 1
    
    print(f"\n正样本对数量: {num_pos_pairs}")
    if num_pos_pairs == 0:
        print("⚠️  警告: 没有正样本对！SupConLoss 将无法计算有效梯度")
    
    # 计算损失
    loss_output = criterion(features, labels)
    if isinstance(loss_output, dict):
        loss = loss_output['total']
    else:
        loss = loss_output
    
    print(f"\n损失值: {loss.item():.6f}")
    print(f"损失 requires_grad: {loss.requires_grad}")
    
    print(f"{'='*70}\n")
    
    return loss


# 使用示例（插入到训练循环中）:
"""
# 在训练循环的 train_epoch 方法中添加:

def train_epoch(self, epoch: int):
    self.model.train()
    
    for batch_idx, batch in enumerate(self.train_loader):
        # ... 数据移动到设备 ...
        
        # DEBUG: 检查初始参数状态（第一个batch）
        if batch_idx == 0:
            from scripts.debug_gradients import debug_model_gradients_and_params
            debug_model_gradients_and_params(self.model, self.optimizer, "before_forward")
        
        # Forward pass
        features = self.model(batch)
        
        # DEBUG: 检查损失计算（第一个batch）
        if batch_idx == 0:
            from scripts.debug_gradients import debug_loss_computation
            loss = debug_loss_computation(features, batch["label"], self.criterion, batch_idx)
        else:
            loss_output = self.criterion(features, batch["label"])
            if isinstance(loss_output, dict):
                loss = loss_output['total']
            else:
                loss = loss_output
        
        # Backward pass
        self.optimizer.zero_grad()
        
        # DEBUG: 检查梯度前状态
        if batch_idx == 0:
            debug_model_gradients_and_params(self.model, self.optimizer, "after_zero_grad")
        
        loss.backward()
        
        # DEBUG: 检查梯度后状态
        if batch_idx == 0:
            debug_model_gradients_and_params(self.model, self.optimizer, "after_backward")
        
        # Gradient clipping
        if self.config.get("gradient_clip", 0) > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["gradient_clip"]
            )
        
        self.optimizer.step()
        
        # DEBUG: 检查参数更新后状态
        if batch_idx == 0:
            debug_model_gradients_and_params(self.model, self.optimizer, "after_step")
            
            # 如果这里仍然显示没有梯度或参数没变化，说明有严重问题
            print("\\n⚠️  如果上面显示'有梯度参数: 0'，说明：")
            print("   1. 模型参数被完全冻结")
            print("   2. 或者计算图被断开（detach）")
            print("   3. 或者损失函数返回了常量")
        
        # 正常训练继续...
"""
