# 训练优化记录 - 2025年10月20日

## 问题诊断

### 原始问题
训练集只有单一类别（Good，标签 0），导致：
- Loss 恒定在 3.43（SupConLoss 数学上的极限值）
- 采样器无法混合类别
- 训练完全无效

### 解决方案
重新划分数据集（80/20 训练/测试比例），确保训练集包含所有 12 个类别：
- **训练集**: 3231 样本，12 个类别
- **测试集**: 809 样本，12 个类别

## 训练观察

### 初始配置（过拟合）
```python
"learning_rate": 1e-4
"weight_decay": 1e-4
"dropout": 0.1  # Fusion & Sensor
```

**观察结果**（前 4 个 epoch）:
- ✅ 训练 Loss 正常下降: 2.95 → 2.31 → 2.09 → 1.85
- ❌ 验证 Loss 上升: 3.37 → 3.29 → 3.57 → 3.36
- **诊断**: 模型过拟合训练集

### 优化后配置（增强正则化）
```python
"learning_rate": 5e-5      # 降低峰值学习率（从 1e-4）
"weight_decay": 1e-3       # 增强权重衰减（从 1e-4）
"dropout": 0.2             # 增强 Dropout（从 0.1）
"warmup_epochs": 10        # 保持 warmup
"warmup_start_lr": 1e-7    # 保持起始 LR
```

**正则化策略**:
1. **降低学习率**: 5e-5（防止过快拟合训练集）
2. **增强权重衰减**: 1e-3（L2 正则化）
3. **增强 Dropout**: 
   - Sensor Encoder: 0.1 → 0.2
   - Fusion Module: 0.1 → 0.2
4. **更新类别数**: NUM_CLASSES = 12（匹配实际数据集）

## 预期效果

### 训练曲线（期望）
```
Epoch  1: Train 2.95, Val 3.37  (Warmup 开始)
Epoch  5: Train 2.10, Val 2.80  (Warmup 中)
Epoch 10: Train 1.65, Val 2.40  (Warmup 结束)
Epoch 20: Train 1.20, Val 1.85  (开始收敛)
Epoch 40: Train 0.85, Val 1.45  (持续改善)
Epoch 60: Train 0.60, Val 1.25  (接近最优)
```

**关键指标**:
- ✅ 训练 Loss 稳定下降
- ✅ 验证 Loss 跟随下降（不上升）
- ✅ Train/Val Gap < 0.5（泛化良好）

### 可能的调整

如果验证 Loss 仍然上升：
1. **进一步降低学习率**: 3e-5
2. **增加 weight_decay**: 5e-3
3. **增加 Dropout**: 0.3
4. **数据增强**: 添加更强的数据增强策略

如果训练过慢：
1. **适当提高学习率**: 7e-5
2. **减少 warmup**: 5 epochs
3. **调整 batch_size**: 64（如果显存允许）

## 文件修改

| 文件 | 修改内容 | 原值 → 新值 |
|------|---------|------------|
| `configs/train_config.py` | learning_rate | 1e-4 → 5e-5 |
| `configs/train_config.py` | weight_decay | 1e-4 → 1e-3 |
| `configs/model_config.py` | SENSOR_ENCODER.dropout | 0.1 → 0.2 |
| `configs/model_config.py` | FUSION.dropout | 0.1 → 0.2 |
| `configs/model_config.py` | NUM_CLASSES | 6 → 12 |
| `configs/manifest.csv` | 重新划分 | 80/20 分割 |

## 运行命令

```bash
# 在服务器上运行（已优化配置）
bash scripts/train.sh

# 或直接运行
python src/train.py

# 调试模式（查看详细信息）
python src/train.py --debug
```

## 监控建议

关注以下指标：
1. **Train/Val Loss Gap**: 应该 < 0.5
2. **Val Loss 趋势**: 应该持续下降
3. **Early Stopping**: 如果 10 个 epoch 无改善则停止
4. **学习率曲线**: Warmup 应该平滑

---

**更新时间**: 2025年10月20日 19:50  
**状态**: ✅ 配置已优化，等待服务器训练验证
