# 防过拟合优化 v3 - 2025年10月21日

## 问题分析

### 训练观察（Epoch 22 停止）
```
Training Loss: ~1.4    ⬇️ 持续下降
Validation Loss: ~3.23 ➡️ 几乎不动（Epoch 8: 3.2471 → Epoch 22: 3.2319）
Gap: ~1.83             ❌ 严重过拟合
```

**诊断**: 
- 模型在训练集上学习过度，但无法泛化到验证集
- 14 个 epoch 仅改善 0.015，说明验证集几乎没有学习
- 之前的正则化（dropout=0.2, weight_decay=1e-3）不够强

## 全面优化策略

### 1. 超参数调整

#### configs/train_config.py
```python
"learning_rate": 3e-5          # ⬇️ 从 5e-5（更保守的学习）
"weight_decay": 5e-3           # ⬆️ 从 1e-3（5倍L2正则化）
"temperature": 0.10            # ⬆️ 从 0.07（更平滑的对比学习）
"gradient_clip": 0.5           # ⬇️ 从 1.0（更严格的梯度控制）
"early_stopping_patience": 15  # ⬆️ 从 10（给更多改善机会）
```

**效果**:
- **降低 LR**: 防止模型过快拟合训练集噪声
- **增强 weight_decay**: 强制参数保持小值，增强泛化
- **提高 temperature**: 让 SupConLoss 对相似度不那么敏感，减少过拟合
- **收紧梯度**: 防止大幅参数更新
- **延长 patience**: 验证集改善可能更慢

#### configs/model_config.py
```python
SENSOR_ENCODER = {
    "dropout": 0.4,  # ⬆️ 从 0.2（2倍dropout）
}

FUSION = {
    "dropout": 0.4,  # ⬆️ 从 0.2（2倍dropout）
}
```

### 2. 架构级 Dropout 强化

#### 未冻结模块及 Dropout 位置

**可训练模块**（未冻结）:
1. ✅ **Sensor Encoder** (完全可训练)
   - Transformer 层: dropout=0.4
   - 投影层: 内置 dropout

2. ✅ **Fusion Module** (完全可训练)
   - 模态投影: dropout=0.4（前后各一次）
   - Cross-Attention: dropout=0.4
   - 聚合层: dropout=0.4（输入、中间、输出）
   - 最终投影: dropout=0.4（前后各一次）

3. ✅ **Video Encoder 投影层** (可训练)
   ```python
   self.projection = nn.Sequential(
       nn.Dropout(0.3),        # 新增
       nn.Linear(1024, 1024),
       nn.Dropout(0.3),        # 新增
   )
   ```

4. ✅ **Image Encoder 投影层** (可训练)
   ```python
   self.projection = nn.Sequential(
       nn.Dropout(0.3),        # 新增
       nn.Linear(768, 768),
       nn.Dropout(0.3),        # 新增
   )
   ```

5. ✅ **Audio Encoder 投影层** (可训练)
   ```python
   self.projection = nn.Sequential(
       nn.Dropout(0.3),        # 新增
       nn.Linear(768, 768),
       nn.Dropout(0.3),        # 新增
   )
   ```

**冻结模块**（freeze_backbone=True）:
- ❌ Video Encoder (V-JEPA backbone) - 冻结，不可训练
- ❌ Image Encoder (DINOv2 backbone) - 冻结，不可训练
- ❌ Audio Encoder (AST backbone) - 冻结，不可训练

### 3. Dropout 应用层级

```
输入数据
    ↓
[冻结] V-JEPA Backbone
    ↓
Dropout(0.3) ← 新增
    ↓
Linear(1024→1024)
    ↓
Dropout(0.3) ← 新增
    ↓
[冻结] DINOv2/AST Backbone (同理)
    ↓
Dropout(0.3) × 4 模态 ← 新增
    ↓
Fusion Module:
    Dropout(0.4) ← 投影前
    Linear(各维度→512)
    Dropout(0.4) ← 投影后
    ↓
    Cross-Attention (内置 dropout=0.4)
    ↓
    Dropout(0.4) ← 聚合输入
    MLP(2048→1024→512)
    Dropout(0.4) ← 聚合中间
    Dropout(0.4) ← 聚合输出
    ↓
    Dropout(0.4) ← 最终投影前
    Linear(512→512)
    Dropout(0.4) ← 最终投影后
    ↓
LayerNorm + L2 Normalize
    ↓
输出特征 (512维)
```

**总 Dropout 层数**: ~20+ 层

## 修改文件清单

| 文件 | 修改内容 | 新增 Dropout 层 |
|------|---------|----------------|
| `configs/train_config.py` | LR, weight_decay, temperature, gradient_clip | - |
| `configs/model_config.py` | SENSOR/FUSION dropout: 0.2→0.4 | - |
| `src/models/video_encoder.py` | 投影层添加 dropout=0.3 | 2 |
| `src/models/image_encoder.py` | 投影层添加 dropout=0.3 | 2 |
| `src/models/audio_encoder.py` | 投影层添加 dropout=0.3 | 2 |
| `src/models/fusion.py` | 投影、聚合、输出全面添加 dropout | 12 |

**总计新增**: 18 个 Dropout 层

## 预期效果

### 理想训练曲线
```
Epoch  5: Train 2.80, Val 2.95  Gap=0.15 ✅
Epoch 10: Train 2.35, Val 2.50  Gap=0.15 ✅
Epoch 20: Train 1.85, Val 2.05  Gap=0.20 ✅
Epoch 30: Train 1.50, Val 1.75  Gap=0.25 ✅
Epoch 50: Train 1.20, Val 1.50  Gap=0.30 ✅
```

**关键指标**:
- ✅ Val Loss 跟随 Train Loss 下降
- ✅ Gap 保持在 0.3 以内
- ✅ Val Loss 持续改善（不卡住）

### 可能的调整

**如果仍然过拟合** (Val Loss 不降):
```python
"learning_rate": 2e-5      # 进一步降低
"weight_decay": 1e-2       # 更强正则化
"dropout": 0.5             # 更高 dropout（极限）
"temperature": 0.15        # 更平滑对比
```

**如果欠拟合** (Train Loss 很高):
```python
"learning_rate": 5e-5      # 适当提高
"weight_decay": 2e-3       # 适当放松
"dropout": 0.3             # 降低 dropout
"temperature": 0.08        # 更严格对比
```

**如果训练太慢**:
```python
"warmup_epochs": 5         # 减少预热
"batch_size": 48           # 增大批次（需要显存）
```

## 运行命令

```bash
# 在服务器上重新训练
bash scripts/train.sh

# 或直接运行
python src/train.py

# 后台运行并保存日志
nohup python src/train.py > training_v3.log 2>&1 &
tail -f training_v3.log
```

## 监控建议

**每 5 个 epoch 检查**:
1. **Train/Val Gap**: 应该 < 0.5（理想 < 0.3）
2. **Val Loss 趋势**: 应该持续下降（不卡住）
3. **学习率**: Warmup 10 epoch 后应该开始衰减
4. **Early Stopping**: 15 epoch 无改善才停止

**异常信号**:
- ⚠️ Gap > 1.0: 严重过拟合
- ⚠️ Val Loss 10 epoch 无变化: 正则化过强或 LR 太小
- ⚠️ Train Loss 不降: 模型容量不足或 LR 太小

## 正则化总结

### 当前正则化策略（由弱到强）

| 方法 | 强度 | 位置 |
|------|------|------|
| L2 Normalization | ✅ | 输出特征 |
| LayerNorm | ✅ | 各模块输出 |
| Dropout (0.3) | ⚡⚡ | 编码器投影层 |
| Dropout (0.4) | ⚡⚡⚡ | Sensor + Fusion |
| Weight Decay (5e-3) | ⚡⚡⚡ | 所有参数 |
| Gradient Clip (0.5) | ⚡⚡ | 梯度 |
| Low LR (3e-5) | ⚡⚡⚡ | 优化器 |
| High Temperature (0.10) | ⚡⚡ | SupConLoss |

**总体强度**: ⚡⚡⚡⚡ (极强正则化)

---

**更新时间**: 2025年10月21日  
**版本**: v3（激进防过拟合）  
**状态**: ✅ 配置已优化，等待服务器验证  
**预期**: Val Loss 应该开始下降，Gap < 0.5
