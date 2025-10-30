# 训练优化总结 - 2025年10月20日

## 🎯 问题

训练 loss 完全不下降，即使调整学习率也无效：
- `lr=1e-4`: loss 卡在 3.4340
- `lr=1e-6`: loss 卡在 3.4531  
- `lr=1e-10`: loss 卡在 3.5419

## ✅ 已实施的解决方案

### 1. Linear Warmup（线性预热）

**目的**: 避免训练初期学习率过大导致不稳定

**配置** (`configs/train_config.py`):
```python
"lr_scheduler": "cosine_warmup",
"warmup_epochs": 10,           # 前 10 个 epoch 预热
"warmup_start_lr": 1e-7,       # 起始 LR
"learning_rate": 1e-4,         # 目标 LR
```

**效果**:
- Epoch 1: LR = 1e-7
- Epoch 5: LR = 5e-5  
- Epoch 10: LR = 1e-4
- Epoch 11+: Cosine 衰减到 1e-6

### 2. 特征归一化

**目的**: 稳定训练，提升对比学习效果

**实现** (`src/models/quadmodal_model.py`):
```python
# LayerNorm: 稳定特征分布
self.feature_norm = nn.LayerNorm(512)

# L2 归一化: 投影到单位超球面
fused_features = F.normalize(fused_features, p=2, dim=1)
```

**配置** (`configs/model_config.py`):
```python
FUSION = {
    ...
    "l2_normalize": True,  # 启用 L2 归一化
}
```

### 3. 增强调试工具

#### 调试模式训练
```bash
python src/train.py --debug
```

**输出示例（正常）**:
```
[DEBUG] Batch 0 labels - unique: [0, 1, 2, 3, 4, 5], counts: [3, 2, 4, 3, 2, 2]
[DEBUG] ✅ Good! Batch contains 6 different classes
```

**输出示例（异常）**:
```
[DEBUG] Batch 0 labels - unique: [0], counts: [32]
[DEBUG] ⚠️  WARNING: ALL 32 samples in batch are from class 0!
[DEBUG] ⚠️  StratifiedBatchSampler may be broken!
```

#### 采样器检查工具
```bash
python scripts/check_sampler.py
```

检查前 5 个批次是否正确混合类别。

## 📝 使用指南

### 步骤 1: 检查采样器

```bash
python scripts/check_sampler.py
```

**预期**: 每个批次包含 6 个不同类别  
**异常**: 批次只有单一类别 → 采样器有 bug

### 步骤 2: 启用调试训练

```bash
python src/train.py --debug
```

观察：
- ✅ Warmup LR 曲线: 1e-7 → 1e-4
- ✅ 批次类别混合: `[0,1,2,3,4,5]`
- ✅ 特征统计正常: 无 NaN/Inf

### 步骤 3: 正常训练

```bash
python src/train.py
```

**预期**:
- 前 10 epoch: Warmup 阶段，loss 可能略微震荡
- 10-20 epoch: Loss 开始明显下降
- 20+ epoch: 持续收敛

## 🔧 调试清单

如果 loss 仍然不降，按顺序检查：

### 1️⃣ 采样器问题
```bash
python scripts/check_sampler.py
```
- ❌ 所有批次只有类别 0 → 修复 `src/samplers.py`
- ✅ 批次混合多个类别 → 继续下一步

### 2️⃣ 数据集不平衡
检查类别分布：
```bash
python scripts/check_sampler.py
```
看输出：
```
类别分布:
  类别 0: 550 样本  ← 占 95%！
  类别 1: 10 样本
  类别 2: 8 样本
  ...
```
- ❌ 某类占 >90% → 数据集严重不平衡，需要重采样
- ✅ 类别分布均匀 → 继续下一步

### 3️⃣ 学习率问题
```bash
# 尝试调整学习率
python src/train.py --debug

# 观察前几个 epoch 的 loss 变化
```
- Loss 爆炸 (NaN) → LR 太大，降到 1e-5
- Loss 完全不动 → LR 太小，升到 5e-4
- Loss 缓慢下降 → ✅ 正常

### 4️⃣ 模型冻结问题
检查 `configs/model_config.py`:
```python
VIDEO_ENCODER = {
    "freeze_backbone": True,  # 如果为 True，视频编码器不会学习
}
```
- 所有编码器都冻结 → 只有融合层在学习，可能不够
- 建议: 至少解冻一个编码器（如 sensor）

### 5️⃣ 梯度流问题
```bash
python src/train.py --debug
```
看输出：
```
[DEBUG] Gradient check:
        Params with grad: 93
        Total grad norm: 5.638672
```
- `Params with grad: 0` → 模型完全冻结或梯度消失
- `Total grad norm: 0` → 梯度消失，检查 loss 计算

## 📊 预期训练曲线

### 正常情况
```
Epoch  1: Loss 3.54 (warmup, LR=1e-7)
Epoch  5: Loss 3.21 (warmup, LR=5e-5)
Epoch 10: Loss 2.87 (warmup结束, LR=1e-4)
Epoch 15: Loss 2.34 (开始收敛)
Epoch 20: Loss 1.89
Epoch 30: Loss 1.45
...
```

### 异常情况 1: 完全不动
```
Epoch  1: Loss 3.43
Epoch  5: Loss 3.43
Epoch 10: Loss 3.43
Epoch 20: Loss 3.43  ← 可能是采样器问题
```
→ 检查采样器和数据集分布

### 异常情况 2: 爆炸
```
Epoch  1: Loss 3.54
Epoch  2: Loss 12.34
Epoch  3: Loss NaN  ← LR 太大
```
→ 降低学习率或增加 warmup epochs

## 🔍 核心文件

| 文件 | 作用 |
|------|------|
| `configs/train_config.py` | 训练配置（LR, warmup, batch_size） |
| `configs/model_config.py` | 模型配置（归一化开关） |
| `src/train.py` | 训练循环（warmup逻辑，调试输出） |
| `src/models/quadmodal_model.py` | 模型定义（特征归一化层） |
| `src/samplers.py` | 批次采样器（确保类别混合） |
| `scripts/check_sampler.py` | 采样器验证工具 |

## 💡 快速诊断命令

```bash
# 1. 检查采样器（30秒）
python scripts/check_sampler.py

# 2. 调试训练 3 个 epoch（5分钟）
python src/train.py --debug --num-epochs 3

# 3. 正常训练（如果前两步正常）
python src/train.py
```

## ✅ 成功标志

训练成功的标志：
- ✅ Warmup 阶段 LR 正常递增
- ✅ 批次包含 6 个不同类别
- ✅ Loss 在 10-20 epoch 开始下降
- ✅ 无 NaN/Inf 值
- ✅ 梯度范数正常（1-10）

---

**创建时间**: 2025年10月20日  
**更新日志**: docs/PROGRESS_QUADMODAL.md (Section 9)
