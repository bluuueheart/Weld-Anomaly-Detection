# 问题回答：未冻结模块与 Dropout 配置

## Q: 哪些模块是未冻结的（可训练的）？

### ✅ 完全可训练（未冻结）

1. **Sensor Encoder** (完全可训练)
   - 全部参数都参与训练
   - 包含 Transformer 编码器（4层）
   - 当前 dropout = 0.4

2. **Fusion Module** (完全可训练)
   - 全部参数都参与训练
   - 包含交叉注意力、投影层、聚合层
   - 当前 dropout = 0.4

3. **编码器投影层** (部分可训练)
   - Video Encoder 的投影层（Linear 1024→1024）
   - Image Encoder 的投影层（Linear 768→768）
   - Audio Encoder 的投影层（Linear 768→768）
   - **新增 dropout = 0.3**（前后各一次）

### ❌ 已冻结（不可训练）

1. **Video Encoder Backbone** (V-JEPA)
   - `freeze_backbone: True`
   - 预训练参数不更新

2. **Image Encoder Backbone** (DINOv2)
   - `freeze_backbone: True`
   - 预训练参数不更新

3. **Audio Encoder Backbone** (AST)
   - `freeze_backbone: True`
   - 预训练参数不更新

---

## Q: 现在的 Dropout 是多少？

### 原配置（v2，过拟合）
```python
SENSOR_ENCODER.dropout = 0.2
FUSION.dropout = 0.2
编码器投影层 = 无 dropout ❌
```

### 新配置（v3，激进防过拟合）✅

#### 1. 配置文件
```python
# configs/model_config.py
SENSOR_ENCODER = {
    "dropout": 0.4,  # ⬆️ 从 0.2 提升
}

FUSION = {
    "dropout": 0.4,  # ⬆️ 从 0.2 提升
}
```

#### 2. 编码器投影层（新增）
```python
# src/models/video_encoder.py
self.projection = nn.Sequential(
    nn.Dropout(0.3),  # 新增
    nn.Linear(1024, 1024),
    nn.Dropout(0.3),  # 新增
)

# src/models/image_encoder.py  
self.projection = nn.Sequential(
    nn.Dropout(0.3),  # 新增
    nn.Linear(768, 768),
    nn.Dropout(0.3),  # 新增
)

# src/models/audio_encoder.py
self.projection = nn.Sequential(
    nn.Dropout(0.3),  # 新增
    nn.Linear(768, 768),
    nn.Dropout(0.3),  # 新增
)
```

#### 3. Fusion Module（强化）
```python
# src/models/fusion.py

# 模态投影（每个模态）
self.video_proj = nn.Sequential(
    nn.Dropout(0.4),  # 新增
    nn.Linear(video_dim, 512),
    nn.Dropout(0.4),  # 新增
)
# image/audio/sensor 同理

# 聚合层
self.aggregation = nn.Sequential(
    nn.Dropout(0.4),  # 新增（输入）
    nn.Linear(2048, 1024),
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Dropout(0.4),  # 原有（中间）
    nn.Linear(1024, 512),
    nn.Dropout(0.4),  # 新增（输出）
)

# 最终投影
self.output_proj = nn.Sequential(
    nn.Dropout(0.4),  # 新增
    nn.Linear(512, 512),
    nn.Dropout(0.4),  # 新增
)
```

---

## Dropout 层级统计

| 模块 | Dropout 层数 | Dropout 率 |
|------|-------------|-----------|
| Video 投影 | 2 | 0.3 |
| Image 投影 | 2 | 0.3 |
| Audio 投影 | 2 | 0.3 |
| Sensor Encoder | 内置（Transformer） | 0.4 |
| Fusion 模态投影 | 8 (4模态 × 2) | 0.4 |
| Fusion Cross-Attention | 4 (内置) | 0.4 |
| Fusion 聚合层 | 3 | 0.4 |
| Fusion 输出投影 | 2 | 0.4 |
| **总计** | **~20+ 层** | **0.3-0.4** |

---

## 参数统计

```python
总参数: 516M
可训练参数: 18M (3.5%)

可训练参数分布:
- Sensor Encoder: ~6M
- Fusion Module: ~10M
- 投影层: ~2M
```

---

## 正则化强度对比

### v2 配置（过拟合）
```
Learning Rate: 5e-5
Weight Decay: 1e-3
Dropout: 0.2（仅 Sensor/Fusion）
Temperature: 0.07
→ Gap: 1.83 ❌
```

### v3 配置（激进防过拟合）✅
```
Learning Rate: 3e-5      ⬇️ 40% 降低
Weight Decay: 5e-3       ⬆️ 5倍 增强
Dropout: 0.3-0.4         ⬆️ 2倍 + 全覆盖
Temperature: 0.10        ⬆️ 更平滑
→ 预期 Gap: < 0.5 ✅
```

---

## 下一步

```bash
# 在服务器上重新训练
python src/train.py

# 观察指标
# - Val Loss 应该跟随 Train Loss 下降
# - Gap 应该保持在 0.3-0.5 范围
# - 如果 Gap 仍然 > 1.0，考虑进一步降低 LR 到 2e-5
```

---

**总结**: 已在所有可训练模块添加 Dropout，并大幅增强正则化。预期 Gap 从 1.83 降到 < 0.5。
