# 训练日志与可视化

## 📝 训练日志

训练过程中会自动保存日志到 `logs/training_log.json` (或配置的 LOG_DIR 路径)

### 日志内容

```json
{
  "train": [
    {
      "loss": 2.95,
      "time": 45.2,
      "lr": 1e-6
    },
    ...
  ],
  "val": [
    {
      "loss": 3.40,
      "time": 12.3
    },
    ...
  ],
  "config": { ... },
  "best_metric": 2.94
}
```

### 实时保存逻辑

**✅ 每个epoch结束后立即保存**:
- 训练损失: 每个epoch结束后保存
- 验证损失: 每 `val_interval` 个epoch保存一次 (默认=1，即每个epoch)
- 完整配置: 包含所有训练超参数
- 最佳指标: 实时更新最佳验证损失

**保存时机**:
1. 每个训练epoch结束时调用 `_save_logs()`
2. 训练完全结束时再次保存 (确保完整性)
3. 即使训练中断，已完成的epoch数据也会保留

### 日志路径配置

```python
# configs/train_config.py
LOG_DIR = "/root/autodl-tmp/outputs/logs"  # 服务器路径
# 或
LOG_DIR = "logs"  # 本地相对路径
```

---

## 📊 绘制损失曲线

### 基础用法

```bash
# 使用默认路径 (自动查找日志文件)
python scripts/plot_loss.py

# 或使用 shell 脚本
bash scripts/plot_loss.sh
```

### 自定义参数

```bash
# 指定日志文件路径
python scripts/plot_loss.py --log /root/autodl-tmp/outputs/logs/training_log.json

# 指定输出目录
python scripts/plot_loss.py --output my_plots

# 生成详细分析图
python scripts/plot_loss.py --detailed
```

### 自动日志查找

脚本会按以下优先级自动查找日志文件:
1. `/root/autodl-tmp/outputs/logs/training_log.json` (服务器路径)
2. `logs/training_log.json` (本地相对路径)
3. `outputs/logs/training_log.json` (备用路径)

### 输出文件

脚本会生成以下文件到 `outputs/` 目录:

1. **`loss_curves.png`** - 基础损失曲线 (高清PNG)
2. **`loss_curves.pdf`** - 矢量图版本 (适合论文)
3. **`training_analysis.png`** - 详细分析dashboard (使用 `--detailed`)

---

## 📈 生成的图表

### 1. 基础损失曲线 (`loss_curves.png`)

包含:
- 训练损失曲线 (蓝色)
- 验证损失曲线 (橙色)
- 最佳验证点标记 (绿色星标)
- 最佳epoch的垂直虚线

**示例输出**:
```
✅ 损失曲线已保存到: outputs/loss_curves.png
✅ PDF版本已保存到: outputs/loss_curves.pdf

📊 训练统计:
  总Epoch数: 17
  初始训练损失: 2.9500
  最终训练损失: 1.4200
  训练损失降幅: 1.5300

  初始验证损失: 3.4000
  最终验证损失: 2.9700
  最佳验证损失: 2.9400 (Epoch 9)
```

### 2. 详细分析图 (`training_analysis.png`, 需要 `--detailed`)

包含4个子图:
1. **Loss曲线** - 训练和验证损失
2. **Epoch时间** - 每个epoch的训练耗时
3. **损失变化率** - 每个epoch的损失变化 (负值=改善)
4. **验证改进点** - 标记所有验证损失改善的epoch

---

## 🔍 实时监控

### 训练中查看损失

训练过程中会实时输出:

```
Epoch 1/100
----------------------------------------------------------------------
  [  1][ 10/101] Loss: 2.9523 | Avg: 2.9501 | LR: 1.00e-06
  [  1][ 20/101] Loss: 2.9345 | Avg: 2.9478 | LR: 1.00e-06
  ...
  Training Loss: 2.9501
  Validation Loss: 3.4012
  ✅ Saved best model (epoch 1)
  Epoch time: 45.2s

Epoch 2/100
----------------------------------------------------------------------
...
```

### 关键指标

- **Loss**: 当前batch损失
- **Avg**: 当前epoch平均损失
- **LR**: 当前学习率
- **Epoch time**: 单epoch耗时

---

## 💡 使用场景

### 场景1: 训练中实时查看

```bash
# 在另一个终端运行
watch -n 5 python scripts/plot_loss.py
```

每5秒自动刷新图表(Linux/Mac)

### 场景2: 训练完成后分析

```bash
# 生成完整分析
python scripts/plot_loss.py --detailed

# 查看输出图片
ls outputs/
# loss_curves.png
# loss_curves.pdf
# training_analysis.png
```

### 场景3: 对比多次训练

```bash
# 保存不同训练的图表
python scripts/plot_loss.py --log logs/run1/training_log.json --output plots/run1
python scripts/plot_loss.py --log logs/run2/training_log.json --output plots/run2
```

---

## 🛠️ 依赖

绘图脚本需要 matplotlib:

```bash
pip install matplotlib
```

如果在服务器(无显示器)上运行,脚本已自动使用 `Agg` backend,无需额外配置。

---

## 📁 文件结构

```
Weld-Anomaly-Detection/
├── logs/
│   └── training_log.json          # 训练日志 (自动生成)
├── outputs/
│   ├── loss_curves.png            # 损失曲线图
│   ├── loss_curves.pdf            # PDF版本
│   └── training_analysis.png      # 详细分析 (--detailed)
└── scripts/
    ├── plot_loss.py               # 绘图脚本
    └── plot_loss.sh               # Shell包装脚本
```

---

## 🐛 故障排查

### 问题1: 找不到 training_log.json

**原因**: 日志文件在不同路径，或训练尚未开始

**自动查找**: 脚本会自动查找常见路径:
- `/root/autodl-tmp/outputs/logs/training_log.json` (服务器)
- `logs/training_log.json` (本地)
- `outputs/logs/training_log.json` (备用)

**手动指定**:
```bash
python scripts/plot_loss.py --log /root/autodl-tmp/outputs/logs/training_log.json
```

**解决**:
```bash
# 检查日志是否存在
bash scripts/check_log.sh

# 如果不存在，启动训练
bash scripts/train.sh
```

### 问题2: 图表为空

**原因**: 日志文件损坏或格式错误

**解决**:
```bash
# 检查日志内容
cat logs/training_log.json | head -n 20

# 验证JSON格式
python -c "import json; json.load(open('logs/training_log.json'))"
```

### 问题3: matplotlib 导入错误

**原因**: 未安装 matplotlib

**解决**:
```bash
pip install matplotlib
```

---

## 📊 统计信息说明

### 训练损失降幅
```
训练损失降幅 = 初始训练损失 - 最终训练损失
```

**解读**:
- 大降幅(>1.0): 模型学习良好
- 小降幅(<0.5): 可能欠拟合或学习率过低

### 最佳验证点

标记验证损失最低的epoch。

**V4期望**: Epoch 9-12 之间

### 损失变化率

展示每个epoch的损失变化:
- **负值**: 损失下降 (✅ 好)
- **正值**: 损失上升 (⚠️ 过拟合信号)
- **接近0**: 收敛或停滞

---

## 🎨 自定义绘图

如需自定义图表样式,编辑 `scripts/plot_loss.py`:

```python
# 修改颜色
color='#1f77b4'  # 训练损失颜色

# 修改图表大小
figsize=(12, 6)  # 宽x高(英寸)

# 修改DPI(分辨率)
dpi=300  # 默认300,可调至600以获得更高清图片
```

---

## 📚 更多信息

- **训练配置**: `configs/train_config.py`
- **训练脚本**: `src/train.py`
- **完整文档**: `docs/QUICKSTART.md`
