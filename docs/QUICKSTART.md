# 快速开始指南 (Quick Start Guide)

> **项目**: 四模态焊接缺陷检测 - 基于监督对比学习的深度融合网络  
> **更新时间**: 2025年10月10日

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [环境检查](#2-环境检查)
3. [模块测试](#3-模块测试)
4. [完整测试](#4-完整测试)
5. [训练模型](#5-训练模型)
6. [预期输出](#6-预期输出)
7. [故障排查](#7-故障排查)

---

## 1. 环境准备

### 1.1 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n weld_sota python=3.10
conda activate weld_sota
```

### 1.2 安装 PyTorch

**根据您的 CUDA 版本选择合适的命令:**

```bash
# CUDA 11.8 (推荐)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本 (仅用于测试)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 1.3 安装项目依赖

```bash
# 模型加载与处理
pip install transformers>=4.30.0
pip install timm>=0.9.0

# 数据处理
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
pip install librosa>=0.10.0
pip install opencv-python

# 实验辅助工具
pip install einops
pip install tqdm
pip install numpy
```

**可选 (用于实验跟踪):**
```bash
pip install wandb  # Weights & Biases
pip install tensorboard  # TensorBoard
```

### 1.4 下载预训练模型 (可选)

**如果有网络连接的机器:**

```bash
# 安装 git-lfs
git lfs install

# 创建模型文件夹
mkdir -p models
cd models

# 下载 V-JEPA (视频编码器)
git clone https://huggingface.co/facebook/vjepa2-vitl-fpc64-256

# 下载 DINOv2 (图片编码器)
git clone https://huggingface.co/facebook/dinov2-base

# 下载 AST (音频编码器)
git clone https://huggingface.co/MIT/ast-finetuned-audioset-14-14-0.443

cd ..
```

**注意**: 如果没有预训练模型,代码会自动使用 Dummy 版本进行测试。

---

## 2. 环境检查

### 2.1 检查 PyTorch 安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出:**
```
PyTorch: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
```

### 2.2 检查关键依赖

```bash
python -c "import transformers; import timm; import pandas; import librosa; print('✅ All dependencies installed')"
```

**预期输出:**
```
✅ All dependencies installed
```

### 2.3 检查项目结构

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.models import QuadModalSOTAModel; print('✅ Project imports working')"
```

**预期输出:**
```
✅ Project imports working
```

---

## 3. 模块测试

按照以下顺序逐个测试各个模块,确保每个组件正常工作。

### 3.1 测试数据加载 (Step 1)

**运行测试:**
```bash
bash scripts/test_dataset.sh
```

**预期输出:**
```
======================================================================
Testing WeldingDataset (Dummy Mode)
======================================================================

Configuration:
  Data root: Data
  Video length: 32
  Audio sample rate: 16000
  Audio duration: 2.0
  Sensor length: 256
  Image size: 224
  Number of angles: 5
  Dummy mode: True

Dataset created successfully
  Total samples: 100
  Sample keys: dict_keys(['video', 'post_weld_images', 'audio', 'sensor', 'label', 'sample_id'])

Single sample shapes:
  ✅ Video: (32, 3, 224, 224)
  ✅ Post-weld images: (5, 3, 224, 224)
  ✅ Audio: (1, 128, 256)
  ✅ Sensor: (256, 6)
  ✅ Label: scalar

Batch loading test:
  Batch size: 4
  ✅ Video batch: (4, 32, 3, 224, 224)
  ✅ Image batch: (4, 5, 3, 224, 224)
  ✅ Audio batch: (4, 1, 128, 256)
  ✅ Sensor batch: (4, 256, 6)
  ✅ Label batch: (4,)

✅ All dataset tests passed!
```

**结果保存**: 无 (纯测试,不保存)

---

### 3.2 测试编码器 (Step 2)

**运行测试:**
```bash
bash scripts/test_encoders.sh
```

**预期输出:**
```
======================================================================
Testing VideoEncoder (Dummy Mode)
======================================================================
  Input: (2, 32, 3, 224, 224)
  Output: (2, 8, 1024)
  Parameters: 1,234,567
  ✅ VideoEncoder test passed

======================================================================
Testing ImageEncoder (Dummy Mode)
======================================================================
  Input: (2, 5, 3, 224, 224)
  Output: (2, 5, 768)
  Parameters: 987,654
  ✅ ImageEncoder test passed

======================================================================
Testing AudioEncoder (Dummy Mode)
======================================================================
  Input: (2, 1, 128, 256)
  Output: (2, 12, 768)
  Parameters: 876,543
  ✅ AudioEncoder test passed

======================================================================
Testing SensorEncoder
======================================================================
  Input: (2, 256, 6)
  Output: (2, 256, 256)
  Parameters: 654,321
  ✅ SensorEncoder test passed

======================================================================
Testing Gradients
======================================================================
  ✅ All gradients valid (no NaN/Inf)

✅ ALL ENCODER TESTS PASSED!
```

**结果保存**: 无 (纯测试,不保存)

---

### 3.3 测试融合模块 (Step 3)

**运行测试:**
```bash
bash scripts/test_fusion.sh
```

**预期输出:**
```
======================================================================
Testing CrossAttentionFusionModule
======================================================================

Configuration:
  Batch size: 4
  Video: seq_len=8, dim=1024
  Image: seq_len=5, dim=768
  Audio: seq_len=12, dim=768
  Sensor: seq_len=256, dim=256
  Hidden dim: 512

Model parameters:
  Total: 4,234,240
  Trainable: 4,234,240

Forward pass (without attention)...
  Output shape: (4, 512)
  Output range: [-2.3456, 3.1234]
  ✅ Output shape correct and values valid

Forward pass (with attention)...
  Output shape: (4, 512)
  Attention weights returned for: ['video', 'image', 'audio', 'sensor']
    video: (4, 4, 8)
    image: (4, 4, 5)
    audio: (4, 4, 12)
    sensor: (4, 4, 256)
  ✅ Attention weights shape correct

Testing gradient flow...
  ✅ All gradients valid

✅ CrossAttentionFusionModule test passed!

======================================================================
Testing DummyCrossAttentionFusion (Lightweight)
======================================================================
  Model parameters: 1,573,376
  Output shape: (4, 512)
  ✅ DummyCrossAttentionFusion test passed!

======================================================================
Testing Fusion with Different Batch Sizes
======================================================================
  Batch size  1: (1, 512) ✅
  Batch size  2: (2, 512) ✅
  Batch size  8: (8, 512) ✅
  Batch size 16: (16, 512) ✅

✅ All batch sizes passed!

======================================================================
✅ ALL FUSION TESTS PASSED!
======================================================================
```

**结果保存**: 无 (纯测试,不保存)

---

### 3.4 测试完整模型 (Step 4)

**运行测试:**
```bash
bash scripts/test_model.sh
```

**预期输出:**
```
======================================================================
Testing QuadModalSOTAModel (Dummy Encoders)
======================================================================

Model Configuration:
  Total parameters: 8,567,890
  Trainable parameters: 8,567,890
  Output dimension: 512

Input shapes:
  video: (4, 32, 3, 224, 224)
  post_weld_images: (4, 5, 3, 224, 224)
  audio: (4, 1, 128, 256)
  sensor: (4, 256, 6)

Forward pass (without attention)...
  Output shape: (4, 512)
  Output range: [-1.2345, 2.3456]
  ✅ Forward pass successful

Forward pass (with attention)...
  Output shape: (4, 512)
  Attention keys: ['video', 'image', 'audio', 'sensor']
  ✅ Attention weights returned

Testing gradient flow...
  ✅ All gradients valid

✅ QuadModalSOTAModel (Dummy) test passed!

======================================================================
Testing create_quadmodal_model Factory
======================================================================
  Model created via factory:
  Output dimension: 512
  Parameters: 8,567,890
  Output shape: (2, 512)

✅ Factory function test passed!

======================================================================
Testing Encoder Freezing
======================================================================
  Initial trainable parameters: 8,567,890
  After freezing encoders: 1,234,567
  After unfreezing encoders: 8,567,890

✅ Encoder freezing test passed!

======================================================================
Testing Different Batch Sizes
======================================================================
  Batch size  1: (1, 512) ✅
  Batch size  2: (2, 512) ✅
  Batch size  4: (4, 512) ✅
  Batch size  8: (8, 512) ✅

✅ All batch sizes passed!

======================================================================
Testing with Real DataLoader
======================================================================
  DataLoader batch shapes:
    video: (2, 32, 3, 224, 224)
    post_weld_images: (2, 5, 3, 224, 224)
    audio: (2, 1, 128, 256)
    sensor: (2, 256, 6)
    labels: (2,)
  
  Model output shape: (2, 512)

✅ DataLoader integration test passed!

======================================================================
✅ ALL QUADMODAL MODEL TESTS PASSED!
======================================================================
```

**结果保存**: 无 (纯测试,不保存)

---

### 3.5 测试损失函数 (Step 5)

**运行测试:**
```bash
bash scripts/test_losses.sh
```

**预期输出:**
```
======================================================================
LOSS FUNCTIONS TEST
======================================================================

Testing SupConLoss...
  Batch size: 8
  Feature dim: 512
  Num classes: 6
  Loss: 2.3456
  ✅ Backward pass successful
  Loss (all same class): 0.0001
  Loss (all different): 4.5678
  ✅ SupConLoss test passed!

Testing CombinedLoss...
  SupCon only:
    Total loss: 2.3456
    SupCon loss: 2.3456

  SupCon + CE:
    Total loss: 3.1234
    SupCon loss: 2.3456
    CE loss: 1.5558

  ✅ CombinedLoss test passed!

======================================================================
✅ ALL LOSS TESTS PASSED!
======================================================================
```

**结果保存**: 无 (纯测试,不保存)

---

## 4. 完整测试

运行所有测试确保整个流程正常:

```bash
# 逐个运行所有测试
bash scripts/test_dataset.sh
bash scripts/test_encoders.sh
bash scripts/test_fusion.sh
bash scripts/test_model.sh
bash scripts/test_losses.sh
```

**或者创建一个完整测试脚本:**

创建 `scripts/test_all.sh`:
```bash
#!/bin/bash
# Complete test suite

echo "======================================================================"
echo "RUNNING COMPLETE TEST SUITE"
echo "======================================================================"
echo ""

cd "$(dirname "$0")/.."

echo ">>> Test 1/5: Dataset"
bash scripts/test_dataset.sh
if [ $? -ne 0 ]; then echo "❌ Dataset test failed"; exit 1; fi
echo ""

echo ">>> Test 2/5: Encoders"
bash scripts/test_encoders.sh
if [ $? -ne 0 ]; then echo "❌ Encoder test failed"; exit 1; fi
echo ""

echo ">>> Test 3/5: Fusion"
bash scripts/test_fusion.sh
if [ $? -ne 0 ]; then echo "❌ Fusion test failed"; exit 1; fi
echo ""

echo ">>> Test 4/5: Model"
bash scripts/test_model.sh
if [ $? -ne 0 ]; then echo "❌ Model test failed"; exit 1; fi
echo ""

echo ">>> Test 5/5: Losses"
bash scripts/test_losses.sh
if [ $? -ne 0 ]; then echo "❌ Loss test failed"; exit 1; fi
echo ""

echo "======================================================================"
echo "✅ ALL TESTS PASSED!"
echo "======================================================================"
```

**运行完整测试:**
```bash
bash scripts/test_all.sh
```

**预期总耗时**: 约 2-5 分钟 (取决于硬件)

---

## 5. 训练模型

### 5.1 配置训练参数

编辑 `configs/train_config.py`:

```python
TRAIN_CONFIG = {
    # 优化参数
    "batch_size": 16,          # 根据GPU显存调整 (16/32/64)
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    
    # 学习率调度
    "lr_scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    
    # 损失函数
    "loss_type": "supcon",
    "temperature": 0.07,
    
    # 训练策略
    "gradient_clip": 1.0,
    "mixed_precision": True,   # 开启混合精度训练
    
    # 设备
    "device": "cuda",          # 使用GPU
    "num_workers": 8,          # 数据加载线程数
    
    # 日志
    "log_interval": 10,        # 每10个batch记录一次
    "val_interval": 1,         # 每个epoch验证一次
    "save_interval": 5,        # 每5个epoch保存一次
}
```

### 5.2 开始训练

**基础训练 (使用默认配置):**
```bash
bash scripts/train.sh
```

**自定义训练 (修改配置后):**
```bash
# 直接运行
python src/train.py

# 或使用 nohup 后台运行
nohup bash scripts/train.sh > training.log 2>&1 &

# 查看训练日志
tail -f training.log
```

### 5.3 训练输出

**实时日志:**
```
======================================================================
INITIALIZING MODEL
======================================================================
  Total parameters: 45,678,901
  Trainable parameters: 45,678,901
  Output dimension: 512
  Device: cuda

======================================================================
INITIALIZING DATA LOADERS
======================================================================
  Train samples: 800
  Val samples: 200
  Batch size: 16
  Train batches: 50
  Val batches: 13

======================================================================
INITIALIZING OPTIMIZER
======================================================================
  Optimizer: adamw
  Learning rate: 0.0001
  Weight decay: 0.0001
  Scheduler: cosine

======================================================================
INITIALIZING LOSS
======================================================================
  Loss: Supervised Contrastive
  Temperature: 0.07

======================================================================
STARTING TRAINING
======================================================================
  Epochs: 100
  Start time: 2025-10-10 14:23:45

Epoch 1/100
----------------------------------------------------------------------
  [  1][  1/ 50] Loss: 2.3456 | Avg: 2.3456 | LR: 1.00e-04
  [  1][ 10/ 50] Loss: 2.1234 | Avg: 2.2145 | LR: 1.00e-04
  [  1][ 20/ 50] Loss: 1.9876 | Avg: 2.1123 | LR: 1.00e-04
  [  1][ 30/ 50] Loss: 1.8543 | Avg: 2.0456 | LR: 1.00e-04
  [  1][ 40/ 50] Loss: 1.7654 | Avg: 1.9876 | LR: 1.00e-04
  [  1][ 50/ 50] Loss: 1.6789 | Avg: 1.9234 | LR: 1.00e-04
  Validation Loss: 2.0543
  ✅ Saved best model (epoch 1)
  Epoch time: 45.3s

Epoch 2/100
----------------------------------------------------------------------
  [  2][  1/ 50] Loss: 1.8765 | Avg: 1.8765 | LR: 9.50e-05
  ...
```

**结果保存位置:**
```
outputs/
├── checkpoints/
│   ├── latest.pth         # 最新模型
│   ├── best.pth           # 最佳模型 (验证损失最低)
│   ├── epoch_005.pth      # 第5个epoch
│   ├── epoch_010.pth      # 第10个epoch
│   └── ...
└── logs/
    └── training_log.json  # 训练历史 (JSON格式)
```

### 5.4 监控训练

**查看训练日志:**
```bash
# 查看 JSON 日志
python -m json.tool outputs/logs/training_log.json

# 提取关键指标
python -c "
import json
with open('outputs/logs/training_log.json') as f:
    log = json.load(f)
    train_losses = [m['loss'] for m in log['train']]
    val_losses = [m['loss'] for m in log['val']]
    print(f'Train loss: {train_losses[-1]:.4f}')
    print(f'Val loss: {val_losses[-1]:.4f}')
    print(f'Best val loss: {log[\"best_metric\"]:.4f}')
"
```

**可选: 使用 TensorBoard 可视化**
```bash
# 安装 TensorBoard
pip install tensorboard

# 启动 (如果实现了 TensorBoard 集成)
tensorboard --logdir outputs/logs
```

---

## 6. 预期输出汇总

### 6.1 测试输出

| 测试模块 | 测试脚本 | 预期耗时 | 输出结果 | 保存位置 |
|---------|---------|---------|---------|---------|
| 数据加载 | `test_dataset.sh` | ~10s | ✅ 所有数据形状正确 | 无 |
| 编码器 | `test_encoders.sh` | ~20s | ✅ 4个编码器全部通过 | 无 |
| 融合模块 | `test_fusion.sh` | ~15s | ✅ 融合和注意力正确 | 无 |
| 完整模型 | `test_model.sh` | ~30s | ✅ 端到端测试通过 | 无 |
| 损失函数 | `test_losses.sh` | ~5s | ✅ SupConLoss 正确 | 无 |

### 6.2 训练输出

| 输出类型 | 文件路径 | 内容 |
|---------|---------|------|
| 最新模型 | `outputs/checkpoints/latest.pth` | 最后一个epoch的模型状态 |
| 最佳模型 | `outputs/checkpoints/best.pth` | 验证损失最低的模型 |
| 周期检查点 | `outputs/checkpoints/epoch_XXX.pth` | 定期保存的模型 |
| 训练日志 | `outputs/logs/training_log.json` | 完整训练历史(损失、学习率等) |

**训练日志结构 (training_log.json):**
```json
{
  "train": [
    {"loss": 2.3456, "time": 45.3, "lr": 0.0001},
    {"loss": 1.8765, "time": 44.8, "lr": 0.000095},
    ...
  ],
  "val": [
    {"loss": 2.0543},
    {"loss": 1.9234},
    ...
  ],
  "config": {
    "batch_size": 16,
    "num_epochs": 100,
    ...
  },
  "best_metric": 1.2345
}
```

---

## 7. 故障排查

### 7.1 常见问题

**问题 1: ImportError: No module named 'torch'**
```bash
# 解决方案: 重新安装 PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**问题 2: CUDA out of memory**
```bash
# 解决方案: 减小 batch_size
# 编辑 configs/train_config.py:
"batch_size": 8,  # 从16改为8
```

**问题 3: 模型加载失败 (transformers)**
```bash
# 解决方案: 使用 Dummy 模式测试
# 测试脚本已经默认使用 dummy=True
# 或者检查网络连接下载预训练模型
```

**问题 4: DataLoader num_workers 错误 (Windows)**
```bash
# 解决方案: 减少 worker 数量
# 编辑 configs/train_config.py:
"num_workers": 0,  # Windows 下使用 0
```

### 7.2 验证安装

**快速验证脚本:**
```bash
python << EOF
import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# 测试导入
sys.path.insert(0, '.')
from src.models import QuadModalSOTAModel
from src.losses import SupConLoss
from src.dataset import WeldingDataset

print("✅ All imports successful!")
EOF
```

### 7.3 获取帮助

**查看详细文档:**
- 技术方案: `README.md`
- 实现进度: `docs/PROGRESS_QUADMODAL.md`
- 项目结构: `docs/PROJECT_STRUCTURE.md`

**检查代码:**
- 模型定义: `src/models/`
- 配置文件: `configs/`
- 测试脚本: `tests/`

---

## 🎯 下一步

完成上述步骤后,您应该能够:
- ✅ 成功运行所有测试
- ✅ 理解四模态架构
- ✅ 开始训练自己的模型
- ✅ 在服务器上部署训练

**推荐学习路径:**
1. 理解每个编码器的工作原理 (`src/models/`)
2. 研究交叉注意力融合机制 (`src/models/fusion.py`)
3. 调整训练配置优化性能 (`configs/train_config.py`)
4. 实验不同的超参数组合

**进阶任务:**
- 实现 k-NN 评估协议 (Step 6)
- 添加更多数据增强策略
- 实验不同的融合策略
- 可视化注意力权重

---

**快速开始指南更新完成!** 📚
