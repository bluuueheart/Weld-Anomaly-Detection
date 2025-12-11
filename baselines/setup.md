# 公平对比实验设置指南
为了确保在焊接缺陷检测任务中对比不同模型baseline的实验结果具有公平性和可比性，baselines/中每个子文件夹为一个复现的模型，都需要统一遵循以下详细的实验设置指南进行实现。baselines/外为已实现的sota模型，若无明确指令，不要做任何修改，只能把当前baseline（如Late_Fusion）的代码放在对应文件夹内。baselines/每个子文件夹中的README.md已提供了该baseline的详细复现说明或原代码仓库，需要在其末尾更新5. 中提到的记录说明。
---

### **1. 核心原则：控制变量法**

在对比不同模型时，除了模型架构本身，其他所有条件都应尽可能相同。这包括：

*   **数据集**: 使用完全相同的训练、验证、测试集划分。
*   **预处理**: 对所有模型应用相同的输入预处理步骤。
*   **评估指标**: 使用相同的指标（如AUC）和相同的聚合方法（如视频用“Max over 2s-MA”，音频用“Expected Value”）。
*   **训练配置**: 尽可能对齐优化器、学习率调度、批次大小、训练轮次等。
*   **硬件与软件**: 在相同或相似的硬件环境下运行，使用相同的深度学习框架版本。

---

### **2. 数据集与划分 (Dataset & Split)**
*   **严格复现原始划分**: 您必须使用论文中描述的**完全相同的**数据划分。
*   **模态适配**:
    *   对于仅使用视频的Baseline，只向它们提供您数据集中的视频模态，忽略音频、图像和传感器数据。
    *   对于使用RGBD的Baseline，提供RGB图像，D换成传感器（补齐其原有模态数量），融合策略不变。
    *   其他baseline以此类推。
    *   **训练集 (Train)**: 仅包含576个“Good”样本。
    *   **验证集 (Validation)**: 包含122个“Good”样本 + 1610个“Defective”样本。
    *   **测试集 (Test)**: 包含121个“Good”样本 + 1611个“Defective”样本。
    *   **关键**: baselines/目录外，原sota模型应已实现数据加载逻辑，对应configs\manifest.csv应已经提供了合格的数据划分，可以直接迁徙复用。
*   **数据预处理**:
    *   **音频**: 采样率192kHz，单声道。对于您的新模型，如果它们接受原始波形，就直接使用；如果接受频谱图，请使用与论文相同的STFT参数（如FFT窗口=16384, Hop Length=8192）。
    *   **视频**: 使用相同的帧率（约30 FPS），并确保音视频对齐方式一致。

---

### **3. 训练设置与超参数对齐 (Training Setup & Hyperparameters)**

#### **3.1 统一的训练协议 (Unified Training Protocol)**

*   **优化器**: 统一使用 **Adam**。
*   **学习率调度**: 统一采用 **One-Cycle Learning Rate Scheduler**。
    *   **峰值学习率**: 可以设定为 `1e-4` 作为起点。
    *   **总训练轮次 (Epochs)**: 设定一个固定的值，例如 **50轮**。
    *   **Warmup比例**: 无。
    *   **Anneal比例**: 余下的90%。
*   **损失函数**: 统一使用 **MSE (Mean Squared Error)**。
*   **批次大小 (Batch Size)**: 32。可以适当减小批次大小，但请记录下来并在报告中说明。
*   **随机种子 (Random Seed)**: 设置固定的随机种子（42），以保证实验的可复现性。

#### **3.3 关于“Epoch”的对齐**
  为所有模型设定一个**固定的、足够长的训练轮次**，例如 **100 epochs**。

---

### **4. 评估指标与方法 (Evaluation Metrics & Method)**

这是确保结果可比性的核心。

*   **统一使用 AUC**: 这是论文的核心指标，也最适合异常检测任务。
*   **统一帧级分数聚合方法**:
    *   **音频**: 使用**期望值 (Expected Value)** 来聚合帧级分数。
    *   **视频**: 使用**“Max over 2s-MA”** 方法来聚合帧级分数。
    *   **多模态**: 使用**标准化后的加权融合**，权重 `w=0.37` (音频) 和 `1-w=0.63` (视频)。
*   **报告完整结果**:
    *   **总体AUC**: 报告所有缺陷类别的平均AUC。
    *   **分项AUC**: 报告每种缺陷类型的AUC，以便分析模型在不同类型缺陷上的表现。
    *   **置信区间**: 如果可能，通过多次运行（改变随机种子）计算AUC的均值和标准差，以显示结果的稳定性。
*   **可视化**: 绘制ROC曲线，直观展示不同模型的性能差异。

---

### **5. 实验记录与报告 (Experiment Logging & Reporting)**

为了保证实验的透明度和可复现性，必须详细记录每一次实验的配置。baselines/每个子文件夹中的README.md已提供了该baseline的详细复现说明或代码仓库，需要在末尾更新以下内容：

*   **记录内容**:
    *   模型名称和架构。
    *   所有超参数（学习率、批次大小、优化器、损失函数、dropout率等）。
    *   训练轮次（或早停轮次）。
    *   验证集AUC（用于选择超参数）。
    *   测试集AUC（最终报告的性能）。
    *   训练时间、GPU显存占用等。
    *   随机种子。
*   **工具**: 使用 `W&B (Weights & Biases)`、`TensorBoard` 或 `MLflow` 等工具自动记录实验日志。

---

### **6. 总结：公平对比的 checklist**

在开始您的实验前，请确认以下几点：

✅ **数据集**: 使用与论文完全相同的训练/验证/测试集划分。

✅ **预处理**: 对所有模型应用相同的输入预处理。

✅ **评估指标**: 统一使用AUC，并采用论文中指定的帧级分数聚合方法。

✅ **训练协议**: 为所有模型设定统一的优化器、学习率调度、损失函数和最大训练轮次。

✅ **超参数搜索**: 在统一协议的基础上，为每个模型独立进行超参数搜索，但搜索空间和评估标准必须一致。

✅ **随机性控制**: 设置固定的随机种子。

✅ **结果报告**: 报告总体AUC和分项AUC，并记录所有实验配置。

## M3DM Baseline 实现记录（2025-12-09）

### 问题诊断与解决

**原始报错**：
```
WARNING: No samples found for split TRAIN!
Root dir: Data
RuntimeError: Empty feature libraries detected! xyz: 0, rgb: 0, fusion: 0.
```

**根本原因**：
1. 数据路径配置错误：默认使用 `Data`，但服务器实际路径为 `/root/autodl-tmp/Intel_Robotic_Welding_Multimodal_Dataset/raid/intel_robotic_welding_dataset`
2. 训练集为空导致特征库为空列表，`torch.cat([], 0)` 抛出底层错误
3. 缺乏有效的错误提示机制

### 实施的修复

#### 1. 数据路径配置更新
- **修改文件**: `weld_config.py`, `weld_main.py`, `train.sh`
- **修改内容**: 更新默认数据路径为服务器路径
- **灵活性**: 保留命令行参数和环境变量覆盖能力

#### 2. 空数据防护机制
- **修改文件**: `feature_extractors/multiple_features.py`
- **修改内容**: 为所有6个特征提取类添加空列表检查
- **影响类**: RGBFeatures, PointFeatures, FusionFeatures, DoubleRGBPointFeatures, DoubleRGBPointFeatures_add, TripleFeatures
- **优势**: 提供清晰的错误信息，指导用户快速定位问题

#### 3. 运行脚本完善
- **新增文件**: `test_data_loading.sh` (数据加载测试脚本)
- **更新文件**: `train.sh` (添加数据验证和错误处理)

### 代码修改统计

| 项目 | 数量 | 说明 |
|------|------|------|
| 修改文件 | 4个 | weld_config.py, weld_main.py, multiple_features.py, train.sh |
| 新增文件 | 1个 | test_data_loading.sh |
| 代码行数 | +70行 | 防御性检查、配置更新、脚本增强 |
| 遵循原则 | ✅ | 最小修改、保持风格、SOLID原则 |

### 运行指南

**首次运行**：
```bash
# 1. 测试数据加载
bash baselines/M3DM/test_data_loading.sh

# 2. 如果测试通过，运行训练
bash baselines/M3DM/train.sh
```

**自定义数据路径**：
```bash
WELD_DATA_PATH=/your/path bash baselines/M3DM/train.sh
```

### 待服务器验证项

- [ ] 训练集能否正常加载（期望：576个Good样本）
- [ ] 特征提取是否成功
- [ ] 内存使用是否在可接受范围
- [ ] 测试集评估能否正常运行
- [ ] AUC指标计算是否符合预期

### 技术要点

1. **模态适配**: 传感器数据 → 伪深度图(224x224x3)
2. **训练模式**: 无监督特征提取 + 内存库
3. **评估指标**: Image AUC, Pixel AUC, AU-PRO
4. **配置对齐**: 遵循setup.md统一训练协议

详细说明请参阅：`baselines/M3DM/README.md` 第15节

---

TODO:
1. 仔细阅读baselines\setup.md，理解任务和当前状态，始终遵守下方全局执行要求。
2. 现在进度：Epoch [50/50] Train Loss: 0.009289 Val Loss: 0.009620

Training completed. Final model saved to baselines/Late_Fusion/checkpoints/audio_autoencoder_final.pth

======================================================================
VIDEO AUTO-ENCODER
======================================================================
======================================================================
TRAINING VIDEO AUTO-ENCODER (Stage 2)
======================================================================
Total parameters: 2,715,840
Trainable parameters: 2,715,840
Device: cuda
Max epochs: 1000
Batch size: 32
Learning rate: 0.0005

Epoch [1/1000] Batch [10/18] Loss: 0.000180
Epoch [1/1000] Train Loss: 0.000342 Val Loss: 0.000020
两个baseline:Late_Fusion和M3DM的epoch对齐了吗？为什么这个Late_Fusion一会儿50epoch一会儿1000epoch？


## 全局执行要求：
（step by step 你自己安排）
最后汇报需求实现情况，集中中文写在当前baseline的README里面更新，不得有其他文档，不用更新setup.md。
不配置环境，代码之后迁移到服务器上运行
（修改代码优雅符合原风格，注意代码规范符合SOLID原则，代码分区有序分文件和文件夹放置，每个不得超过800行，prompt和config单列文件放置，运行脚本命令管理使用sh。非必要不新增文件，不要写不必要的回退逻辑，尽量对原文件作最小修改，切中要害，严格实现我的需求）