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

需要image/sensor/fusion（若有多个模态）的mean AUC和每个defect type AUC。若无多个模态，直接report该模态的mean AUC和每个defect type AUC。

*   **统一使用 AUC**:
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

- [x] 训练集能否正常加载（期望：576个Good样本）
- [x] 特征提取是否成功
- [x] 内存使用是否在可接受范围
- [x] 测试集评估能否正常运行
- [x] AUC指标计算是否符合预期

### 技术要点

1. **模态适配**: 传感器数据 → 伪深度图(224x224x3)
2. **训练模式**: 无监督特征提取 + 内存库
3. **评估指标**: Image AUC, Pixel AUC, AU-PRO
4. **配置对齐**: 遵循setup.md统一训练协议

详细说明请参阅：`baselines/M3DM/README.md` 第15节

---

## M3DM Evaluate修复记录（2025-12-13）

### 问题诊断

**报错**: `AttributeError: 'TripleFeatures' object has no attribute 'categories'`

**根本原因**:
1. `load_state()` 方法未重新初始化跟踪列表（`categories`, `image_preds`等）
2. PyTorch的 `nn.Module.__getattr__` 机制无法找到普通Python属性
3. 决策层模型（`detect_fuser`和`seg_fuser`）未在评估时加载

### 实施修复

#### 1. 完善状态加载逻辑
- **修改文件**: `feature_extractors/features.py`
- **修改内容**: `load_state()` 方法新增跟踪列表初始化
- **修复行数**: +9行

#### 2. 评估时加载融合模型
- **修改文件**: `weld_m3dm_runner.py` - `evaluate()` 方法
- **修改内容**: 加载 `detect_fuser` 和 `seg_fuser` 模型
- **修复行数**: +15行

#### 3. 训练时保存融合模型
- **修改文件**: `weld_m3dm_runner.py` - `fit()` 方法
- **修改内容**: 保存分割模型 `seg_fuser`（原本只保存了检测模型）
- **修复行数**: +8行

#### 4. 修复字典迭代bug
- **修改文件**: `weld_m3dm_runner.py` - line 176
- **错误**: `self.methods.values()` 无法解包为 `(key, value)`
- **正确**: `self.methods.items()`

### 代码修改统计

| 项目 | 数量 | 说明 |
|------|------|------|
| 修改文件 | 2个 | features.py, weld_m3dm_runner.py |
| 新增代码 | +35行 | 状态初始化、模型加载与保存 |
| 修复bug | 1处 | 字典迭代语法错误 |
| 遵循原则 | ✅ | 最小修改、保持风格、SOLID原则 |

### 验证指南

**快速测试**:
```bash
# 1. 检查checkpoint
ls -lh /root/autodl-tmp/save/weld_*
# 预期：feature_extraction.pkl + decision_model.pkl + segmentation_model.pkl

# 2. 运行评估
bash baselines/M3DM/evaluate.sh

# 3. 预期输出：
# ✓ Loaded decision model: /root/autodl-tmp/save/weld_DINO+Point_MAE+Fusion_decision_model.pkl
# ✓ Loaded segmentation model: /root/autodl-tmp/save/weld_DINO+Point_MAE+Fusion_segmentation_model.pkl
# Extracting test features for weld: 100%|████████| 1732/1732
# Class: weld, Image ROCAUC: 0.xxx, Pixel ROCAUC: 0.xxx, AU-PRO: 0.xxx
```

### 技术要点

1. **PyTorch模块属性**: 普通Python属性必须在对象生命周期内显式初始化
2. **Checkpoint设计**: 分离持久化状态（特征库）与瞬态状态（跟踪列表）
3. **Late Fusion模型**: `detect_fuser`（图像级）和`seg_fuser`（像素级）需单独保存

详细说明请参阅：`baselines/M3DM/README.md` - "评估阶段AttributeError修复"章节

---

## M3DM Device Mismatch 修复完成（2025-12-15）

### 已完成修复
✅ **修复6**: 设备不匹配错误 - 统计量作为Python float标量保存/加载
- 修改文件: `feature_extractors/features.py`
  - `get_state()`: 使用`.item()`将统计量转为Python float
  - `load_state()`: 保持统计量为Python float（不转为tensor）
- 原理: PyTorch对Python标量自动设备广播，无需手动迁移
- 优势: 零设备开销、内存高效、checkpoint更小、设备无关

### 需要服务器执行的操作

**⚠️ 必须重新训练生成新checkpoint** (因为修改了保存格式)

**选项1 - 使用自动化脚本** (推荐):
```bash
bash baselines/M3DM/retrain_with_new_checkpoint.sh
```
此脚本会自动：
1. 备份旧checkpoint到 `.backup_YYYYMMDD_HHMMSS`
2. 删除旧的feature/decision/segmentation模型
3. 运行完整训练生成新checkpoint
4. 验证新checkpoint是否生成成功

**选项2 - 手动执行**:
```bash
# 1. 删除旧checkpoint
rm -f /root/autodl-tmp/save/weld_feature_extraction.pkl
rm -f /root/autodl-tmp/save/weld_DINO+Point_MAE+Fusion_decision_model.pkl
rm -f /root/autodl-tmp/save/weld_DINO+Point_MAE+Fusion_segmentation_model.pkl

# 2. 重新训练
bash baselines/M3DM/train.sh

# 3. 验证evaluate
bash baselines/M3DM/evaluate.sh
```

**预期结果**:
- 新checkpoint大小约500MB
- evaluate阶段不再出现 `RuntimeError: device mismatch`
- 代码可在CPU/CUDA间无缝切换

详细技术说明请参阅: `baselines/M3DM/README.md` - "修复6"章节

TODO:
1. 仔细阅读baselines\setup.md 理解任务和当前状态，始终遵守下方全局执行要求。
2. 修复m3dm评估：
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Extracting test features for weld: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1732/1732 [44:35<00:00,  1.54s/it]
✓ Test features saved: /root/autodl-tmp/save/weld_test_features.pkl
  Size: 3565.21 MB
Class: weld, DINO+Point_MAE+Fusion Image ROCAUC: 0.682, DINO+Point_MAE+Fusion Pixel ROCAUC: 0.642, DINO+Point_MAE+Fusion AU-PRO: 0.328

DINO+Point_MAE+Fusion - Per-Defect-Type AUC:
============================================================
  Good                                    : N/A
  Excessive_Convexity                     : 0.000
  Undercut                                : 0.000
  Lack_of_Fusion                          : 0.000
  Porosity                                : 0.000
  Spatter                                 : 0.000
  Burnthrough                             : 0.000
  Porosity_w_Excessive_Penetration        : 0.000
  Excessive_Penetration                   : 0.000
  Crater_Cracks                           : 0.000
  Warping                                 : 0.000
  Overlap                                 : 0.000
============================================================

运行了# 删除旧的test features（包含错误的category）
rm /root/autodl-tmp/save/weld_test_features.pkl

# 重新运行评估
bash baselines/M3DM/evaluate.sh
还是如上 category不显示
   
## 全局执行要求：
（step by step 你自己安排）
最后汇报需求实现情况，集中中文写在当前baseline的README里面更新，不得有其他文档，不用更新setup.md。
不配置环境，代码之后迁移到服务器上运行
（修改代码优雅符合原风格，注意代码规范符合SOLID原则，代码分区有序分文件和文件夹放置，每个不得超过800行，prompt和config单列文件放置，运行脚本命令管理使用sh。非必要不新增文件，不要写不必要的回退逻辑，尽量对原文件作最小修改，切中要害，严格实现我的需求）
