### **1. 数据集 (Dataset)**

这是整个研究的基础，包含了超过4000个样本。

*   **来源与环境**: 在真实的汽车工厂环境中，由供应商使用6轴弧焊机器人采集。
*   **材料**: 两种钢材：
    *   FE410 (常用于汽车应用)
    *   BSK46 (含碳量更高，用于高强度应用)
*   **厚度**: 主要为7mm，部分特定缺陷（如烧穿）使用3mm。
*   **缺陷类别**: 共12类，其中“Good”为正常样本，其余11类为缺陷。
    *   Good (819个)
    *   Excessive Convexity (160个)
    *   Undercut (160个)
    *   Crater Cracks (161个)
    *   Overlap (160个)
    *   Excessive Penetration (480个)
    *   Porosity w/Excessive Penetration (480个)
    *   Spatter (320个)
    *   Lack Of Fusion (320个)
    *   Warping (320个)
    *   Porosity (340个)
    *   Burnthrough (320个)
    *   **总计**: 4040个样本
*   **数据划分**:
    *   **训练集 (Train)**: 仅包含“Good”样本 (576个)。这是无监督学习的关键，模型只在正常数据上训练。
    *   **验证集 (Validation)**: 包含“Good”样本 (122个) 和“Defective”样本 (1610个)。
    *   **测试集 (Test)**: 包含“Good”样本 (121个) 和“Defective”样本 (1611个)。
    *   **注意**: 缺陷样本在验证集和测试集之间平均分配。
*   **传感器与采集**:
    *   **视频**: 使用KML焊接相机，安装在焊枪臂上，距离约200mm。记录帧率约为30 FPS，保存为AVI格式。
    *   **音频**: 使用两个Earthworks SR314高带宽麦克风，安装在工作台上，距离焊机运动轴约300mm。采样率为192 KHz，保存为无损FLAC格式。
    *   **同步**: 音频数据作为时间基准，通过分析音频和光照变化来对齐音视频数据。
*   **标签**: 由焊接专家指导机器人生成缺陷，但未进行后处理验证，因此数据集存在一定的标签噪声。

---

### **2. 模型架构与参数设置**

采用自编码器（Auto-Encoder）进行无监督异常检测。模型的目标是重建输入，对于异常数据，重建误差会更大，从而产生更高的异常分数。

#### **2.1 音频异常检测模型 (Audio Anomaly Detection)**

一个1D CNN自编码器。

*   **输入**: 原始音频信号，采样率192 kHz，使用单声道。
*   **预处理**: 使用短时傅里叶变换（STFT）将音频转换为频谱图。模型的输入是频谱图的时间序列。
*   **模型架构 (Table 5)**:
    *   **编码器 (Encoder)**:
        *   `BatchNorm1D`: 对输入进行批归一化。
        *   `Conv1D`: 输入通道数为`n-bins`（频谱图的频率bins数量），输出通道1024，卷积核大小3，步长1。
        *   `3 x Conv1D`: 输入输出通道均为1024，卷积核大小3，步长1。
        *   `Conv1D`: 输入通道1024，输出通道为`bottleneck size`（瓶颈层维度），卷积核大小3，步长1。
    *   **解码器 (Decoder)**:
        *   `ConvTranspose1D`: 输入通道为`bottleneck size`，输出通道1024，卷积核大小3，步长1。
        *   `3 x ConvTranspose1D`: 输入输出通道均为1024，卷积核大小3，步长1。
        *   `ConvTranspose1D`: 输入通道1024，输出通道`n-bins`，卷积核大小3，步长1。
    *   **激活函数**: 除最后一层外，所有层均使用Leaky-ReLU；最后一层使用PReLU。
    *   **总参数量**: 31,670,306。
*   **超参数搜索**:
    *   **FFT窗口大小 (FFT Window Size)**: 4096, 16384, 32768, 65536。
    *   **瓶颈层维度 (Bottleneck Dimension)**: 16, 32, 48, 64。
    *   **Hop Length**: 固定为FFT窗口大小的50%。
    *   **最佳配置**: 根据验证集AUC，最终选择 **FFT窗口=16384，瓶颈层维度=48**。
*   **延迟 (Latency)**: 等于Hop Length。例如，Hop Length=8192时，延迟约为42.7ms。
*   **输入缓冲区大小 (Buffer Size)**: 必须满足公式 `buffer size = hop length × (10 + FFT window / hop length)`。例如，FFT=16384, Hop=8192时，缓冲区需容纳 `8192 × (10 + 16384/8192) = 8192 × 12 = 98304` 个样本。
*   **训练设置**:
    *   优化器: Adam。
    *   学习率调度: 一个周期学习率调度（one-cycle learning schedule），峰值学习率为 `1×10⁻⁴`。
    *   训练轮次: 50 epochs。
    *   损失函数: 均方误差 (MSE)。

#### **2.2 视觉异常检测模型 (Visual Anomaly Detection)**

这是一个两阶段模型。

*   **第一阶段：特征提取**
    *   **模型**: 使用预训练且权重固定的SlowFast模型（来自MMAction2库，具体为`slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth`）。
    *   **输入**: 视频片段。
    *   **处理方式**: 使用一个64帧的滑动窗口（对应约2秒，30FPS），逐帧滑动以提取每个帧的特征向量。
    *   **输出**: 每个帧被编码为一个2304维的特征向量。
*   **第二阶段：异常检测**
    *   **模型**: 一个简单的全连接自编码器。
    *   **输入**: 2304维的特征向量。
    *   **模型架构 (Table 6)**:
        *   **编码器 (Encoder)**:
            *   `Linear`: (2304 -> 512)
            *   `Linear`: (512 -> 256)
            *   `Dropout(p=0.5)`
            *   `Linear`: (256 -> 128)
            *   `Dropout(p=0.5)`
            *   `Linear`: (128 -> 64)
            *   `Dropout(p=0.5)`
            *   `Linear`: (64 -> 64) - 这是瓶颈层。
        *   **解码器 (Decoder)**:
            *   `Linear`: (64 -> 64)
            *   `Linear`: (64 -> 128)
            *   `Linear`: (128 -> 256)
            *   `Linear`: (256 -> 512)
            *   `Dropout(p=0.5)`
            *   `Linear`: (512 -> 2304)
        *   **激活函数**: 所有线性层后使用ReLU。
    *   **训练设置**:
        *   优化器: Adam。
        *   学习率: 0.0005。
        *   训练轮次: 最多1000 epochs。
        *   损失函数: 均方误差 (MSE)。
        *   **重要**: 在训练第二阶段时，第一阶段的SlowFast模型权重是**冻结**的，不进行反向传播。
*   **延迟**: 约1秒，因为使用了64帧的滑动窗口，且窗口中心是对准当前帧的。

#### **2.3 多模态异常检测 (Multi-modal Anomaly Detection)**

采用后期融合（Late Fusion）策略。

*   **步骤**:
    1.  分别使用上述最佳音频模型和视频模型，对每个样本的每一帧计算出异常分数。
    2.  **标准化 (Standardization)**: 对每个模态的异常分数，使用其在**训练集**上的均值和标准差进行标准化，使其具有可比性。
    3.  **加权融合 (Weighted Combination)**: 将标准化后的音频分数 (`S_audio`) 和视频分数 (`S_video`) 进行凸组合：`S_fused = w * S_audio + (1-w) * S_video`。
    4.  **权重优化**: 在**验证集**上，通过网格搜索（步长0.01）找到最优权重 `w`。
    5.  **最终结果**: 使用找到的最佳权重，在**测试集**上计算最终的融合异常分数，并据此评估性能。
*   **最佳权重**: 实验中发现最佳权重为 `w=0.37` (音频) 和 `1-w=0.63` (视频)。

---

### **3. 评估指标与方法 (Evaluation Metrics and Methods)**

核心思想是将问题视为**异常检测**而非分类。

*   **核心指标**: **AUC (Area Under the ROC Curve)**。
    *   AUC衡量的是模型区分正常样本和异常样本的能力，取值范围[0, 1]，值越大越好。
    *   它独立于阈值选择，是评估排序能力的理想指标。
    *   计算公式: `AUC = ∫(from 0 to 1) ROC(x) dx`，其中 `x` 是假阳性率(FPR)，`ROC(x)` 是真阳性率(TPR)。
*   **如何处理帧级分数到样本级标签?**
    *   由于模型在每一帧都输出一个异常分数，而每个样本只有一个标签（正常或某种缺陷），需要将所有帧的分数聚合为一个样本级别的分数。
    *   **音频**: **期望值 (Expected Value)**
    *   **视频**: 
        *   `Max over 2s-MA`: 在2秒的滑动窗口内取平均后再取最大值。
*   **评估流程**:
    1.  在验证集上调整超参数（如音频的FFT窗口、瓶颈维度，视频的聚合方法，多模态的权重）。
    2.  使用在验证集上表现最好的配置，在测试集上计算最终的AUC得分。
    3.  报告总体AUC以及按缺陷类型细分的AUC，以分析不同缺陷的检测难度。

---

### **4. 关键实验结果摘要**

*   **音频模型 (Test Set)**: 最佳AUC为 **0.8460**。
*   **视频模型 (Test Set)**: 最佳AUC为 **0.8977**。
*   **多模态模型 (Test Set)**: 最佳AUC为 **0.9178**。
*   **结论**: 视频模态通常优于音频模态，但两者结合能带来显著提升。特别是对于单模态表现较差的缺陷类别（如“Lack of fusion”, “Warping”），融合后性能均有改善。

---

### **5. 复现要点总结**

1.  **数据准备**: 收集或获取包含正常和多种缺陷的焊接音视频数据。确保数据标注清晰，并按比例划分为训练、验证、测试集。注意音视频同步。
2.  **音频模型**:
    *   实现1D CNN自编码器结构。
    *   进行STFT预处理。
    *   在验证集上搜索FFT窗口和瓶颈维度。
    *   使用Adam优化器，学习率调度，训练50轮。
    *   使用期望值聚合帧级分数。
3.  **视频模型**:
    *   下载并加载预训练的SlowFast模型。
    *   使用64帧滑动窗口提取特征。
    *   构建并训练一个全连接自编码器（冻结SlowFast）。
    *   使用Adam优化器，学习率0.0005，训练最多1000轮。
    *   使用“Max over 2s-MA”方法聚合分数。
4.  **多模态融合**:
    *   分别得到音频和视频的帧级分数。
    *   使用训练集的统计量对分数进行标准化。
    *   在验证集上用网格搜索找到最佳融合权重。
    *   在测试集上应用该权重计算最终分数。
5.  **评估**:
    *   计算整体AUC。
    *   计算每种缺陷类型的AUC。
    *   绘制ROC曲线进行可视化分析。

---

### **6. 实现记录**

#### **6.1 实现概述**
本实现严格遵循论文规范和`baselines/setup.md`中的公平对比要求，实现了Late Fusion基线模型。

#### **6.2 代码结构**
```
baselines/Late_Fusion/
├── config.py          # 配置文件（音频/视频/融合参数）
├── models.py          # 模型定义（AudioAutoEncoder, VideoAutoEncoder, LateFusionModel）
├── utils.py           # 工具函数（STFT, 分数聚合, AUC计算, ROC绘图）
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本
├── train.sh           # 训练Shell脚本
├── evaluate.sh        # 评估Shell脚本
└── README.md          # 本文档
```

#### **6.3 实现细节**

**音频自编码器 (AudioAutoEncoder)**
- 架构：严格按照Table 5实现
- 输入：STFT频谱图 (n_bins=8193, 对应n_fft=16384)
- 瓶颈维度：48 (论文最佳配置)
- 激活函数：Leaky-ReLU + 最后一层PReLU
- 参数量：31,670,306

**视频自编码器 (VideoAutoEncoder)**
- Stage 1: 冻结的SlowFast特征提取器 (feature_dim=2304)
- Stage 2: 全连接自编码器，严格按照Table 6实现
- Dropout: 0.5
- 激活函数：ReLU

**后期融合 (LateFusionModel)**
- 标准化：使用训练集统计量
- 融合方法：加权求和，权重通过验证集网格搜索优化
- 默认权重：w_audio=0.37, w_video=0.63

#### **6.4 训练配置**

遵循`baselines/setup.md`的统一训练协议：

**音频模型**
- 优化器：Adam
- 学习率调度：One-Cycle LR (max_lr=1e-4)
- 训练轮次：50 epochs
- 损失函数：MSE
- 批次大小：32

**视频模型**
- 优化器：Adam (lr=5e-4)
- 训练轮次：最大1000 epochs (带早停，patience=50)
- 损失函数：MSE
- 批次大小：32

**通用设置**
- 随机种子：42
- 数据集划分：使用`configs/manifest.csv`
  - 训练集：576个Good样本
  - 验证集：122 Good + 1610 Defective
  - 测试集：121 Good + 1611 Defective

#### **6.5 评估方法**

**分数聚合**
- 音频：Expected Value (帧级分数的均值)
- 视频：Max over 2s-MA (2秒移动平均后取最大值)

**评估指标**
- 主要指标：AUC (Area Under ROC Curve)
- 报告：整体AUC + 各缺陷类型AUC
- 可视化：ROC曲线对比图

#### **6.6 使用方法**

**训练**
```bash
# 训练两个模型
bash baselines/Late_Fusion/train.sh --modality both

# 仅训练音频模型
bash baselines/Late_Fusion/train.sh --modality audio

# 仅训练视频模型
bash baselines/Late_Fusion/train.sh --modality video

# 使用dummy数据测试
bash baselines/Late_Fusion/train.sh --modality both --dummy
```

**评估**
```bash
# 评估并融合
bash baselines/Late_Fusion/evaluate.sh

# 使用dummy数据测试
bash baselines/Late_Fusion/evaluate.sh --dummy
```

#### **6.7 预期结果**

根据论文报告，预期性能：
- 音频模型 Test AUC: ~0.8460
- 视频模型 Test AUC: ~0.8977
- 融合模型 Test AUC: ~0.9178

#### **6.8 实现日期**
- 实现日期：2025-12-07
- 实现者：GitHub Copilot (Claude Sonnet 4.5)
- 状态：完成，待服务器验证