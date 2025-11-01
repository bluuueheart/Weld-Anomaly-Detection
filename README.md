# Weld-Anomaly-Detection

## 一句话总结
四模态深度融合：冻结三类预训练视觉/音频 backbone（视频/图像/音频），训练可学习的传感器编码器与 Cross-Attention 融合头，采用 Supervised Contrastive Loss + 下游分类/检测头微调。

## 合同（Inputs / Outputs / 错误模式）
- 输入：字典形式的四模态样本 {
  - `video`: Tensor (B, T, 1, H, W) — 采样帧数 T，单通道（灰度）
  - `post_images`: Tensor (B, N, 3, H, W) — N 个多角度静态图
  - `audio`: Tensor (B, C, L) 或 Mel (B, 1, M, F)
  - `sensors`: Tensor (B, S, F_s) — 多通道时间序列
  - `label`: int 或 Tensor (B,) 
- 输出：{
  - `features`: Tensor (B, D_fusion) — 融合特征
  - `logits`: Tensor (B, num_classes)
  - `anomaly_score`(可选)：float 或 Tensor (B,)
}
- 常见错误模式：模态不同步/长度不一致、后处理未归一化、部分模态缺失未按 mask 处理。

## 模型详细架构（分块描述）
1) Backbones（冻结为主）
   - VideoEncoder: V-JEPA（vit-based）
     - 输入 -> 时空特征序列 (B, T', Dv)
     - 冻结大部分参数，仅允许最后投射层/adapter 学习
   - ImageEncoder: DINOv2
     - 对 N 张图分别编码 -> (B, N, Di)
   - AudioEncoder: AST (或可替换为 Wav2Vec2/Audioset 变体)
     - 输出 (B, Da)
   - SensorEncoder: TransformerEncoder（从零训练）
     - 输入时间序列 -> 输出时序特征 (B, S', Ds)

2) Cross-Attention Fusion Module（Trainable）
   - 将四路特征视作 Key/Value/Query 的集合：允许 Video 或 Sensor 作为主查询，动态聚合其他模态信息。
   - 结构：多头 cross-attention -> 层归一化 -> FFN -> 池化（或 CLS token）-> 得到融合向量 D_fusion。

3) Heads
   - SupCon Projection Head（用于对比学习）：2-layer MLP -> L2 正则化
   - 分类 Head / 异常评分 Head：单层或两层 MLP 输出 logits / score

## 训练与微调策略（实用配置）
- 优化器：AdamW；基础 lr 1e-4（head），backbone lr 0 / 微小 lr（如 1e-6）用于少量解冻层
- Batch size 建议：2–16（受显存限制）
- 混合精度：推荐启用（AMP）
- 学习率调度：Cosine 或 ReduceLROnPlateau；warmup 500–1000 steps
- 冻结策略：默认冻结 backbone；对比实验：尝试解冻最后 1–3 层或使用 adapters/LoRA
- 损失：SupConLoss(primary) + CrossEntropy（辅助）或加权合并
- 检查点：按 val metric（e.g., F1）保存最佳模型，并周期性保存最近 N 个

## 数据管线要点
- 视频：固定帧率采样、中心/均匀采样 T 帧、resize 并标准化。
- 图片：读取 N 个角度，resize->normalize；缺角时 pad 常数或重复视角并 mask。
- 音频：resample->Mel spectrogram（或直接时域模型），归一化。
- 传感器：按时间对齐，线性/样条插值缺失，Z-score 标准化，固定长度截断/补零。

## 评估协议（可复现）
- 特征库 + kNN：在训练集构建特征库，测试时用 k-NN（k=5）评估分类/异常检出
- 指标：Accuracy, Macro F1, Per-class F1, AUROC（异常评分），Confusion Matrix
- 快速验证（过拟合测试）：用每类少量样本（如 2 个/类）训练，期望训练集准确率接近 100%。

## 优化与创新建议（按优先级，短句直接可落地）
1) 轻量微调优先：Adapters/LoRA 替代全量微调，极大减少可训练参数且易部署。
2) 时序对齐器：显式学习视频/传感器/音频的时间对齐（cross-modal temporal attention），减少模态错位误差。
3) 对比挖掘策略：采用困难负样本采样（hard negative mining）提升 SupCon 效果。
4) 半/自监督预训练：对本域无标签视频/音频做短期自监督（SimCLR/MAE 式）再微调。
5) 多任务头：同时训练分类 + 重建（或预测下步传感器），提升泛化与鲁棒性。
6) 校准与不确定量化：用温度缩放与贝叶斯后处理（MC Dropout 或 DeepEnsemble）获得可靠异常置信度。
7) 推理优化：动态量化 + 剪枝 + TorchScript 导出，确保工业部署延迟可控。
8) 可解释性：为图像/视频加入 GradCAM / attention 可视化，辅助工程师定位缺陷源。
9) 数据增强：针对焊接场景的合成瑕疵样本与时间域扰动增强，缓解类不平衡。
10) 融合探索：比较 Late Fusion vs Mid Fusion vs Cross-Attention（定量报告每项对 F1/AUROC 的影响）。

![最后一次运行的结果](image.png)
**最后一次运行的结果**：说实话挺令人疑惑的，疑似过拟合但acc没有提升空间了。暂时这样吧。已经调参许久了。

## 总结
大创TODO: 
控制部分：
- 实现

缺陷检测部分（当前）：
- 模块创新：方案确定和实施
- baseline复现
- 成果转化(可选): 具体工作量参考论文 Cross-Modal Learning for Anomaly Detection in Complex Industrial Process: Methodology and Benchmark, b刊1区，带benchmark，复现了10多个baseline
----------------------------------------------------------------------

# 详细技术方案与实现路径

## 基线设计与对比实验（Baselines & Ablations）

| 类别 (Category) | 模型/基线名称 (Model/Baseline Name) | 核心技术 (Core Technology) | 实验目的 / 对比对象 (Experimental Purpose / Comparison Target) |
| :--- | :--- | :--- | :--- |
| **A. 论文复现基线**<br>(Paper Reproduction) | `Paper-Baseline (Reproduced)` | Audio 1D CNN-AE + Video Slowfast-AE，然后进行后期分数融合 (Late Fusion) 。 | **最重要的基准**。验证您的实验环境，并提供一个必须超越的、最直接的性能参照点。 |
| **B. 单模态SOTA基线**<br>(Single-Modality SOTA) | `In-Process-Video-Only` | **V-JEPA (Fine-tuned)**: 微调用于实时黑白视频的自监督时空特征提取器。 | 证明先进的自监督视频编码器在过程监控中的优势。 |
| | `Post-Weld-Image-Only` | **DINOv2 (Fine-tuned)**: 微调用于焊后多角度静态图片的自监督外观特征提取器。 | **衡量新模态的独立贡献**。证明最终成品外观本身就包含了丰富的缺陷信息。 |
| | `Audio-Only` | **AST (Fine-tuned)**: 微调用于音频信号的事件特征提取器。 | 对比`Paper-Baseline`的音频部分，证明更先进的音频编码器能带来提升。 |
| | `Sensor-Only` | **Transformer Encoder**: 仅使用`.csv`数据的时间序列特征提取。 | 证明传感器数据本身就包含有效的缺陷信息，是您工作的创新点之一。 |
| **C. 简单融合基线**<br>(Simple Fusion) | `SOTA-Late-Fusion` | 将B组中**四个**SOTA模型的异常分数进行后期加权融合。 | 对比`Paper-Baseline`，证明即使只用简单的后期融合，采用SOTA编码器和新增模态也能大幅提升性能。 |
| | `SOTA-Mid-Fusion` | 将B组中**四个**SOTA模型提取的特征向量进行拼接（Concatenate），再通过MLP进行融合。 | **证明中层特征融合的价值**。对比`SOTA-Late-Fusion`，展示在特征层面进行融合优于在分数层面融合。 |
| **D. 最终提出的SOTA模型**<br>(Proposed SOTA Model) | `Proposed-Deep-Fusion` (您的模型) | **冻结Backbone并微调头部**：采用交叉注意力机制（Cross-Attention）对**四个**SOTA模态的特征进行深度、动态的融合。 | **核心贡献**。对比所有A, B, C组基线，特别是`SOTA-Mid-Fusion`，证明您的深度融合策略能够最有效地利用过程与结果的多模态信息，实现最佳性能。 |

### 技术方案：基于微调的监督对比学习四模态深度融合网络

### **1. 部署模型选型 (Model Selection & Deployment)**

| 组件 | 最终选型 | 预训练数据集 | 选型与**训练策略 (Rationale & Training Strategy)** |
| :--- | :--- | :--- | :--- |
| **实时视频编码器** | `facebook/vjepa2-vitl-fpc64-256` | 无标签视频 (自监督学习) | **SOTA自监督模型**。V-JEPA学习通用时空规律，适合分析焊接**过程**。**策略：冻结（Freeze）其绝大部分参数，只训练头部或最后几层。 |
| 焊后图片编码器 | `facebook/dinov2-base` | 无标签图片 (自监督学习) | 顶级通用视觉特征提取器。DINOv2特征强大，适合分析焊后结果**的静态纹理。\*\*策略：冻结（Freeze）\*\*其绝大部分参数。 |
| **音频编码器** | `MIT/ast-finetuned-audioset-14-14-0.443` | AudioSet (通用音频事件) | **领域相关性强**。适合分辨工业噪声和异常事件。**策略：冻结（Freeze）其绝大部分参数。 |
| 传感器编码器 | `torch.nn.TransformerEncoder` | 从零开始训练 (Train from Scratch) | 标准且强大。捕捉时序依赖的SOTA选择。策略：全量训练（Trainable），因为没有预训练权重。 |
| 核心融合模块 | Cross-Attention Fusion Module | 从零开始训练 | 超越简单融合的SOTA方法。主动查询四个模态，建模过程与结果**的关联。**策略：全量训练（Trainable）**，是学习任务知识的核心。 |
| **核心训练方法** | **Supervised Contrastive Loss (SupConLoss)** | N/A | **构建优质特征空间**。能学习到一个更具区分度的特征空间，为高精度分类和未来潜在的未知缺陷检测奠定基础。 |

### **2. 环境与包选择 (Environment & Package Selection)**

  * **基础环境**:
      * `Python`: 3.10+
      * `CUDA`: 11.8+
      * **环境管理**: Conda
  * **Python包列表 (`requirements.txt` 格式)**:
    ```
    # 核心框架
    torch>=2.1.0
    torchvision
    torchaudio

    # 模型加载与处理
    transformers>=4.30.0
    timm>=0.9.0

    # 数据处理
    pandas>=2.0.0
    scikit-learn>=1.3.0
    librosa>=0.10.0
    opencv-python

    # 实验辅助与效率工具
    einops
    wandb
    tqdm
    numpy
    ```

### **3. 快速开始 (Quick Start)**

1.  **创建并激活Conda环境**:
    ```bash
    conda create -n weld_sota python=3.10
    conda activate weld_sota
    ```
2.  **安装核心依赖 (根据您的CUDA版本调整PyTorch命令)**:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers timm pandas scikit-learn librosa opencv-python einops wandb tqdm numpy
    ```
3.  **预先手动拉取模型 (在有网络的机器上执行)**:
    ```bash
    # 安装 git-lfs
    git lfs install

    # 创建一个文件夹来存放模型
    mkdir -p models
    cd models

    # 克隆V-JEPA模型 (实时视频)
    git clone https://huggingface.co/facebook/vjepa2-vitl-fpc64-256

    # 克隆DINOv2模型 (焊后图片)
    git clone https://huggingface.co/facebook/dinov2-base

    # 克隆AST模型 (音频)
    git clone https://huggingface.co/MIT/ast-finetuned-audioset-14-14-0.443
    ```
    完成后，将整个`models`文件夹拷贝到您的项目目录中。

### **4. 详细实现路径 (Step-by-Step Implementation Path)**

这是一个严谨的、每一步都包含验证的实现流程。

#### **Step 1: 构建统一数据管道 (Data Pipeline)**

  * **任务**: `WeldingDataset` 类的 `__getitem__` 方法需返回一个包含**四种**模态数据和标签的字典。
  * **实现细节**:
      * **新增-焊后图片**: 读取所有（例如5张）多角度静态图片。将它们`resize`到模型输入尺寸（例如224x224），归一化，然后堆叠成一个张量，形状如 `(num_angles, 3, 224, 224)`。
      * **实时视频**: 读取黑白视频，采样固定数量帧，`resize`并归一化。注意通道数为1。
      * **音频/传感器**: 使用Librosa加载音频，重采样到16kHz，计算成梅尔频谱图；使用Pandas读取传感器数据，对齐时间戳，插值缺失值，然后对所有数值列进行Z-score标准化。截取或填充（padding）到固定长度。
  * **✅ 如何测试**: `sample = dataset[0]`，额外打印并检查 `sample['post_weld_images'].shape` 是否符合预期 `(5, 3, 224, 224)`以及其他模态的维度。

#### **Step 2: 封装单模态编码器（并实现冻结逻辑）**

  * **任务**: 封装四个独立的 `nn.Module` 模块，并在初始化时增加参数冻结的逻辑。
  * **实现细节**:
      * 在`VideoEncoder`, `ImageEncoder`, `AudioEncoder`的`__init__`方法中，加载预训练模型后，立即遍历其参数并设置 `param.requires_grad = False`。
    <!-- end list -->
    ```python
    # 示例:
    class VideoEncoder(nn.Module):
        def __init__(self, model_path, freeze=True):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_path)
            if freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
    ```
  * **✅ 如何测试 (单元测试)**:
    1.  实例化模块 `encoder = VideoEncoder(..., freeze=True)`。
    2.  编写一个循环检查 `param.requires_grad` 是否全部为 `False`。
    3.  运行前向传播，确保输出维度正确。

#### **Step 3: 实现并测试核心融合模块**

  * **任务**: 实现 `CrossAttentionFusionModule`。
  * **✅ 如何测试 (单元测试)**: 创建**四个**虚拟的特征序列（即Step 2中编码器的输出），送入融合模块，并`assert`检查最终输出的融合向量维度是否为 `(batch_size, 1, fusion_dim)`，并检查输出值是否为`NaN`或`inf`。

#### **Step 4: 整合并测试完整模型**

  * **任务**: 组装最终的 `QuadModalSOTAModel`。
  * **实现细节**: 在模型初始化时，打印出总参数量和**可训练参数量**。
  * **✅ 如何测试 (集成测试)**:
    1.  实例化完整模型。
    2.  **关键**: 确认打印出的**可训练参数量**远小于总参数量（例如，从5亿降至2千万）。
    3.  从 **Step 1** 的 `DataLoader` 中取出一个真实的批次 `batch`，送入完整模型的`forward`方法，确保包含四种模态的`forward`方法能顺利执行，并能与`SupConLoss`函数无缝对接。

#### **Step 5: 实现并验证训练循环**

  * **任务**: 编写 `train.py` 脚本，包含完整的训练和验证逻辑。
  * **实现细节**: 定义AdamW优化器，学习率调度器，使用`wandb`或`Tensorboard`记录损失、学习率等指标，并保存模型检查点。
  * **✅ 如何测试 (过拟合测试)**:
    1.  创建一个仅包含少量数据（例如，每个类别2个样本，共24个样本）的`DataLoader`。
    2.  使用这个`DataLoader`训练您的完整模型几个Epoch。
    3.  **预期结果**: 训练损失应该能迅速下降到接近0。如果损失不下降，说明您的模型结构、梯度流或训练循环中存在bug。这是一个定位问题的黄金标准测试。

#### **Step 6: 实现并验证评估协议**

  * **任务**: 编写 `evaluate.py` 脚本。
  * **实现细节**:
    1.  加载训练好的模型检查点。
    2.  **构建特征库**: 遍历**训练集**，用模型提取所有样本的特征，并与标签一同保存。
    3.  **进行评估**: 遍历**测试集**，提取每个样本的特征，然后使用`scikit-learn`的 `KNeighborsClassifier` 在特征库上进行k-NN分类，计算准确率、F1分数等指标。
  * **✅ 如何测试**:
    1.  使用**Step 5**中过拟合的模型和那24个样本。
    2.  用这24个样本构建特征库，再用它们自己作为测试集。
    3.  **预期结果**: k-NN分类的准确率应该为100%。如果不是，说明您的特征提取或k-NN实现逻辑有误。

完成以上所有步骤和测试，您将拥有一个结构清晰、经过充分验证、技术先进且**参数量合理、可训练**的SOTA研究项目代码库。

## 快速测试与训练

### 本地快速检测: 数据集管线单元测试

项目包含一个轻量级的 `WeldingDataset` 的 dummy 实现，用于在没有完整依赖和数据的情况下快速验证数据管线。
运行方法（在装好 `pytest` 与 `numpy` 的环境下）:

```bash
python -m pip install pytest numpy
python -m pytest -q
```

测试文件：`tests/test_dataset.py`。该测试仅依赖于 `numpy` 和 `pytest`，并不会读取真实数据。

### 服务器环境完整测试

在服务器上运行完整测试（需要PyTorch + CUDA）:

```bash
# 一键运行所有测试
bash scripts/test_server.sh

# 或分步运行
python tests/test_dataset_labels.py    # 检查真实数据标签分布
python tests/test_loss_and_labels.py   # 测试loss函数和采样器
python src/train.py --quick-test --debug   # 快速训练测试
bash scripts/evaluate.sh
```

### 正式训练

```bash
# 使用默认配置（微调策略，CUDA自动检测，batch_size=2）
python src/train.py

# 指定参数
python src/train.py --batch-size 16 --num-epochs 100

# 混合精度训练（推荐GPU环境）
python src/train.py --batch-size 32 --mixed-precision

# 调试模式（查看第一个batch的详细信息并检查可训练参数）
python src/train.py --debug
```