| 类别 (Category) | 模型/基线名称 (Model/Baseline Name) | 核心技术 (Core Technology) | 实验目的 / 对比对象 (Experimental Purpose / Comparison Target) |
| :--- | :--- | :--- | :--- |
| **A. 论文复现基线**<br>(Paper Reproduction) | `Paper-Baseline (Reproduced)` | Audio 1D CNN-AE + Video Slowfast-AE，然后进行后期分数融合 (Late Fusion) 。 | **最重要的基准**。验证您的实验环境，并提供一个必须超越的、最直接的性能参照点。 |
| **B. 单模态SOTA基线**<br>(Single-Modality SOTA) | `In-Process-Video-Only` | **V-JEPA**: 用于实时黑白视频的自监督时空特征提取。 | 证明先进的自监督视频编码器在过程监控中的优势。 |
| | `Post-Weld-Image-Only` | **DINOv2**: 用于焊后多角度静态图片的自监督外观特征提取。 | **衡量新模态的独立贡献**。证明最终成品外观本身就包含了丰富的缺陷信息。 |
| | `Audio-Only` | **AST**: 用于音频信号的事件特征提取。 | 对比`Paper-Baseline`的音频部分，证明更先进的音频编码器能带来提升。 |
| | `Sensor-Only` | **Transformer Encoder**: 仅使用`.csv`数据的时间序列特征提取。 | 证明传感器数据本身就包含有效的缺陷信息，是您工作的创新点之一。 |
| **C. 简单融合基线**<br>(Simple Fusion) | `SOTA-Late-Fusion` | 将B组中**四个**SOTA模型的异常分数进行后期加权融合。 | 对比`Paper-Baseline`，证明即使只用简单的后期融合，采用SOTA编码器和新增模态也能大幅提升性能。 |
| | `SOTA-Mid-Fusion` | 将B组中**四个**SOTA模型提取的特征向量进行拼接（Concatenate），再通过MLP进行融合。 | **证明中层特征融合的价值**。对比`SOTA-Late-Fusion`，展示在特征层面进行融合优于在分数层面融合。 |
| **D. 最终提出的SOTA模型**<br>(Proposed SOTA Model) | `Proposed-Deep-Fusion` (您的模型) | 采用交叉注意力机制（Cross-Attention）对**四个**SOTA模态的特征进行深度、动态的融合。 | **核心贡献**。对比所有A, B, C组基线，特别是`SOTA-Mid-Fusion`，证明您的深度融合策略能够最有效地利用过程与结果的多模态信息，实现最佳性能。 |

### **最终SOTA技术方案：基于监督对比学习的四模态深度融合网络**

### **1. 部署模型选型 (Model Selection & Deployment)**

| 组件 | 最终选型 | 预训练数据集 | 选型理由 (Rationale) |
| :--- | :--- | :--- | :--- |
| **实时视频编码器** | `facebook/vjepa2-vitl-fpc64-256` | 无标签视频 (自监督学习) | **SOTA自监督模型**。V-JEPA学习的是通用的视觉时空规律，非常适合分析焊接**过程**中的动态变化。 |
| **焊后图片编码器** | `facebook/dinov2-base` | 无标签图片 (自监督学习) | **顶级通用视觉特征提取器**。DINOv2在海量无标签图片上学习，其特征对于各种下游任务都极为强大，非常适合分析焊后**结果**的静态外观纹理。 |
| **音频编码器** | `MIT/ast-finetuned-audioset-14-14-0.443` | AudioSet (通用音频事件) | **领域相关性强**。在通用音频事件数据集AudioSet上预训练，非常适合用于分辨焊接过程中的工业噪声和异常事件。 |
| **传感器编码器** | `torch.nn.TransformerEncoder` | 从零开始训练 (Train from Scratch) | **标准且强大**。对于多变量时间序列，Transformer编码器是当前捕捉长距离依赖和变量间复杂交互的SOTA选择。 |
| **核心融合模块** | **Cross-Attention Fusion Module** | 从零开始训练 | **超越简单融合的SOTA方法**。通过可学习的`[FUSION_TOKEN]`主动查询**四个**模态，能动态建模**过程与结果**之间细粒度的、非线性的关联。 |
| **核心训练方法** | **Supervised Contrastive Loss (SupConLoss)** | N/A | **构建优质特征空间**。能学习到一个更具区分度的特征空间，将同类样本拉近、异类样本推远，为高精度分类和未来潜在的未知缺陷检测奠定基础。 |

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
      * **音频/传感器**: 保持不变。
  * **✅ 如何测试**: `sample = dataset[0]`，额外打印并检查 `sample['post_weld_images'].shape` 是否符合预期 `(5, 3, 224, 224)`。

#### **Step 2: 封装单模态编码器**

  * **任务**: 增加并封装 `ImageEncoder` 模块。
  * **实现细节**:
      * **ImageEncoder**: 内部使用 `AutoModel.from_pretrained("./models/dinov2-base")` 加载模型。其`forward`方法接收一批图片 `(batch_size, num_angles, 3, 224, 224)`，需要先将其reshape为 `(batch_size * num_angles, 3, 224, 224)`，送入模型，然后对同一焊件的多个角度特征进行聚合（例如取平均或最大池化），最终为每个焊件输出一个特征序列。
  * **✅ 如何测试 (单元测试)**: 为 `ImageEncoder` 编写测试脚本，检查输入一批多角度图片后，输出的特征维度是否正确 `(batch_size, expected_seq_len, expected_dim)`。

#### **Step 3: 实现并测试核心融合模块**

  * **任务**: 更新 `CrossAttentionFusionModule` 以处理**四个**输入特征序列。
  * **实现细节**: 在`forward`方法中增加一个对`image_features`的交叉注意力步骤。
  * **✅ 如何测试 (单元测试)**: 创建**四个**虚拟的特征序列，送入融合模块，并`assert`检查输出维度。

#### **Step 4: 整合并测试完整模型**

  * **任务**: 组装最终的 `QuadModalSOTAModel`。
  * **✅ 如何测试 (集成测试)**: 从`DataLoader`中取出一个真实的批次`batch`，送入完整模型，确保包含四种模态的`forward`方法能顺利执行，并能与`SupConLoss`函数无缝对接。

#### **Step 5: 实现并验证训练循环**

  * **任务**: 保持不变，该步骤与模型内部结构解耦。
  * **✅ 如何测试 (过拟合测试)**: 保持不变。用一个包含所有四种模态的极小数据集进行测试，确保损失能快速收敛。

#### **Step 6: 实现并验证评估协议**

  * **任务**: 保持不变，评估逻辑作用于模型最终输出的特征向量，与输入模态数量无关。
  * **✅ 如何测试**: 保持不变。

完成以上所有步骤和测试，您将拥有一个结构清晰、经过充分验证、技术先进且极具竞争力的SOTA研究项目代码库。

-----

## 📦 实现状态

### ✅ Step 1 完成: 数据管线

**功能**: 四模态数据加载（实时视频+焊后图片+音频+传感器）、标签解析
**测试**: `bash scripts/test_dataset.sh`

### ✅ Step 2 完成: 单模态编码器

**功能**: VideoEncoder (V-JEPA) + ImageEncoder (DINOv2) + AudioEncoder (AST) + SensorEncoder (Transformer)
**测试**: `bash scripts/test_encoders.sh`

**详细进度**: 见 `PROGRESS.md`

-----

## 本地快速检测: 数据集管线单元测试

项目包含一个轻量级的 `WeldingDataset` 的 dummy 实现，用于在没有完整依赖和数据的情况下快速验证数据管线。

运行方法（在装好 `pytest` 与 `numpy` 的环境下）:

```bash
python -m pip install pytest numpy
python -m pytest -q
```

测试文件：`tests/test_dataset.py`。该测试仅依赖于 `numpy` 和 `pytest`，并不会读取真实数据。