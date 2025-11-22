# CHANGELOG

This file consolidates recent updates and code-change notes. It is a curated, human-friendly timeline and summary—original detailed docs are archived under `docs/archive/`.

## Summary (Top-level)

### V7.1 — Mixed Pooling for Small Defect Detection (2025-11-21)

- **Problem**: Global Average Pooling on Layer 4 diluted small defects (Cracks), leading to low AUROC (0.78).
- **Fix**: Implemented **Mixed Pooling** in `ImageEncoder`.
    - **Layer 12**: Keeps **Mean Pooling** (for global structure/warping).
    - **Layer 4**: Switches to **Max Pooling** (to capture peak anomalies like cracks).
    - **Fusion**: Concatenates `[Mean-L12, Max-L4]`.
- **Goal**: Improve sensitivity to small, local defects while maintaining global structural awareness.

### V7 — Multi-Scale Global Reconstruction (2025-11-21)

- **Core Logic Change**: Shifted from Patch-based reconstruction (which caused Mode Collapse) to **Multi-Scale Global Reconstruction**.
- **L0 (Backbone)**: Modified `ImageEncoder` to extract features from **Layer 4** (texture) and **Layer 12** (semantics), apply Global Average Pooling, and concatenate them (1536 dim).
- **L2 (Result Encoder)**: Updated `ResultEncoder` to take the 1536-dim multi-scale vector, pool over views, and project to 128-dim `Z_result`.
- **L3 (Decoder)**: Reverted to a global `CausalDecoder` (MLP-based) that reconstructs the 128-dim global vector.
- **L4 (Loss/Inference)**: Reverted to global cosine similarity loss and anomaly scoring.
- **Reasoning**: Layer 4 captures low-level defects (cracks, porosity) while Layer 12 captures structural anomalies. Fusing them forces the model to learn a complete global signature without the difficulty of spatial hallucination.

### V6 — Spatial/Patch-based Anomaly Detection (2025-11-21)

- **Core Logic Change**: Shifted from Global feature vector to Spatial/Patch-based feature vectors.
- **L0 (Backbone)**: Modified `ImageEncoder` to extract patch tokens from DINOv2 instead of CLS token.
- **L2 (Result Encoder)**: Implemented `SpatialResultEncoder` to project patches while maintaining spatial dimensions.
- **L3 (Decoder)**: Implemented `SpatialCausalDecoder` with learnable positional embeddings to generate spatial image features from global process vector.
- **L4 (Loss/Inference)**:
    - Training: Per-patch reconstruction loss.
    - Inference: Max-pooling of patch errors for anomaly scoring (sensitive to small defects like Undercut).

### V5 — Causal-FiLM Implementation (2025-11-10)

- **Major Architectural Shift**: Transitioned from supervised contrastive learning to unsupervised anomaly detection
- **Model**: Implemented Causal-FiLM (Causal-Hierarchical Fusion with Sensor-Modulation)
- **Key Innovation**: 
  - L1: FiLM sensor modulation (gamma/beta conditioning)
  - L2: Causal hierarchical encoders (Process + Result)
  - L3: Anti-generalization decoder with Linear Attention
  - L4: Reconstruction loss + CLIP text constraint
- **Training**: Only normal samples used (unsupervised anomaly detection paradigm)
- **Evaluation**: Anomaly scores via reconstruction error (1 - cosine_similarity)

### V4 — Early Stop & Feature-Level MixUp (2025-10-21)

- Date: 2025-10-21
- Release: V4 — Early Stop & Feature-Level MixUp
- Goal: Fix early-overfitting where best validation models appear at Epoch 4-10 and later epochs overfit.
- Key strategies:
  - Feature-level MixUp (config `use_mixup: True`, `mixup_alpha: 0.2`)
  - Faster warmup and learning rate adjustment (`learning_rate: 5e-5`, `warmup_epochs: 5`)
  - Stronger L2 (`weight_decay: 1e-2`) and dropout tuning
  - Aggressive Early Stopping (`early_stopping_patience: 8`)
  - Removed: Label Smoothing (user-requested removal — no longer applied in SupCon)

## Timeline

### 2025-11-11 — Evaluation Metrics Refactoring

#### Problem Identified
- **Issue**: P-AUROC 与 I-AUROC 值完全相同，引发对指标准确性的质疑
- **Root Causes**:
  1. 数据集只有图像级标签（正常/异常），无像素级缺陷掩码
  2. 模型输出全局嵌入向量，无空间特征图
  3. 像素级异常图使用全局分数均匀填充，无真实空间变化
  4. P-AUROC/P-AUPRO 基于图像级标签的近似计算，不反映真实像素定位能力

#### Solution: Remove Unreliable Pixel-Level Metrics

**Removed Metrics** (不准确的近似值):
- ❌ P-AUROC (Pixel-level AUROC)
- ❌ P-AUPRO@30% / @10% / @5% / @1%
- ❌ `_generate_pixel_anomaly_maps()` 方法
- ❌ `compute_pro_metric()` 方法
- ❌ `compute_metrics_with_pro()` 方法

**Implemented Reliable Image-Level Metrics**:
- ✅ **I-AUROC / AUC**: Area Under ROC Curve (主指标)
- ✅ **I-AP / I-mAP**: Average Precision
- ✅ **Accuracy (Acc)**: Overall classification accuracy
- ✅ **F1-Score (F1)**: Harmonic mean of precision and recall
- ✅ **FDR**: False Discovery Rate = FP / (FP + TP)
- ✅ **MDR**: Missed Detection Rate = FN / (FN + TP)
- ✅ **Precision / Recall**: At optimal threshold (Youden's J)
- ✅ **Confusion Matrix**: TP, FP, TN, FN

#### Code Changes

**File: `src/evaluate_causal_film.py`**
- Simplified `extract_anomaly_scores()`: 只提取图像级分数
- Rewrote `compute_metrics()`: 计算可靠的图像级指标
- Updated output format: 清晰展示所有图像级指标
- Removed pixel-level approximations: 删除所有不可靠的像素级计算

**Documentation Updates**:
- File header: 明确说明只计算图像级指标
- Removed confusing disclaimers about P-AUROC/P-AUPRO
- Focus on reliable metrics for image-level anomaly detection

#### Why This Is Correct

**可靠的指标** (Image-Level):
- I-AUROC 准确反映模型区分正常/异常图像的能力
- 有明确的图像级标签作为ground truth
- 标准的二分类评估方法

**不可靠的指标** (Pixel-Level):
- 需要像素级缺陷掩码（我们没有）
- 基于图像级标签的近似 → 无法评估定位能力
- P-AUROC ≈ I-AUROC 是必然结果，不是bug

#### Future Work (如需像素级评估)
1. 标注像素级缺陷分割掩码
2. 修改模型保留空间特征（patch-level features）
3. 计算每个patch的重建误差
4. 使用注意力图进行弱监督定位

### 2025-11-10 — Causal-FiLM Implementation

- **New Modules**:
  - `src/models/film_modulation.py`: SensorModulator with GRU-based FiLM parameter generation
  - `src/models/causal_encoders.py`: ProcessEncoder (cross-attention) + ResultEncoder (MLP)
  - `src/models/causal_decoder.py`: CausalDecoder with LinearAttention and NoisyBottleneck
  - `src/models/causal_film_model.py`: Complete Causal-FiLM architecture (L0-L3)
  
- **New Loss Functions**:
  - `src/losses.py`: Added ReconstructionLoss, CLIPTextLoss, CausalFILMLoss
  
- **New Training/Evaluation**:
  - `src/train_causal_film.py`: Unsupervised training (normal samples only)
  - `src/evaluate_causal_film.py`: Anomaly detection metrics
    - **I-AUROC**: Image-level detection AUROC
    - **P-AUPRO**: Pixel-level segmentation AUPRO (Per-Region Overlap)
    - Supports multiple FPR thresholds (@0.3, @0.1, @0.05, @0.01)
  
- **Configuration**:
  - `configs/model_config.py`: Added CAUSAL_FILM_CONFIG
  - `configs/train_config.py`: Added lambda_text parameter
  
- **Scripts**:
  - `scripts/train_causal_film.sh`: Training script for Causal-FiLM
  - `scripts/evaluate_causal_film.sh`: Evaluation script for Causal-FiLM

- 2025-10-21 15:00 — V4 strategy implemented
  - Add Feature-Level MixUp
  - Adjust LR scheduling and warmup
  - Implement aggressive Early Stopping
  - Add validation & verification scripts

- 2025-10-21 16:00 — Label Smoothing removed
  - Per user decision, `label_smoothing` parameter and related logic removed from configs and loss implementation
  - Keep MixUp + Early Stop
  - Updated docs and verification scripts

## Code-level highlights

### V7.1 (Mixed Pooling for Small Defect Detection)

- **Architecture Design**:
  - Backbone: DINOv2 for multi-scale feature extraction
  - Unified feature dimension (d_model=128)
  - ResultEncoder: projects multi-scale vector to 128-dim `Z_result`
  - CausalDecoder: reconstructs global vector from `Z_result`
  - Anti-generalization: Linear attention + noisy bottleneck prevents overfitting to anomalies

- **Loss Function**:
  - L_total = L_recon + λ * L_text
  - L_recon: Cosine distance between Z_result and Z_result_pred
  - L_text: CLIP-based semantic constraint ("a normal weld")
  - λ = 0.1 (default)

- **Training Strategy**:
  - Only normal samples in training set
  - Test set includes both normal and anomalies
  - Anomaly score = 1 - cos_sim(Z_result, Z_result_pred)
  - Higher score = anomaly

### V7 (Multi-Scale Global Reconstruction)

- **Architecture Design**:
  - Backbone: DINOv2 for multi-scale feature extraction
  - Unified feature dimension (d_model=128)
  - ResultEncoder: projects multi-scale vector to 128-dim `Z_result`
  - CausalDecoder: reconstructs global vector from `Z_result`
  - Anti-generalization: Linear attention + noisy bottleneck prevents overfitting to anomalies

- **Loss Function**:
  - L_total = L_recon + λ * L_text
  - L_recon: Cosine distance between Z_result and Z_result_pred
  - L_text: CLIP-based semantic constraint ("a normal weld")
  - λ = 0.1 (default)

- **Training Strategy**:
  - Only normal samples in training set
  - Test set includes both normal and anomalies
  - Anomaly score = 1 - cos_sim(Z_result, Z_result_pred)
  - Higher score = anomaly

### V6 (Spatial/Patch-based Anomaly Detection)

- **Architecture Design**:
  - Backbone: DINOv2 for patch token extraction
  - Unified patch dimension (d_model=128)
  - SpatialResultEncoder: projects patches while preserving spatial structure
  - SpatialCausalDecoder: generates spatial features with learnable positional embeddings
  - Anti-generalization: Linear attention + noisy bottleneck prevents overfitting to anomalies

- **Loss Function**:
  - L_total = L_recon + λ * L_text
  - L_recon: Cosine distance between Z_result and Z_result_pred
  - L_text: CLIP-based semantic constraint ("a normal weld")
  - λ = 0.1 (default)

- **Training Strategy**:
  - Only normal samples in training set
  - Test set includes both normal and anomalies
  - Anomaly score = 1 - cos_sim(Z_result, Z_result_pred)
  - Higher score = anomaly

### V5 (Causal-FiLM)

- **Architecture Design**:
  - Frozen backbones (V-JEPA, DINOv2, AST) for feature extraction
  - Lightweight trainable components (~1-2M parameters)
  - Unified feature dimension (d_model=128)
  - FiLM modulation: sensor data as context, not fusion target
  - Causal hierarchy: Process (video+audio) → Result (images)
  - Anti-generalization: Linear attention + noisy bottleneck prevents overfitting to anomalies

- **Loss Function**:
  - L_total = L_recon + λ * L_text
  - L_recon: Cosine distance between Z_result and Z_result_pred
  - L_text: CLIP-based semantic constraint ("a normal weld")
  - λ = 0.1 (default)

- **Training Strategy**:
  - Only normal samples in training set
  - Test set includes both normal and anomalies
  - Anomaly score = 1 - cos_sim(Z_result, Z_result_pred)
  - Higher score = anomaly

### V4 (SupCon + MixUp)

- configs/train_config.py
  - Added/kept: `use_mixup`, `mixup_alpha`, `early_stopping_patience`, altered lr/warmup/min_lr/weight_decay
  - Removed: `label_smoothing` (no longer present)

- src/train.py
  - Added: `_mixup_features()` implementation (feature-level MixUp)
  - Integrated MixUp into training loop when `use_mixup=True`
  - `_setup_loss()` no longer passes `label_smoothing` to SupConLoss

- src/losses.py
  - Restored to standard SupConLoss (no label smoothing) — behavior identical to original when `label_smoothing` absent

- scripts/verify_v4_config.sh
  - Script added/updated to verify critical config keys and V4 readiness

### Fixes (2025-11-21)
- **Fixed Shape Mismatch in CausalFiLMModel**: Resolved `ValueError: not enough values to unpack` by correctly handling 3D input `(B, Total_Patches, D)` from `ImageEncoder` and removing incorrect 4D unpacking.
- **Updated DummyImageEncoder**: Changed output size to 16x16 patches (256 total) to match DINOv2 behavior and `CausalFiLMModel` expectations.
- **Dynamic Patch Count**: Updated `CausalFiLMModel` to calculate `total_patches` based on aggregation method ("mean" -> 256, "concat" -> N*256).

## How to use

### For Mixed Pooling for Small Defect Detection (V7.1)

```bash
# Training
bash scripts/train_causal_film.sh

# Evaluation
bash scripts/evaluate_causal_film.sh /path/to/checkpoint.pth
```

### For Multi-Scale Global Reconstruction (V7)

```bash
# Training
bash scripts/train_causal_film.sh

# Evaluation
bash scripts/evaluate_causal_film.sh /path/to/checkpoint.pth
```

### For Spatial/Patch-based Anomaly Detection (V6)

```bash
# Training
bash scripts/train_causal_film.sh

# Evaluation
bash scripts/evaluate_causal_film.sh /path/to/checkpoint.pth
```

### For Causal-FiLM (V5)

```bash
# Training
bash scripts/train_causal_film.sh

# Evaluation
bash scripts/evaluate_causal_film.sh /path/to/checkpoint.pth
```

### For SupCon (V4)

- Keep top-level docs small. Recommended docs to read:
  - `README.md` (root)
  - `docs/QUICKSTART.md`
  - `docs/CHANGELOG.md` (this file)

- All other historical/auxiliary docs are copied into `docs/archive/` for reference.

## Files archived

The following files were copied into `docs/archive/` (originals remain in `docs/` until you choose to delete them):

- ANTI_OVERFITTING_V3.md
- ANTI_OVERFITTING_V4_EARLY_STOP.md
- LOGGING_AND_VISUALIZATION.md
- PROGRESS_QUADMODAL.md
- PROJECT_STRUCTURE.md
- STEP3_FUSION_REPORT.md
- TRAINABLE_MODULES_DROPOUT.md
- TRAINING_LOG_20251020.md
- TRAINING_OPTIMIZATION.md
- V4_CODE_CHANGES.md
- UPDATE_2025-10-21_V4_EARLY_STOP.md

(You can remove the originals from `docs/` after verifying the archive. Recommended: keep `QUICKSTART.md` and `README.md` in their places.)

## Notes

- I did not delete any original files. I only created this consolidated changelog and archived copies. Please confirm which originals you want removed; I list recommended deletions below for your convenience.

---

Generated: 2025-10-30
Branch: main
