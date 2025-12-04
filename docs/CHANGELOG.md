# CHANGELOG

This file consolidates recent updates and code-change notes. It is a curated, human-friendly timeline and summary—original detailed docs are archived under `docs/archive/`.

## Summary (Top-level)

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

### 2025-12-04 — Unsupervised Dataset Resplit

#### Goal
Strictly enforce unsupervised learning paradigm by ensuring the training set contains **only** normal ("Good") samples, while validation and test sets contain both normal and defective samples for realistic evaluation.

#### Changes
- **Dataset Resplit**:
  - Updated `scripts/resplit_dataset.py` with `--unsupervised` mode.
  - **New Split Statistics**:
    - **Train**: 576 samples (All "Good").
    - **Val**: 1732 samples (122 "Good" + 1610 "Defective").
    - **Test**: 1732 samples (121 "Good" + 1611 "Defective").
  - This ensures zero leakage of anomalies into the training set.
- **Training Script Update**:
  - Modified `src/train_causal_film.py` to use `split='val'` for the validation data loader (previously used `split='test'`).
  - This aligns the training validation step with the new explicit validation set.

### 2025-12-02 — Dimension Upgrade & Selective Dropout

#### Goal
Address the performance drop (AUC ~0.6) caused by excessive dropout regularization leading to underfitting.

#### Changes
- **Increased Model Capacity**:
  - Upgraded `d_model` from 256 to **512** in `configs/model_config.py` and all model components.
  - This provides more redundant capacity to handle the noise introduced by dropout.
- **Selective Dropout**:
  - **Removed Dropout** from `video_projector` and `audio_projector` in `src/models/causal_film_model.py`.
  - **Removed Dropout** from `RobustResultEncoder` in `src/models/causal_encoders.py`.
  - **Kept Dropout** only in `AntiGenDecoder` (src/models/causal_decoder.py) to maintain the "Noisy Bottleneck" effect without destroying the feature adaptation process.

### 2025-12-02 — Dropout Regularization

#### Goal
Prevent overfitting to the training set (normal samples) and improve the model's ability to generalize to unseen normal samples, thereby reducing false positives.

#### Changes
- **Added Dropout (p=0.1)** to key trainable components:
  - **Projectors**: Added to `video_projector` and `audio_projector` MLPs in `src/models/causal_film_model.py`.
  - **Result Encoder**: Added to `RobustResultEncoder.projector` in `src/models/causal_encoders.py`.
  - **Decoder**: Added to `AntiGenDecoder.net` in `src/models/causal_decoder.py` (restoring the "Noisy Bottleneck" effect).

### 2025-12-02 — MLP Projection Layers

#### Goal
Improve the model's ability to adapt pre-trained features to the causal space by increasing the capacity of the projection layers.

#### Changes
- **Architecture Upgrade**:
  - Replaced single-layer `nn.Linear` projectors with **MLP (Multi-Layer Perceptron)** blocks in `src/models/causal_film_model.py`.
  - **New Structure**: `Linear(Input -> 512) -> LayerNorm -> GELU -> Linear(512 -> d_model)`.
  - Applied to both `video_projector` (1024 -> d_model) and `audio_projector` (768 -> d_model).
  - `image_projector` remains `nn.Identity()` as the image features are handled by the `RobustResultEncoder`.

### 2025-12-02 — Hard Feature Mining & Loss Optimization

#### Goal
Enhance the model's ability to detect subtle anomalies by focusing on hard-to-reconstruct features and stabilizing training.

#### Changes
- **Hard Feature Mining**:
  - Implemented Top-K L1 Loss in `src/losses.py`.
  - Added `top_k_ratio` (default 0.5) to `configs/train_config.py`.
  - During training, only the top 50% of feature dimensions with the highest reconstruction error contribute to the L1 loss, aligning training objective with the inference anomaly score logic.
- **Loss Weight Parameterization**:
  - Extracted the hardcoded L1 loss weight (10.0) into `configs/train_config.py` as `lambda_l1`.
  - Allows flexible tuning of the balance between Cosine Distance (Direction) and L1 Error (Intensity).
- **Training Stability**:
  - Increased `warmup_epochs` from 2 to 5 in `configs/train_config.py`.
  - Provides a longer warmup period for the projection layers to stabilize before full optimization, reducing the risk of early divergence or overfitting.

### 2025-11-30 — Loss & Hyperparameter Optimization

#### Goal
Further optimize the Causal-FiLM model for better training stability and decoder capacity, aiming for higher AUROC.

#### Changes
- **Loss Function**:
  - **Training**: Switched from `L1 Loss` to `Smooth L1 Loss` (Huber Loss) in `src/losses.py`. This provides more stable gradients when errors are small, preventing oscillation near the optimum.
  - **Inference**: Anomaly score calculation remains `L1 Loss` (Absolute Distance) + Cosine Distance, as this "hard" metric is better for detecting anomalies.
- **Hyperparameters**:
  - **CLIP Constraint**: Reduced `lambda_text` from `0.1` to `0.01` in `configs/train_config.py`. This weakens the semantic constraint, allowing the decoder more freedom to learn subtle feature reconstructions without being overly pulled towards the generic "normal weld" text embedding.

### 2025-11-29 — Dimension Upgrade (Z_result -> 256)

#### Goal
Increase the capacity of the Causal-FiLM model by upgrading the unified feature dimension (`d_model`) from 128 to 256.

#### Changes
- **Configuration**: Verified `configs/model_config.py` sets `d_model: 256`.
- **Codebase Defaults**: Updated default `d_model` values from 128 to 256 in:
  - `src/models/causal_film_model.py` (Model & Factory)
  - `src/models/causal_encoders.py` (ProcessEncoder, ResultEncoder, Dummy encoders)
  - `src/models/film_modulation.py` (SensorModulator, DummySensorModulator)
  - `src/models/causal_decoder.py` (CausalDecoder, AntiGenDecoder, DummyCausalDecoder)
  - `src/losses.py` (CausalFILMLoss, CLIPTextLoss)

#### Benefit
- **Higher Capacity**: Allows the model to capture more detailed features from the multimodal inputs.
- **Consistency**: Ensures that all components (encoders, decoders, loss functions) default to the same dimension, reducing the risk of dimension mismatch errors if config is missing.

### 2025-11-29 — Unified Training Entry Point

#### Goal
Align the training logic with the configuration `loss_type: "causal_film"`. Ensure that running `src/train.py` respects this configuration and launches the appropriate trainer.

#### Changes
- **Configuration**: Updated `configs/train_config.py` to set default `"loss_type": "causal_film"`.
- **Unified Entry Point**: Modified `src/train.py` to:
  - Import `CausalFiLMTrainer` from `src/train_causal_film.py`.
  - Check `config["loss_type"]` in `main()`.
  - Automatically switch to `CausalFiLMTrainer` if `loss_type` is `"causal_film"`.
  - Fallback to standard `Trainer` for `"supcon"` or `"combined"`.

#### Benefit
- Simplifies usage: Users can now use `src/train.py` as the single entry point for all training modes.
- Consistency: The `loss_type` config now directly controls the training behavior across the board.

### 2025-11-28 — SensorModulator Upgrade (Mamba)

#### Goal
Enhance the temporal modeling capability of the sensor modality by replacing the GRU-based modulator with a Mamba (State Space Model) based architecture.

#### Changes
- **Upgraded `SensorModulator`**:
  - Replaced `nn.GRU` with `Mamba` (State Space Model) blocks.
  - **Architecture**:
    - Input: Sensor data (B, T, 6).
    - Embedding: Linear projection to hidden dimension.
    - Encoder: Stacked `MambaBlock` layers (Mamba + Residual + LayerNorm).
    - Pooling: Last token extraction (causal context).
    - Output: Linear projection to FiLM parameters (gamma, beta).
  - **Configuration**: Added `configs/mamba_config.py` with default Mamba settings (`d_state=16`, `d_conv=4`, `expand=2`).
  - **Dependency**: Added `mamba-ssm` and `causal-conv1d` to requirements.

### 2025-11-28 — RobustResultEncoder Upgrade (Adaptive Gated Fusion)

#### Goal
Optimize "Plan E" architecture to reach SOTA (Target: 92% AUC) by addressing the bottleneck in static fusion.

#### Changes
- **Upgraded `RobustResultEncoder`**:
  - Replaced static `Concat` fusion with **Adaptive Gated Fusion** (Channel-wise Attention).
  - **Mechanism**:
    - Input: Concatenated L12 (Structure) and L8 (Texture) features (1536 dim).
    - Gate Network: `Linear(1536->384) -> ReLU -> Linear(384->1536) -> Sigmoid`.
    - Fusion: Element-wise modulation of concatenated features by learned gates.
  - **Projector Update**: Switched to `SiLU` activation and updated output dimension to 256.
  - **Benefit**: Allows the model to dynamically weight Structure vs. Texture features, improving performance on geometric defects like "Convexity".
  - **Fix**: Updated `CausalFILMLoss` initialization in `train_causal_film.py` to dynamically use `d_model` from the model config (256), resolving dimension mismatch error.

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

### 2025-11-25 — Late Fusion Strategy (Plan E + Video AE)

#### Goal
Reach SOTA by combining Causal-FiLM (Plan E) with a dedicated Video Autoencoder for Convexity detection.

#### Changes
- **New Model**: `SimpleVideoAE` (MLP Autoencoder) for video features.
  - Input: DINOv2 Layer 12 Mean Features (768 dim).
  - Architecture: 768 -> 128 -> 64 -> 128 -> 768.
- **New Scripts**:
  - `src/train_video_ae.py`: Trains the Video AE on normal samples.
  - `src/evaluate_fusion.py`: Evaluates fusion of Causal-FiLM and Video AE.
- **Fusion Logic**:
  - Standardize scores (Z-score normalization) from both models.
  - Sum standardized scores: `Final = Z_A (Causal) + Z_B (VideoAE)`.

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

### 2025-11-27 — WandB Integration

#### Feature Added
- **WandB Support**: Integrated Weights & Biases (wandb) for experiment tracking and visualization.
- **Files Modified**:
  - `src/train_causal_film.py`: Added wandb initialization, logging of training/validation metrics, and argument parsing.
  - `scripts/train_causal_film.sh`: Updated to pass wandb arguments (`--wandb`, `--wandb_project`, `--wandb_name`).
  - `configs/train_config.py`: Added default wandb configuration options.
- **Usage**:
  - Run `bash scripts/train_causal_film.sh` to start training with wandb logging enabled.
  - Metrics logged: Loss (Total, Recon Cosine, Recon L1, CLIP Text), Learning Rate, AUROC, Anomaly Scores.

### 2025-11-27 — Fix ModuleNotFoundError in Video AE Training

#### Bug Fix
- **Issue**: `ModuleNotFoundError: No module named 'configs.video_ae_config'` when running `scripts/train_video_ae.sh`.
- **Fix**: 
  - Updated `scripts/train_video_ae.sh` to explicitly export `PYTHONPATH`.
  - Added `configs/__init__.py` to ensure `configs` is treated as a proper Python package.

### 2025-11-27 — WandB Integration for Video AE

#### Feature Added
- **WandB Logging**: Integrated Weights & Biases (WandB) for Video Autoencoder training.
- **Configuration**: Added `use_wandb`, `wandb_project`, `wandb_entity`, and `wandb_run_name` to `configs/video_ae_config.py`.
- **Implementation**:
  - Initialized WandB in `src/train_video_ae.py`.
  - Logged training loss (batch and epoch level) to WandB.

### 2025-11-27 — Fix PyTorch 2.6+ Checkpoint Loading

#### Bug Fix
- **Issue**: `_pickle.UnpicklingError` due to PyTorch 2.6+ defaulting `weights_only=True` in `torch.load`, which blocks numpy scalars.
- **Fix**: Updated `torch.load` calls in `src/evaluate_fusion.py` to use `weights_only=False`.
- **Improvement**: Updated `scripts/evaluate_fusion.sh` to explicitly export `PYTHONPATH`.

### 2025-11-27 — Fix CausalFiLMModel Forward Call in Evaluation

#### Bug Fix
- **Issue**: `TypeError: CausalFiLMModel.forward() takes from 2 to 3 positional arguments but 5 were given` in `src/evaluate_fusion.py`.
- **Fix**: Updated `src/evaluate_fusion.py` to pass inputs as a dictionary to `model_a`, matching the `CausalFiLMModel.forward` signature.

## Code-level highlights

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

## How to use

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

### 2025-12-03 — Feature Noise Injection

#### Goal
Prevent the model from overfitting to the reconstruction task (where `recon_l1` drops too low) by injecting noise into the features during training.

#### Changes
- **Configuration**:
  - Added `"feature_noise": 0.01` to `configs/train_config.py`.
- **Model Architecture**:
  - Updated `CausalFiLMModel` in `src/models/causal_film_model.py` to accept `feature_noise`.
  - **Input Noise**: During training, Gaussian noise is added to the raw features from the frozen video and audio encoders (`F_video_raw`, `F_audio_raw`). This forces the model to be robust to input variations.
- **Training Script**:
  - Updated `src/train_causal_film.py` to pass the `feature_noise` parameter from the training config to the model factory.

### 2025-12-02 — Loss Function Refinement

#### Goal
Combine the benefits of Hard Feature Mining (Top-K) with the stability of Smooth L1 Loss, while maintaining a lower semantic constraint to avoid underfitting.

#### Changes
- **Top-K Smooth L1 Loss**:
  - Modified `src/losses.py` to use `F.smooth_l1_loss` (beta=0.1) instead of `torch.abs` (L1) when calculating errors for Top-K selection.
  - This ensures that even when focusing on the hardest features, the gradients remain stable and less sensitive to outliers than pure L1.
- **Hyperparameters**:
  - Confirmed `lambda_text` is set to **0.01** in `configs/train_config.py`.
  - This lower weight (compared to 0.1) gives the decoder more freedom to reconstruct the specific details of the input without being overly constrained by the generic "normal weld" text embedding.

### 2025-12-03 — Random Feature Masking

#### Goal
Replace Gaussian noise injection with Random Feature Masking to encourage robustness without altering feature magnitudes, aiming to prevent overfitting.

#### Changes
- **Configuration**:
  - Set `"feature_noise": 0.0` in `configs/train_config.py` (temporarily disabled).
  - Added `"feature_mask_ratio": 0.25` in `configs/train_config.py`.
- **Model Architecture**:
  - Updated `CausalFiLMModel` in `src/models/causal_film_model.py` to accept `feature_mask_ratio`.
  - **Feature Masking**: During training, 25% of the elements in `F_video_raw` and `F_audio_raw` are randomly set to zero. The remaining elements are scaled by `1 / (1 - p)` to maintain the expected magnitude.

## Notes

- I did not delete any original files. I only created this consolidated changelog and archived copies. Please confirm which originals you want removed; I list recommended deletions below for your convenience.

---

Generated: 2025-10-30
Branch: main
