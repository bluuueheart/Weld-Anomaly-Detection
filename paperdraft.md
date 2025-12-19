 # Causal-FiLM for Unified Multimodal Welding Anomaly Detection (Paper Draft)
 
 This draft is generated **strictly from the current codebase** (`.py`). Any detail not verifiable in code is marked as `To be confirmed`.
 
 ## Methodology
 
 ### 1. Data, Inputs, and Preprocessing (Code-Exact)
 Each sample is a 4-modal tuple returned by `src/dataset.py::WeldingDataset`:
 - **Video** `video`: `(T_v, 3, H_v, W_v)` from `_read_video`.
 - **Post-weld images** `post_weld_images`: `(N, 3, H_i, W_i)` from `_read_post_weld_images`.
 - **Audio** `audio`: `(1, F, T_a)` from `_read_audio`.
 - **Sensor** `sensor`: `(T_s, C_s)` from `_read_sensor`.
 
 In `src/train_causal_film.py`, the dataset is constructed with
 `video_length=32`, `image_size=224`, `num_angles=5`, `audio_duration=256`, `sensor_length=256`.
 From `WeldingDataset` defaults, the effective tensor sizes are:
 - `T_v=32`, `H_v=W_v=64` (default `frame_size=64`).
 - `N=5`, `H_i=W_i=224`.
 - `T_a=256`, `F=64` (default `audio_mel_bins=64`).
 - `T_s=256`, `C_s=6`.

 Note: `configs/dataset_config.py` defines `VIDEO_FRAME_SIZE=224` and `AUDIO_N_MELS=128`, but `train_causal_film.py` and `evaluate_causal_film.py` do not pass these into `WeldingDataset`; thus the current pipeline uses the `WeldingDataset` defaults (`frame_size=64`, `audio_mel_bins=64`) unless set explicitly.
 
 Preprocessing details (all in `src/dataset.py`):
 - **Video**: BGR→RGB, resize to `(frame_size, frame_size)`, scale to `[0,1]`, uniform sampling/padding to `num_frames`.
 - **Images**: BGR→RGB, resize to `(image_size, image_size)`, scale to `[0,1]`, pad/truncate to `num_angles`.
 - **Audio (mel)**: `librosa.load(sr=audio_sr)` → `melspectrogram(n_mels=audio_mel_bins)` → `power_to_db` → crop/pad to `audio_frames`.
 - **Sensor**: read numeric columns from `.csv` → per-channel interpolation to `sensor_len` → pad/truncate to `sensor_channels` → per-channel z-score.
 
 ### 2. Model Overview (Causal-FiLM)
 The main model is `src/models/causal_film_model.py::CausalFiLMModel`.
 Let unified dimension be `d=512` (`configs/model_config.py::CAUSAL_FILM_CONFIG`).
 
 #### 2.1 L0: Frozen Backbones
 Under `torch.no_grad()` the model extracts:
 - Video features from `VideoEncoder` (local V-JEPA) → `F_v^{raw}`.
 - Audio features from `AudioEncoder` (local AST) → `F_a^{raw}`.
 - Image outputs from `ImageEncoder` (local DINOv3 with `output_hidden_states=True`) → a dict-like output containing `hidden_states`.
 
 Backbones expose token sequences through the HuggingFace `transformers.AutoModel` API (e.g., `last_hidden_state`, `hidden_states`). Therefore, backbone token/sequence lengths are properties of the external pretrained checkpoints and are **not fixed by this repository** (`To be confirmed` without the checkpoint `config.json` / runtime print).
 
 The downstream encoders only assume a generic token axis:
 - Process branch: `ProcessEncoder` consumes `(B, T_v, d)` and `(B, T_a, d)` and then mean-pools over `T_v`.
 - Result branch: `RobustResultEncoder` pools DINOv3 hidden states over their token dimension (`mean(dim=1)` and GeM pooling), and does not explicitly drop a CLS token.
 
 #### 2.2 Projection to Unified Dimension
 In `CausalFiLMModel.__init__`:
 - Video projector: `Linear(1024→512) → LayerNorm → GELU → Linear(512→d)`.
 - Audio projector: `Linear(768→512) → LayerNorm → GELU → Linear(512→d)`.
 
 #### 2.3 Process-feature Regularization
 In `CausalFiLMModel.forward` (training only):
 - Gaussian noise with std `feature_noise`.
 - Random masking with ratio `feature_mask_ratio=r`, rescaled by `1/(1-r)`.

 Note: In `src/train_causal_film.py::Trainer._setup_model`, `feature_noise` and `feature_mask_ratio` are taken from `configs/train_config.py::TRAIN_CONFIG` and injected into `configs/model_config.py::CAUSAL_FILM_CONFIG` before calling `create_causal_film_model`.
 
 ### 3. L1: Sensor-Guided FiLM Modulation (Mamba)
 Implemented in `src/models/film_modulation.py::SensorModulator`.
 
 Given sensor $X^s\in\mathbb{R}^{B\times T_s\times 6}$:
 - Embedding: `Linear(6→h_s)` with `h_s=sensor_hidden_dim=128`.
 - Temporal encoder: `num_layers=2` stacked `MambaBlock` (each uses pre-norm + residual; `d_state=16, d_conv=4, expand=2`).
 - Last-token pooling: take the last time step.
 - Head: `Linear(h_s→2d)` then split to $(\gamma',\beta)\in\mathbb{R}^{B\times d}$ and set $\gamma=\gamma'+1$; finally reshape to $\gamma,\beta\in\mathbb{R}^{B\times 1\times d}$ for broadcasting.
 - FiLM: for process features $F$ (video/audio),
   $\mathrm{FiLM}(F;\gamma,\beta)=F\odot\gamma+\beta$ (broadcast over tokens).
 
 ### 4. L2: Causal Encoders
 #### 4.1 Process Encoder (Cross-Attention)
 `src/models/causal_encoders.py::ProcessEncoder` is implemented via `nn.TransformerDecoder`:
 - `tgt =` FiLM-modulated video tokens, `memory =` FiLM-modulated audio tokens.
 - `num_layers=2`, `nhead=4`, `dropout=0.1`, `dim_feedforward=2d`.
 - Outputs are mean-pooled over tokens and LayerNorm’ed to obtain $Z_{proc}\in\mathbb{R}^{B\times d}$.
 
 No explicit attention mask is passed in code.
 
 #### 4.2 Result Encoder (RobustResultEncoder)
 `src/models/causal_encoders.py::RobustResultEncoder` consumes DINOv3 `hidden_states` and extracts:
 - Layer 21 (index 20): mean pooling and **GeM pooling** (learnable `p`, initialized `p=6.0`).
 - Layer 32 (index 31): mean pooling.
 
 Each pooled vector is independently LayerNorm’ed, concatenated (dim `1280*3=3840`), and passed through:
 - Gating network: `Linear(3840→960) → SiLU → Linear(960→3840) → Sigmoid`, then elementwise gating.
 - Projector: `Linear(3840→512) → LayerNorm → SiLU → Linear(512→d)`.
 
 Multi-angle aggregation in `CausalFiLMModel.forward`:
 - The result encoder runs on flattened `B·N` images, reshaped to `(B,N,d)`, then **max-pooled over angles** to get $Z_{res}\in\mathbb{R}^{B\times d}$.
 
 ### 5. L3: Anti-Generalization Decoder
 The current instantiated decoder in `CausalFiLMModel` is `src/models/causal_decoder.py::AntiGenDecoder`:
 `Linear(d→d) → LayerNorm → SiLU → Dropout(0.1) → Linear(d→d)`.
 It predicts $Z_{pred}\in\mathbb{R}^{B\times d}$ from $Z_{proc}$.

 Note: `configs/model_config.py::CAUSAL_FILM_CONFIG` contains `decoder_num_layers=2` and `decoder_dropout=0.2`, but in non-dummy mode `CausalFiLMModel` instantiates `AntiGenDecoder(d_model=d)` and does not use these fields; the effective dropout in this decoder is the hard-coded `0.1`.
 
 ### 6. Training Objective (CausalFILMLoss)
 Implemented in `src/losses.py::CausalFILMLoss`:
 \[
 \mathcal{L}=\mathcal{L}_{cos}+\lambda_{l1}\,\mathcal{L}_{l1}+\lambda_{gram}\,\mathcal{L}_{gram}+\lambda_{text}\,\mathcal{L}_{clip}.
 \]
 - $\mathcal{L}_{cos}=1-\cos(Z_{res},Z_{pred})$.
 - $\mathcal{L}_{l1}$ uses Smooth L1 (`beta=0.1`), optionally with top-$k$ mining when `top_k_ratio<1`.
 - $\mathcal{L}_{gram}$ uses Gram matrices $G(z)=zz^\top/d$ with MSE.
 - $\mathcal{L}_{clip}$ is a CLIP text constraint toward the prompt `"a normal weld"` (returns 0 if the `clip` package is unavailable).

 CLIP projection head (trainable part inside `src/losses.py::CLIPTextLoss`):
 `Linear(d→256) → LayerNorm → GELU → Dropout(0.1) → Linear(256→512)`, followed by L2 normalization and cosine distance to the frozen text embedding.
 
 Default weights from `configs/train_config.py`:
 - `lambda_text=0.01`, `lambda_l1=1.0`, `lambda_gram=1.0`, `top_k_ratio=0.5`.
 
 ### 7. Anomaly Score
 Implemented in `CausalFiLMModel.compute_anomaly_score`:
 \[
 s = (1-\cos(Z_{res},Z_{pred})) + 10\cdot \mathrm{mean}(\mathrm{TopK}(|Z_{res}-Z_{pred}|, k)),\quad k=\lfloor d\cdot \rho\rfloor.
 \]
 where $\rho=\texttt{top\_k\_ratio}$.

 Note: In code, the TopK term in the anomaly score uses elementwise absolute error (`abs`) and the ratio comes from `CausalFiLMModel.top_k_ratio` (default `0.5` unless passed via `CAUSAL_FILM_CONFIG`). In contrast, the L1 term in `CausalFILMLoss` uses Smooth L1 (`beta=0.1`) and its `top_k_ratio` comes from `TRAIN_CONFIG`. With current configs both are `0.5`.

### 8. Training Protocol (Code-Exact; Causal-FiLM)

#### 8.1 Authoritative entrypoint (bash)
Training is executed via `scripts/train_causal_film.sh`. With no extra arguments, it sets `PYTHONPATH` and runs:

```bash
python src/train_causal_film.py \
  --config configs/train_config.py \
  --wandb \
  --wandb_project "weld-anomaly-detection" \
  --wandb_name "causal_film_run_$(date +%Y%m%d_%H%M%S)"
```

If any arguments are provided to the script, it forwards them directly to `python src/train_causal_film.py "$@"`.

#### 8.2 Config loading and overrides
In `src/train_causal_film.py::main`:
- Base config is `configs/train_config.py::TRAIN_CONFIG` (shallow `copy()`).
- If `--config` is provided:
  - When the suffix is `.py`, the file is imported dynamically and must define a `TRAIN_CONFIG` dict, which is merged into the base config.
  - Otherwise the file is treated as JSON and merged into the base config.
- CLI flags then override WandB-related fields (`--wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_name`, `--wandb_id`).

#### 8.3 Data split usage and “normal-only” training
In `src/train_causal_film.py::CausalFiLMTrainer._setup_data`:
- Training dataset uses `WeldingDataset(..., split='train', augment=use_augmentations)`.
- Validation dataset uses `WeldingDataset(..., split='val', augment=False)`.
- `WeldingDataset` filters the CSV manifest (`configs/manifest.csv`) by `split.upper()` (e.g., `'train'→'TRAIN'`, `'val'→'VAL'`, `'test'→'TEST'`).

After loading the `split='train'` dataset, the trainer further constructs a `torch.utils.data.Subset` containing only “normal” samples by the following heuristic:
- If `train_dataset._labels[idx] == 0`, keep the sample.
- Else if the sample id/path string contains `'good'` or `'normal'` (case-insensitive), keep the sample.

Validation uses all samples in the `VAL` split; AUROC uses binary labels computed as `(label != 0)`.

#### 8.4 Optimization, schedule, and stabilization
In `src/train_causal_film.py::CausalFiLMTrainer._setup_optimizer`:
- **Optimizer**: `torch.optim.AdamW(model.parameters(), lr, weight_decay, betas=(0.9,0.999), eps=1e-8)`.
- **LR scheduler**: `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=num_epochs, eta_min=min_lr)`.

Important code-truth discrepancy:
- `configs/train_config.py` contains warmup-related keys (`lr_scheduler`, `warmup_epochs`, `warmup_start_lr`) and EMA keys (`use_ema`, `ema_decay`), but `src/train_causal_film.py` does **not** implement warmup nor EMA (no references to these keys).

Gradient clipping and AMP are implemented in `train_epoch`:
- If `gradient_clip > 0`, gradients are clipped by `torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)`.
- If `mixed_precision=True`, training uses `torch.cuda.amp.autocast()` + `torch.cuda.amp.GradScaler()` and performs clipping after `scaler.unscale_(optimizer)`.

#### 8.5 Train-time regularization injected into the model
In `CausalFiLMTrainer._setup_model`, before model creation:
- `CAUSAL_FILM_CONFIG` is copied.
- `feature_noise` and `feature_mask_ratio` from training config are injected into this copy.

During `CausalFiLMModel.forward` (training mode only), this results in:
- Additive Gaussian feature noise on `F_video_raw` and `F_audio_raw` with std `feature_noise`.
- Elementwise Bernoulli masking with probability `feature_mask_ratio=r`, rescaled by `1/(1-r)`.

#### 8.6 Data augmentation (training split only)
When `use_augmentations=True`, `WeldingDataset` applies best-effort augmentations for the training split (`split=='train'`). Parameters are read from `configs/dataset_config.py::AUGMENTATION` when available.

Implemented augmentation operators in `src/dataset.py`:
- **Images** (`_augment_images_numpy`): random resized crop, rotation, H/V flip, brightness/contrast jitter, saturation/hue jitter (HSV), Gaussian blur, Gaussian noise, and cutout.
- **Video** (`_augment_video_numpy`): per-clip random resized crop, rotation, H/V flip, optional temporal shift via `np.roll`, plus per-frame brightness/contrast jitter (hard-coded range `±0.15`).
- **Audio** (`_augment_audio_numpy`): additive Gaussian noise, SpecAugment-style time masking and frequency masking.
- **Sensor** (`_augment_sensor_numpy`): additive Gaussian noise, per-channel scaling, and single-channel dropout.

#### 8.7 Checkpointing, early stopping, and outputs
- **Monitored metric**: `val/auroc` (computed in `validate`).
- **Checkpointing**: only the best checkpoint is saved (when `val/auroc` improves) to `CHECKPOINT_DIR/best_model.pth`.
  - Note: `TRAIN_CONFIG` includes `save_interval`, but `src/train_causal_film.py::_save_checkpoint` returns early unless `is_best=True`.
- **Early stopping**: if `early_stopping_patience>0`, training stops when `epochs_without_improvement >= early_stopping_patience`.
- **Resume**: `--resume best|latest` loads `CHECKPOINT_DIR/best_model.pth`.
- **Logs**: written to `LOG_DIR/training_log.json`.

By default, output directories are defined in `configs/train_config.py`:
- `OUTPUT_DIR="/root/autodl-tmp/outputs"`
- `CHECKPOINT_DIR="/root/autodl-tmp/outputs/checkpoints"`
- `LOG_DIR="/root/autodl-tmp/outputs/logs"`

### 9. Evaluation Protocol (Code-Exact; Causal-FiLM)

#### 9.1 Authoritative entrypoint (bash)
Evaluation is executed via `scripts/evaluate_causal_film.sh <checkpoint_path>`, which runs:

```bash
python src/evaluate_causal_film.py --checkpoint "<checkpoint_path>" --split test --device cuda
```

#### 9.2 Dataset, anomaly scores, and anomaly maps
In `src/evaluate_causal_film.py::CausalFiLMEvaluator.evaluate`:
- The evaluation dataset is `WeldingDataset(..., split=<split>, augment=False)`.
- The dataloader is created with `batch_size=32`, `shuffle=False`, `num_workers=4`, `pin_memory=True`.

For each batch, evaluation computes:
- `output = model(batch)` (without `return_encodings`).
- Image-level anomaly scores using `model.compute_anomaly_score(...)` (same scoring function as in training).

Pixel-level anomaly maps are currently a placeholder in `extract_anomaly_scores_and_maps`:
- For each sample, `anomaly_map = ones(224,224) * score` (uniform map with the global score).
- The script explicitly notes this is not a true per-pixel reconstruction error map.

#### 9.3 Metrics (as implemented)
In `compute_metrics_with_pro`:
- **I-AUROC**: `roc_auc_score(labels, scores)` with binary labels `(label != 0)`.
- **AP**: `average_precision_score(labels, scores)`.
- **Optimal threshold**: chosen on ROC curve by maximizing Youden’s $J=\mathrm{TPR}-\mathrm{FPR}$.
- **Precision/Recall/F1**: computed at the chosen threshold.

P-AUPRO is computed only if `scipy` is available:
- Since no ground-truth pixel masks exist in the dataset interface, `compute_pro_metric` approximates P-AUPRO by treating the whole image as a single region (all-ones GT mask for anomalous images; all-zeros for normal images) and thresholding the placeholder anomaly maps.
- Reported metrics: `P-AUPRO@{0.3, 0.1, 0.05, 0.01}`.
- If `scipy` is unavailable, P-AUPRO is skipped (returns `0.0`) with a warning.

Results are saved to `OUTPUT_DIR/eval_results.json` by default (unless `--output` is provided).

## Experiments Setup

This section documents the experimental setup **strictly from code** (`.py`, `.sh`). For any detail not verifiable from code, we mark it as `To be confirmed`.

### 10.1 Compared methods (code entrypoints)

- **Causal-FiLM (ours)**
  - Train: `bash scripts/train_causal_film.sh` → `src/train_causal_film.py`
  - Eval: `bash scripts/evaluate_causal_film.sh <ckpt>` → `src/evaluate_causal_film.py`
- **Late Fusion baseline** (audio AE + video AE, late score fusion)
  - Train: `bash baselines/Late_Fusion/train.sh` → `baselines/Late_Fusion/train.py`
  - Eval: `bash baselines/Late_Fusion/evaluate.sh` → `baselines/Late_Fusion/evaluate.py`
- **M3DM baseline** (PatchCore-style memory bank + one-class fusion)
  - Train: `bash baselines/M3DM/train.sh` → `baselines/M3DM/weld_main.py` → `baselines/M3DM/weld_m3dm_runner.py`
  - Eval: `bash baselines/M3DM/evaluate.sh` → `baselines/M3DM/weld_main.py --eval_only`
- **Dinomaly baseline** (image-only, reconstruction-style transformer)
  - Train/Eval: `bash baselines/Dinomaly/train_weld.sh` → `baselines/Dinomaly/dinomaly_weld.py`

### 10.2 Dataset and splits (authoritative: `configs/manifest.csv`)

All methods rely on `configs/manifest.csv` (CSV columns include at least `CATEGORY`, `SUBDIRS`, `SPLIT`).

From a code-based parse of the manifest:
- **Total**: 4040 rows
- **Split sizes**:
  - `TRAIN`: 576 (Good-only)
  - `VAL`: 1732 (122 Good + 1610 Defective)
  - `TEST`: 1732 (121 Good + 1611 Defective)

In `src/dataset.py::WeldingDataset`, each sample returns both:
- `label`: binary label (Good=0, Defective=1 via `0 if class_label==0 else 1`)
- `class_label`: the original 12-way class id (`0..11`)

### 10.3 Metrics reported (code-truth)

#### 10.3.1 Unified image-level metrics
Across this repository, the primary image-level detection metric is **I-AUROC**.

Additional image-level metrics are implemented and reported by some scripts:
- **AP** and **F1-max** are reported by Causal-FiLM, Late Fusion, and M3DM; Dinomaly reports them in `--eval_only` mode (via `evaluate_weld(..., return_extra=True)`).

Per-defect-type AUROC is also computed by all methods, but via different implementations:
- Causal-FiLM: `src/evaluate_causal_film.py::compute_per_defect_auroc`
- Late Fusion: `baselines/Late_Fusion/utils.py::compute_auc_per_class`
- M3DM: `baselines/M3DM/feature_extractors/features.py::_calculate_per_category_metrics`
- Dinomaly: `baselines/Dinomaly/dinomaly_weld.py::evaluate_weld` (string parsing from `img_path`)

#### 10.3.2 Pixel-level metrics are only proxies in this repository
There is **no pixel-accurate ground-truth mask** in the welding dataset interface.

- Causal-FiLM: anomaly maps in `extract_anomaly_scores_and_maps` are placeholders (uniform map filled with the global score), so P-AUPRO is only a proxy.
- M3DM: `baselines/M3DM/weld_dataset.py` synthesizes a mask as:
  - Good → all zeros
  - Defect → all ones
  Therefore pixel-level ROCAUC / AU-PRO computed by M3DM are also proxies.

### 10.4 Baseline details (strictly from `baselines/` code)

#### 10.4.1 Late Fusion (audio AE + video AE)

**Dataset usage** (`baselines/Late_Fusion/train.py`, `baselines/Late_Fusion/evaluate.py`):
- Uses `src/dataset.py::WeldingDataset` in real mode.
- Train split: `split="train"`.
- Validation during training: `split="test"` (note: does **not** use the manifest `VAL` split).
- Evaluation: `split="test"`.

**Audio preprocessing and input**:
- Uses `WeldingDataset(..., audio_type="stft", audio_sr=192000, n_fft=16384, hop_length=8192, audio_frames=1024)`.
- In `src/dataset.py::_read_audio`, STFT mode returns `spec` with shape `(1, n_bins, audio_frames)` where `n_bins = n_fft // 2 + 1 = 8193`.

**Audio model** (`baselines/Late_Fusion/models.py::AudioAutoEncoder`):
- 1D CNN AE with LeakyReLU(0.2) activations; bottleneck channel dimension `48`.
- Frame-wise anomaly score is MSE over frequency bins → shape `(batch, time_steps)`.

**Video preprocessing and input**:
- Late Fusion does not override `num_frames` / `frame_size` when constructing `WeldingDataset`.
- Therefore `WeldingDataset` defaults are used (see `src/dataset.py`):
  - `num_frames=8`
  - `frame_size=64`

**Video model** (`baselines/Late_Fusion/models.py::VideoAutoEncoder`):
- Two-stage implementation.
- In current training scripts, `slowfast_model=None`, so Stage-1 falls back to:
  - global average pool each frame over `(H,W)` → `(batch, 3)`
  - a frozen linear projection `Linear(3→2304, bias=False)` → per-frame features `(batch, 2304)`
- Stage-2 is a fully-connected AE with ReLU activations and dropout `0.5` at specified layers.
- Frame-wise anomaly score is MSE over feature dim → `(batch, num_frames)`.

**Aggregation + fusion** (`baselines/Late_Fusion/utils.py`, `baselines/Late_Fusion/evaluate.py`):
- Audio aggregation: expected value (`mean` over time).
- Video aggregation: max over 2s moving average.
  - With default `num_frames=8`, this reduces to `mean(frame_scores)` because `len(frame_scores) < 60`.
- Standardize audio/video sample scores using training-set mean/std.
- Fuse with fixed weights from `baselines/Late_Fusion/config.py`:
  - `w_audio=0.37`, `w_video=0.63`.
  - Note: although `optimize_fusion_weights` exists, `evaluate.py` does **not** call it.

**Training schedules** (`baselines/Late_Fusion/train.py`):
- Audio AE: Adam + MSE + OneCycleLR (`epochs=50`, `pct_start=0.1`, cosine anneal).
- Video AE: Adam + MSE, no scheduler, early stopping with `patience=10`.

#### 10.4.2 M3DM (DINO ViT + Point-MAE + memory bank)

**Weld adaptation dataset** (`baselines/M3DM/weld_dataset.py`):
- RGB: loads the first post-weld image found under `<sample>/images/*.png`, resized to `224×224`, normalized by ImageNet mean/std.
- Sensor: converts the per-sample CSV sensor table into a pseudo “organized point cloud” tensor of shape `(3, 224, 224)`:
  - read numeric columns → normalize to `[0,1]` (global min/max)
  - mean over channels → 1D signal
  - interpolate to length `224`
  - tile to a `(224,224)` depth map
  - build 3 channels `(x_coord, y_coord, depth)`

**Backbones** (`baselines/M3DM/models/models.py`):
- RGB backbone: timm ViT-Base patch8 at 224.
  - If the name contains `_dino`, the code attempts to load DINO weights from `baselines/M3DM/checkpoints/dino_vitbase8_pretrain.pth`.
- “XYZ” backbone: PointTransformer (Point-MAE-style), with defaults:
  - `trans_dim=384`, `depth=12`, `num_heads=6`
  - groups: `group_size=128`, `num_group=1024`
  - feature taps at transformer blocks `[3, 7, 11]`, concatenated → 1152 channels.

**Training = memory bank building + decision-layer fitting** (`baselines/M3DM/weld_m3dm_runner.py`):
- No end-to-end gradient descent on welding labels.
- Builds patch libraries from Good training samples up to `max_sample`.
- Applies coreset selection (`f_coreset=0.1`, `coreset_eps=0.9`) in `feature_extractors/features.py`.
- If `memory_bank == "multiple"`, also fits `SGDOneClassSVM` fusers (`detect_fuser`, `seg_fuser`).

**Evaluation + metrics** (`baselines/M3DM/feature_extractors/features.py`):
- Computes image ROCAUC and pixel ROCAUC and AU-PRO (proxy; see §10.3.2).
- Also computes AP and F1-max.
- Tracks per-defect-type I-AUROC in a fixed order (Good vs each defect).

Note on scripts:
- `baselines/M3DM/train.sh` runs `weld_main.py` with `--method_name DINO+Point_MAE` (mapped in `weld_m3dm_runner.py` to `DoubleRGBPointFeatures`).
- `baselines/M3DM/evaluate.sh` runs `weld_main.py --eval_only` with `--method_name DINO+Point_MAE+Fusion` (mapped to `TripleFeatures`).

`WeldM3DM.evaluate()` loads decision-layer fusers from `/root/autodl-tmp/save/<class>_<method_name>_decision_model.pkl`, so **training and evaluation must use the same `method_name`** for checkpoints to load. In current scripts, `train.sh` and `evaluate.sh` use different `method_name` values; which one is used for the reported baseline table is `To be confirmed`.

#### 10.4.3 Dinomaly (image-only transformer reconstruction)

**Dataset definition** (`baselines/Dinomaly/weld_dataset.py`):
- Training: `WeldImageFolder` loads Good-only samples from manifest `TRAIN`, and adds **every** image under `<sample>/images/*` as an independent training sample.
- Testing: `WeldDataset` loads images from manifest `TEST` (Good + Defective), also as per-image samples.
- No pixel masks exist; the dataset returns a dummy all-zero mask.

**Transforms / input size** (`baselines/Dinomaly/dataset.py::get_data_transforms` + `train_weld.sh`):
- `Resize((448,448)) → ToTensor() → CenterCrop(392) → Normalize(ImageNet mean/std)`.
- Effective input to the model is therefore `392×392`.

**Model + training** (`baselines/Dinomaly/dinomaly_weld.py` + `train_weld.sh` defaults):
- Encoder: `vit_encoder.load("dinov2reg_vit_base_14")`.
  - Code infers `embed_dim=768`, `num_heads=12` for “base”.
- Trainable modules:
  - bottleneck: `bMlp(embed_dim → 4*embed_dim → embed_dim, drop=0.2)`
  - decoder: 8 transformer blocks (`LinearAttention2`, `LayerNorm(eps=1e-8)`)
  - optimizer receives `trainable = ModuleList([bottleneck, decoder])` only.
- Loss: progressive hardness `global_cosine_hm_percent(en, de, p, factor=0.1)` with `p` ramped up to `0.9`.
- Optimizer: `StableAdamW(lr=2e-3, weight_decay=1e-4, amsgrad=True, eps=1e-10)`.
- Scheduler: `WarmCosineScheduler(warmup_iters=100, total_iters=epochs*len(train_loader))`.
- Gradient clipping: `clip_grad_norm_(..., max_norm=0.1)`.
- Model selection: saves `best_model.pth` by monitoring **test split** AUROC.

**Scoring** (`baselines/Dinomaly/dinomaly_weld.py::evaluate_weld`):
- For each selected feature level: `a_map = 1 - cosine_similarity(en[i], de[i])`, then bilinear upsample to `resize_mask=256`.
- Final anomaly map = mean over levels; image-level score = `max` over pixels.

### 10.5 Fairness / alignment audit (what is aligned vs not aligned)

This subsection describes alignment **as actually implemented**.

#### 10.5.1 Split usage and model selection
- Causal-FiLM: trains on `TRAIN`, selects by AUROC on `VAL`, reports on `TEST`.
- Late Fusion: trains on `TRAIN` and uses `TEST` as validation during training; reports on `TEST`.
- M3DM: builds memory banks on `TRAIN`, evaluates on `TEST` (no `VAL` in its runner).
- Dinomaly: trains on `TRAIN`, evaluates during training on `TEST` and saves best by test AUROC.

Therefore, baselines generally use the **test split for model selection**, while Causal-FiLM uses a separate `VAL` split.

#### 10.5.2 Modalities and input resolution

The compared methods are not modality-matched (by design), and they are not strictly resolution-matched in current code:

- Causal-FiLM:
  - video: `num_frames=32`, `frame_size=64` (default)
  - images: `num_angles=5`, `image_size=224`
  - audio: mel with `audio_frames=256`, `audio_mel_bins=64` (default)
  - sensor: `sensor_len=256`, `sensor_channels=6`
- Late Fusion:
  - video: default `num_frames=8`, `frame_size=64`
  - audio: STFT with `n_fft=16384` → `n_bins=8193`, `audio_frames=1024`
- M3DM:
  - RGB: `224×224`
  - sensor: pseudo organized point cloud `(3,224,224)`
- Dinomaly:
  - image-only: `CenterCrop(392)`

#### 10.5.3 Data augmentation
- Causal-FiLM enables `use_augmentations=True` in `configs/train_config.py` by default and passes `augment=...` into `WeldingDataset`.
- Late Fusion constructs `WeldingDataset` without `augment=True`.
- M3DM and Dinomaly use deterministic resize/normalize pipelines (no explicit stochastic augmentation in their welding dataset adapters).

Augmentation is therefore **not aligned** across methods in current code. Whether to disable Causal-FiLM augmentation for strict baseline fairness is `To be confirmed`.

### 10.6 Notes from `README.md` “改进记录” (trusted empirical logbook)

The repository `README.md` section `## 改进记录` is treated as a **trusted empirical logbook**, but it may not match the current implementation.

Key takeaways recorded there:
- Strong regularization (e.g., `lr=2e-5`, `wd≈1e-2`) was critical in the DINOv3 stage.
- Historical winners include: independent LayerNorm + L1-style loss (“Plan E”), Mamba-based sensor modeling, view max pooling, and SiLU.

Code mismatch note:
- `configs/train_config.py` contains `"use_ema": True`, and the logbook discusses EMA, but `src/train_causal_film.py` has no EMA implementation.

## TODO
- [x] Methodology: code-exact input tensor shapes and preprocessing.
- [x] Methodology: code-exact Causal-FiLM architecture (modules, hyperparameters, tensor shapes).
- [x] Methodology: loss and anomaly score formulas.
- [x] Methodology: training/evaluation protocol aligned with `scripts/train_causal_film.sh` and `scripts/evaluate_causal_film.sh`.
- [x] Experiments Setup: baselines (`baselines/`) and fairness/alignment analysis.
- [ ] Confirm backbone token/sequence lengths by inspecting the pretrained checkpoint `config.json` (not versioned in this repo) or printing `last_hidden_state.shape` at runtime (`To be confirmed`).
