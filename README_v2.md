# 论文叙事 (The Story)
标题 (Title):
Causal-FiLM: 用于统一多模态工业异常检测的因果分层融合与传感器调制
(Causal-FiLM: Causal-Hierarchical Fusion with Sensor-Modulation for Unified Multimodal Industrial Anomaly Detection)
摘要 (Abstract / The Pitch):
无监督异常检测 (UAD) 是工业质量控制（如焊接）的关键，因为缺陷形态多样且样本稀缺。然而，现有的SOTA方法存在两个主要鸿沟：
1. MUAD (多类别UAD) 鸿沟：为解决多类别（如不同工件）的统一检测，Dinomaly 等模型专注于解决单模态（图像）的“过度泛化”问题 (2)。但这在缺陷仅存在于其他模态（如音频 (3) 或传感器）中可见时会失效。
2. 多模态融合鸿沟：现有的多模态方法（如 M3DM (4), FmFormer (5), Stemmer et al. (6)）通常采用“扁平” (Flat) 或“后期” (Late) 融合 (7)。这忽视了工业数据中“过程”（Process）模态与“结果”（Result）模态之间固有的因果关系，并错误地将低维、高噪声的传感器数据与高维视觉数据直接混合，导致性能下降。
为解决这些问题，我们提出 Causal-FiLM，一个仅使用“正常”样本训练的UAD框架。我们的贡献是双重的：
1. 架构创新 (因果分层)：我们显式地将模态分为“过程”（实时视频/音频/传感器）和“结果”（焊后图像）。我们适配了 Costanzino et al. 的跨模态重建思想 (9)，通过学习 $f: \text{Process} \to \text{Result}$ 的正向因果映射来进行重建。
2. 模块创新 (FiLM)：我们不“融合”传感器数据，而是将其视为上下文 (Context)。我们引入特征引导的线性调制 (FiLM)，利用传感器数据动态地调整（Modulate）高维视频和音频特征的提取。
Causal-FiLM 只学习“正常”的因果映射。在推理时，任何破坏了这种正常因果链的异常（例如，异常的过程导致了异常的结果）都会产生巨大的重建误差，从而被检测。在我们的新型四模态焊缝基准上，Causal-FiLM 的性能显著优于所有适配的SOTA基线，同时实现了SOTA级的效率（无Memory Bank） (10)。
核心任务定义 (Task Definition)
- 任务：统一多类别无监督异常检测 (MUAD)。
- 定义：训练一个单一、统一 (Unified) 的模型 ，使其能够处理数据集中来自所有不同类别/工件的“正常”样本。
- 数据假设：训练集仅包含“正常” (Good / Normal-condition) 样本 。测试集包含“正常”和“异常” (Defective) 样本。
- 目标：区分“正常”与“异常”，并泛化到所有未见过的缺陷类型。
# Causal-FiLM 模型技术方案 (L0-L4)
核心策略： 您的数据集（约4000个样本）规模不足以训练大型骨干网络。因此，我们必须采用**“冻结Backbone + 轻量级可训练融合头”**的SOTA策略，这与 M3DM (9)和 Costanzino et al. (10) 的做法一致。

---
L0: 冻结的特征提取器 (Frozen Backbones)
  (固定参数，仅用于特征提取)
  实时视频 → V-JEPA → F_video (时空特征)
  实时音频 → AST → F_audio (时序特征)
  焊后图像 → DINOv2 → F_image (N个视角的特征)
  传感器 → (无Backbone) → Data_sensor (原始时序数据)
L1: 传感器引导调制 (FiLM)
  (可训练，轻量级)
  理念：传感器数据是“上下文”，而不是“数据”。
  网络：SensorModulator (一个2层GRU，hidden=64)。
  Data_sensor (e.g., $128 \times 6$) $\to$ SensorModulator $\to$ Linear(D_model*2) $\to$ gamma, beta (调制向量, 维度 D_model=128)。
  应用：FiLM调制 F_video 和 F_audio。
  F_video_mod = F_video * gamma + beta
  F_audio_mod = F_audio * gamma + beta
L2: 因果分层融合 (Causal Encoders)
  (可训练，轻量级)
  ProcessEncoder (一个2层交叉注意力模块，仿 FmFormer (11))：
  Z_process_tokens = CrossAttn(Q=F_v_mod, K/V=F_a_mod)。
  Z_process = MeanPool(Z_process_tokens) $\to$ (128-dim)。
  ResultEncoder (一个MLP)：
  F_image $\to$ MeanPool $\to$ Linear(128) $\to$ Z_result (128-dim)。
L3: “反泛化”重建解码器 (Anti-Generalization Decoder)
  (可训练，轻量级)
  理念：必须防止L3变得“太聪明”而重建异常。
  网络：一个轻量级Transformer解码器 (例如，NumLayers=2, NumHeads=4)。
  Noisy Bottleneck (仿 Dinomaly ())：
  Z_process_noisy = Dropout(Z_process, p=0.2)。
  优势：强迫解码器学习“去噪/恢复”正常模式，而不是简单的“恒等映射”。
  Unfocused Attention (仿 Dinomaly ()()()())：
  解码器内部的注意力层必须使用 Linear Attention (而非Softmax)。
  优势：Linear Attention 天生“无法聚焦”，防止解码器“抄近道” (Shortcut) ()，强迫其从全局 Z_process_noisy 中重建 Z_result。
  输出：Z_result_pred = Decoder(Z_process_noisy)。
L4: 复合损失与推理 (Hybrid Loss & Inference)
  (可训练，轻量级)
  理念：引入语义约束来辅助几何重建。
  损失 (Loss)：
    1. L_recon = CosineDistance(Z_result, Z_result_pred) (主损失)。
    2. L_text (仿 CNC )：L_text = CosineDistance(CLIP_Text_Feat("a normal weld"), CLIP_Vision_Head(Z_result_pred))。
    优势：一个类无关的 (class-agnostic) () 全局约束，强迫所有重建结果在语义上必须是“正常”的，进一步抑制对异常的重建。
    3. L_total = L_recon + \lambda \cdot L_text。
  推理分数 (Inference Score)：
    Anomaly_Score = CosineDistance(Z_result, L3_Decoder(Z_process))。
# 详细实验方案 (主实验 + 3个副实验)
核心数据集： 所有实验均在您的“四模态焊缝数据集”(Weld-4M)上进行。
指标：AUPRO@30% AUPRO@1%，检测I-AUROC、分割P-AUPRO
超参数忠于baseline原设置，除了必要适配均使用原始代码。先复现有源码的，没有源码的先放着。到时候具体的适配细节在附录里面开一章写
理想表格示意：2个这样的表，展示不同类别（检测I-AUROC、分割P-AUPRO）
模态 (Modality)方法 (Method)I-AUROC (↑)P-AUPRO (↑)(A) 过程模态 (Process-Only) (A.1) V+A88.589(A.2) V+A8281.5(A.N) 过程融合 (Ours-Process)Causal-FiLM (V+A)92.593------------(B) 结果模态 (Result-Only)(B.1) 图像-OnlyDinomaly (Image) 9192.5(A.N) 结果模态 (Ours-Process)Causal-FiLM (Image)92.593---------(C) 跨模态融合 (Cross-Modal)(C.1) 简单融合Late Fusion (A+B) 1092.893.2(C.2) 简单融合Concat-AE (A+B)93.193.5(C.3) SOTA基线Costanzino-style 119494.2(C.4) SOTA基线M3DM-style (No-Bank) 1294.294.5(C.N) 我们的模型Causal-FiLM (Ours)95.595.8
-----

# 详细的工程化模型方案 (L0-L4)

这里是基于你的设计，提供一个更接近“工程实现”的伪代码和结构细化。

[cite\_start]我们假设所有模态的统一特征维度 `D_model = 128` [cite: 33, 41, 42]。

#### L0: 冻结的特征提取器

这一步的目标是获取特征序列，并**统一维度**到 `D_model=128`。

```python
# [cite_start]--- L0: 冻结的 Backbones [cite: 22] ---
# (所有 backbone 参数均 .requires_grad = False)

# [cite_start]1. 实时视频 [cite: 24]
v_jepa = load_frozen_vjepa() # (e.g., ViT-B/16)
video_data # (B, T_video, C, H, W)
F_video_raw = v_jepa.extract_features(video_data) # (B, T_v, D_jepa=768)
# 线性投影层 (可训练)
video_projector = nn.Linear(768, 128) 
F_video = video_projector(F_video_raw) # (B, T_v, 128)

# [cite_start]2. 实时音频 [cite: 25]
ast = load_frozen_ast() # (e.g., AST-base)
audio_data # (B, T_audio_clips, Mel_bins, Time_frames)
F_audio_raw = ast.extract_features(audio_data) # (B, T_a, D_ast=768)
# 线性投影层 (可训练)
audio_projector = nn.Linear(768, 128)
F_audio = audio_projector(F_audio_raw) # (B, T_a, 128)

# [cite_start]3. 焊后图像 [cite: 26]
dinov2 = load_frozen_dinov2() # (e.g., ViT-B/14)
image_data # (B, N_views, C, H, W)
B, N, C, H, W = image_data.shape
image_data_flat = image_data.view(B * N, C, H, W)
# 提取 [CLS] token
F_image_raw = dinov2.extract_cls_token(image_data_flat) # (B * N, D_dino=768)
F_image_raw = F_image_raw.view(B, N, 768) # (B, N_views, 768)
# 线性投影层 (可训练)
image_projector = nn.Linear(768, 128)
F_image = image_projector(F_image_raw) # (B, N_views, 128)

# [cite_start]4. 传感器 (原始数据) [cite: 27]
[cite_start]Data_sensor # (B, T_sensor=128, D_sensor=6) [cite: 32]
```

#### L1: 传感器引导调制 (FiLM)

这是你的**创新点 1**。

```python
# [cite_start]--- L1: 传感器引导调制 (FiLM) [cite: 28] ---

class SensorModulator(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, d_model=128):
        super().__init__()
        # [cite_start]2层 GRU [cite: 31]
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # [cite_start]线性层[cite: 33], 输出 2 * D_model 用于 gamma 和 beta
        self.modulator_head = nn.Linear(hidden_dim, d_model * 2) 

    def forward(self, sensor_data):
        # sensor_data shape: (B, T_sensor=128, D_sensor=6)
        # 我们只关心最后的隐状态，它总结了整个时序
        _, last_hidden_state = self.gru(sensor_data) 
        # last_hidden_state shape: (num_layers, B, hidden_dim)
        # 取最后一层的隐状态
        context_vector = last_hidden_state[-1] # (B, hidden_dim=64)
        
        # 映射到 gamma 和 beta
        modulators = self.modulator_head(context_vector) # (B, d_model * 2 = 256)
        
        # [cite_start]分割 [cite: 33]
        gamma, beta = torch.chunk(modulators, 2, dim=1) # (B, 128), (B, 128)
        
        # 返回 (B, 1, 128) 以便广播
        return gamma.unsqueeze(1), beta.unsqueeze(1)

# --- 应用 FiLM ---
sensor_modulator = SensorModulator()
gamma, beta = sensor_modulator(Data_sensor) # (B, 1, 128), (B, 1, 128)

# [cite_start]FiLM 应用 (仿射变换) [cite: 34-37]
F_video_mod = F_video * (gamma + 1.0) + beta # (B, T_v, 128) (+1.0 是为了让 gamma 初始在 1 附近)
F_audio_mod = F_audio * (gamma + 1.0) + beta # (B, T_a, 128)
```

#### L2: 因果分层融合 (Encoders)

这是你的**创新点 2**。

```python
# [cite_start]--- L2: 因果分层融合 [cite: 38] ---

# [cite_start]1. Process Encoder [cite: 40]
# 方案: 仿 FmFormer，使用交叉注意力
# [cite_start](注意: FmFormer [cite: 80] [cite_start]是双向的, 你的定义 [cite: 40] 是单向 Q=V, K/V=A)
# 这是一个轻量级的 Transformer 交叉注意力层
process_cross_attn = nn.TransformerDecoderLayer(
    d_model=128, nhead=4, dim_feedforward=256, batch_first=True
)
# F_v_mod (B, T_v, 128) 作为 target (Q)
# F_a_mod (B, T_a, 128) 作为 memory (K, V)
Z_process_tokens = process_cross_attn(F_video_mod, F_audio_mod) # (B, T_v, 128)

# [cite_start]平均池化得到最终 Z_process [cite: 41]
Z_process = Z_process_tokens.mean(dim=1) # (B, 128)

# [cite_start]2. Result Encoder [cite: 41]
# [cite_start]方案: MLP (MLP 比单个 Linear [cite: 42] 更稳健)
# (B, N_views, 128) -> (B, 128)
pooled_image_feat = F_image.mean(dim=1) 
result_encoder_mlp = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128)
)
Z_result = result_encoder_mlp(pooled_image_feat) # (B, 128)
```

#### L3: "反泛化" 重建解码器

这是实现 UAD 的关键。

```python
# [cite_start]--- L3: 反泛化解码器 [cite: 43] ---

# [cite_start]关键: 实现 "Linear Attention" [cite: 51]
# 这是一个简化的实现，用 QK^T 的特征图近似
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        [cite_start]self.elu_fn = nn.ELU(alpha=1.0) # 仿 "Unfocused Attention" [cite: 50]

    def forward(self, x):
        # x: (B, SeqLen, Dim) e.g., (B, 1, 128)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1) # (q, k, v)
        q, k, v = map(lambda t: t.view(B, N, self.heads, -1).transpose(1, 2), qkv) # (B, H, N, D_h)

        # 核心: 先激活 Q 和 K
        q = self.elu_fn(q) + 1.0 
        k = self.elu_fn(k) + 1.0
        
        # 计算 KV_sum (N, D_h, D_h) -> (B, H, D_h, D_h)
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v) 
        # 计算 Z = 1 / QK_sum
        z = 1.0 / (torch.einsum('b h n d, b h n d -> b h n', q, k.sum(dim=2, keepdim=True)).sum(dim=2) + 1e-6)
        
        # 计算 (QK^T)V 
        out = torch.einsum('b h n d, b h d e, b h n -> b h n e', q, kv, z)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)

# [cite_start]仿 Dinomaly 的轻量级 Transformer 块 [cite: 46]
class AntiGenBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        [cite_start]self.attn = LinearAttention(d_model, nhead) # 使用 Linear Attention [cite: 51]
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CausalDecoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2, nhead=4, dropout_p=0.2):
        super().__init__()
        [cite_start]self.dropout = nn.Dropout(dropout_p) # Noisy Bottleneck [cite: 47, 48]
        self.layers = nn.ModuleList([AntiGenBlock(d_model, nhead) for _ in range(num_layers)])

    def forward(self, Z_process, is_training=True):
        # Z_process: (B, 128)
        
        # [cite_start]1. Noisy Bottleneck [cite: 47]
        Z_process_noisy = self.dropout(Z_process) if is_training else Z_process
        
        # 2. Transformer 需要序列输入 (B, SeqLen=1, 128)
        x = Z_process_noisy.unsqueeze(1)
        
        # 3. 通过反泛化块
        for layer in self.layers:
            x = layer(x)
            
        # 4. 变回 (B, 128)
        Z_result_pred = x.squeeze(1)
        return Z_result_pred

# --- 实例化 ---
[cite_start]L3_Decoder = CausalDecoder(d_model=128, num_layers=2, nhead=4, dropout_p=0.2) [cite: 46, 48]
```

#### L4: 复合损失与推理

```python
# [cite_start]--- L4: 复合损失 [cite: 55] ---

# [cite_start]1. L_recon (主损失) [cite: 59]
# (使用 Z_result.detach() 可以稳定训练，但让 Z_result 可训练也是一种选择)
# (我们先假设 Z_result 不被L_recon梯度更新，只被L_text更新)
# (更正: 你的方案中 Z_result 和 Z_process 应该是共同优化的，所以不 .detach())
Z_result_pred = L3_Decoder(Z_process, is_training=True)
L_recon = 1.0 - F.cosine_similarity(Z_result, Z_result_pred, dim=-1).mean()

# [cite_start]2. L_text (CNC 约束) [cite: 60]
# 加载 CLIP 文本编码器 (冻结)
clip_text_model = load_frozen_clip_text_encoder()
[cite_start]text_prompt = "a normal weld" [cite: 60]
with torch.no_grad():
    text_feat = clip_text_model.encode_text(text_prompt) # (1, D_clip=512)

# 可训练的投影头，将 D_model=128 映射到 D_clip=512
clip_vision_head = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512)
)

# 投影 Z_result_pred 到 CLIP 空间
pred_clip_feat = clip_vision_head(Z_result_pred)
pred_clip_feat = F.normalize(pred_clip_feat, p=2, dim=-1) # CLIP 需要 L2 归一化
text_feat = F.normalize(text_feat, p=2, dim=-1)

L_text = 1.0 - F.cosine_similarity(pred_clip_feat, text_feat, dim=-1).mean()

# [cite_start]3. 总损失 [cite: 62]
lambda_text = 0.1 # 超参数
L_total = L_recon + lambda_text * L_text

# [cite_start]--- 推理 [cite: 63] ---
def get_anomaly_score(Z_process_test, Z_result_test):
    # 推理时关闭 Dropout
    Z_result_pred_test = L3_Decoder(Z_process_test, is_training=False) 
    
    [cite_start]# [cite: 64]
    score = 1.0 - F.cosine_similarity(Z_result_test, Z_result_pred_test, dim=-1)
    return score
```