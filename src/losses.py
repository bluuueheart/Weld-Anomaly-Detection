"""
Loss functions for quad-modal welding anomaly detection.

Implements:
- Supervised Contrastive Loss (SupConLoss) for learning discriminative features
- Causal-FiLM Loss (reconstruction + CLIP text constraint) for anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Reference:
    Khosla et al. "Supervised Contrastive Learning"
    NeurIPS 2020. https://arxiv.org/abs/2004.11362
    
    This loss pulls together samples from the same class and pushes apart
    samples from different classes in the feature space.
    
    Args:
        temperature (float): Temperature parameter for scaling (default: 0.07)
        contrast_mode (str): 'all' or 'one' (default: 'all')
        base_temperature (float): Base temperature (default: 0.07)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature vectors of shape (batch_size, feature_dim)
            labels: Ground truth labels of shape (batch_size,)
            
        Returns:
            loss: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        # anchor_dot_contrast: (batch_size, batch_size)
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # If no positive pairs exist for an anchor, skip it
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Combines Supervised Contrastive Loss with optional Cross Entropy Loss.
    
    Args:
        use_ce (bool): Whether to use cross entropy loss
        ce_weight (float): Weight for cross entropy loss
        supcon_weight (float): Weight for supervised contrastive loss
        temperature (float): Temperature for SupConLoss
        num_classes (int): Number of classes (required if use_ce=True)
    """
    
    def __init__(
        self,
        use_ce: bool = False,
        ce_weight: float = 0.0,
        supcon_weight: float = 1.0,
        temperature: float = 0.07,
        num_classes: int = None,
    ):
        super().__init__()
        self.use_ce = use_ce
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        
        self.supcon_loss = SupConLoss(temperature=temperature)
        
        if use_ce:
            assert num_classes is not None, "num_classes required for CE loss"
            self.ce_loss = nn.CrossEntropyLoss()
        else:
            self.ce_loss = None
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            features: Feature vectors (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
            logits: Class logits (batch_size, num_classes), required if use_ce=True
            
        Returns:
            loss_dict: Dictionary containing:
                - 'total': Total loss
                - 'supcon': Supervised contrastive loss
                - 'ce': Cross entropy loss (if applicable)
        """
        # Supervised contrastive loss
        loss_supcon = self.supcon_loss(features, labels)
        
        # Total loss
        total_loss = self.supcon_weight * loss_supcon
        
        loss_dict = {
            'total': total_loss,
            'supcon': loss_supcon.item(),
        }
        
        # Cross entropy loss
        if self.use_ce:
            assert logits is not None, "logits required for CE loss"
            loss_ce = self.ce_loss(logits, labels)
            total_loss = total_loss + self.ce_weight * loss_ce
            loss_dict['total'] = total_loss
            loss_dict['ce'] = loss_ce.item()
        
        return loss_dict


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss using cosine distance.
    
    Used in Causal-FiLM to measure reconstruction error between
    Z_result (ground truth) and Z_result_pred (reconstructed).
    
    Formula: L_recon = 1 - cosine_similarity(Z_result, Z_result_pred)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        Z_result: torch.Tensor,
        Z_result_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            Z_result: Ground truth result encoding (B, d_model)
            Z_result_pred: Reconstructed result encoding (B, d_model)
            
        Returns:
            loss: Scalar reconstruction loss
        """
        # Cosine distance = 1 - cosine_similarity
        cos_sim = F.cosine_similarity(Z_result, Z_result_pred, dim=1)
        loss = 1.0 - cos_sim.mean()
        return loss


class CLIPTextLoss(nn.Module):
    """
    CLIP-based text constraint loss.
    
    Forces reconstructed features to be semantically close to "a normal weld"
    in CLIP's joint vision-language space. This is a class-agnostic global
    constraint that prevents the decoder from reconstructing anomalies.
    
    Args:
        clip_model_name: CLIP model name (default: "ViT-B/32")
        text_prompt: Text prompt for normal class (default: "a normal weld")
        device: Device to load CLIP model
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        text_prompt: str = "a normal weld",
        device: str = "cuda",
    ):
        super().__init__()
        
        self.text_prompt = text_prompt
        self.device = device
        
        # Load CLIP model (frozen)
        try:
            import clip
            self.clip_model, _ = clip.load(clip_model_name, device=device)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # Encode text prompt (done once)
            with torch.no_grad():
                text_tokens = clip.tokenize([text_prompt]).to(device)
                self.text_features = self.clip_model.encode_text(text_tokens)
                self.text_features = F.normalize(self.text_features, p=2, dim=-1)
            
            self.clip_dim = self.text_features.shape[-1]  # 512 for ViT-B/32
            self.clip_available = True
            
        except ImportError:
            print("Warning: CLIP not available. CLIPTextLoss will return zero.")
            self.clip_available = False
            self.clip_dim = 512
        
        # Projection head: d_model -> CLIP dimension
        # This is the only trainable component
        self.vision_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.clip_dim),
        ).to(device)  # Move to same device as CLIP model
    
    def forward(self, Z_result_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP text constraint loss.
        
        Args:
            Z_result_pred: Reconstructed result encoding (B, d_model)
            
        Returns:
            loss: Scalar CLIP text loss
        """
        if not self.clip_available:
            return torch.tensor(0.0, device=Z_result_pred.device)
        
        # Project to CLIP space
        pred_clip_features = self.vision_projection(Z_result_pred)  # (B, clip_dim)
        pred_clip_features = F.normalize(pred_clip_features, p=2, dim=-1)
        
        # Compute cosine distance to text features
        # text_features: (1, clip_dim) -> broadcast to (B, clip_dim)
        cos_sim = F.cosine_similarity(
            pred_clip_features,
            self.text_features.expand(pred_clip_features.size(0), -1),
            dim=1
        )
        
        # Loss = 1 - similarity (we want high similarity)
        loss = 1.0 - cos_sim.mean()
        return loss


class CausalFILMLoss(nn.Module):
    """
    Combined loss for Causal-FiLM model.
    
    L_total = L_recon + λ * L_text
    
    Where:
        - L_recon: Reconstruction loss (cosine distance)
        - L_text: CLIP text constraint loss
        - λ: Weighting parameter (default: 0.1)
    
    Args:
        lambda_text: Weight for CLIP text loss (default: 0.1)
        clip_model_name: CLIP model name (default: "ViT-B/32")
        text_prompt: Text prompt for normal class (default: "a normal weld")
        device: Device for CLIP model
    """
    
    def __init__(
        self,
        lambda_text: float = 0.1,
        clip_model_name: str = "ViT-B/32",
        text_prompt: str = "a normal weld",
        device: str = "cuda",
    ):
        super().__init__()
        
        self.lambda_text = lambda_text
        
        # Reconstruction loss
        self.recon_loss = ReconstructionLoss()
        
        # CLIP text constraint loss
        self.clip_text_loss = CLIPTextLoss(
            clip_model_name=clip_model_name,
            text_prompt=text_prompt,
            device=device,
        )
    
    def forward(
        self,
        Z_result: torch.Tensor,
        Z_result_pred: torch.Tensor,
    ) -> dict:
        """
        Compute combined Causal-FiLM loss.
        
        Args:
            Z_result: Ground truth result encoding (B, d_model)
            Z_result_pred: Reconstructed result encoding (B, d_model)
            
        Returns:
            loss_dict: Dictionary containing:
                - 'total': Total loss
                - 'recon_cos': Reconstruction loss (cosine)
                - 'recon_mse': Reconstruction loss (MSE)
                - 'clip_text': CLIP text constraint loss
        """
        # 1. Cosine Loss (Direction)
        loss_cos = 1.0 - F.cosine_similarity(Z_result, Z_result_pred, dim=-1).mean()
        
        # 2. L1 Loss (Mean Absolute Error)
        # We want the model to match the INTENSITY of the features
        loss_l1 = F.l1_loss(Z_result, Z_result_pred)
        
        # CLIP text loss
        loss_clip_text = self.clip_text_loss(Z_result_pred)
        
        # Total loss
        # L_total = loss_cos + 10.0 * loss_l1 + lambda_clip * L_clip
        total_loss = loss_cos + 10.0 * loss_l1 + self.lambda_text * loss_clip_text
        
        loss_dict = {
            'total': total_loss,
            'recon_cos': loss_cos.item(),
            'recon_l1': loss_l1.item(),
            'clip_text': loss_clip_text.item(),
        }
        
        return loss_dict


def test_supcon_loss():
    """Test SupConLoss implementation."""
    print("Testing SupConLoss...")
    
    # Create dummy data
    batch_size = 8
    feature_dim = 512
    num_classes = 6
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test SupConLoss
    criterion = SupConLoss(temperature=0.07)
    loss = criterion(features, labels)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print(f"  ✅ Backward pass successful")
    
    # Test with all same labels
    same_labels = torch.zeros(batch_size, dtype=torch.long)
    loss_same = criterion(features, same_labels)
    print(f"  Loss (all same class): {loss_same.item():.4f}")
    
    # Test with all different labels
    diff_labels = torch.arange(batch_size)
    loss_diff = criterion(features, diff_labels)
    print(f"  Loss (all different): {loss_diff.item():.4f}")
    
    print("  ✅ SupConLoss test passed!\n")


def test_combined_loss():
    """Test CombinedLoss implementation."""
    print("Testing CombinedLoss...")
    
    batch_size = 8
    feature_dim = 512
    num_classes = 6
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    logits = torch.randn(batch_size, num_classes)
    
    # Test SupCon only
    criterion = CombinedLoss(use_ce=False, supcon_weight=1.0)
    loss_dict = criterion(features, labels)
    
    print(f"  SupCon only:")
    print(f"    Total loss: {loss_dict['total'].item():.4f}")
    print(f"    SupCon loss: {loss_dict['supcon']:.4f}")
    
    # Test SupCon + CE
    criterion = CombinedLoss(
        use_ce=True,
        ce_weight=0.5,
        supcon_weight=1.0,
        num_classes=num_classes,
    )
    loss_dict = criterion(features, labels, logits)
    
    print(f"\n  SupCon + CE:")
    print(f"    Total loss: {loss_dict['total'].item():.4f}")
    print(f"    SupCon loss: {loss_dict['supcon']:.4f}")
    print(f"    CE loss: {loss_dict['ce']:.4f}")
    
    # Test backward
    loss_dict['total'].backward()
    print(f"\n  ✅ CombinedLoss test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LOSS FUNCTIONS TEST")
    print("=" * 70 + "\n")
    
    test_supcon_loss()
    test_combined_loss()
    
    print("=" * 70)
    print("✅ ALL LOSS TESTS PASSED!")
    print("=" * 70 + "\n")
