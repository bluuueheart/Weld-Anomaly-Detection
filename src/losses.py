"""
Loss functions for quad-modal welding anomaly detection.

Implements Supervised Contrastive Loss (SupConLoss) for learning
discriminative feature representations.
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
