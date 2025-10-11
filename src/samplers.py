"""
Custom samplers for balanced batch sampling.
Ensures each batch contains multiple classes for contrastive learning.
"""
import random
from typing import Iterator, List
from collections import defaultdict

try:
    from torch.utils.data import Sampler
except ImportError:
    Sampler = object


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch contains samples from multiple classes.
    
    For SupCon loss to work effectively, we need diverse labels in each batch.
    This sampler shuffles the dataset but tries to mix classes within batches.
    
    Args:
        labels: List of labels for all samples
        batch_size: Target batch size
        num_classes_per_batch: Minimum number of different classes per batch (if possible)
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        num_classes_per_batch: int = 2,
        drop_last: bool = False,
    ):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes_per_batch = min(num_classes_per_batch, batch_size)
        self.drop_last = drop_last
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        self.num_samples = len(labels)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size > 0:
            self.num_batches += 1
    
    def __iter__(self) -> Iterator[int]:
        # Shuffle indices within each class
        shuffled_indices = {}
        for label, indices in self.label_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            shuffled_indices[label] = shuffled
        
        # Create batches with mixed classes
        all_labels = list(shuffled_indices.keys())
        batches = []
        
        while any(len(v) > 0 for v in shuffled_indices.values()):
            batch = []
            
            # Shuffle class order for this batch
            random.shuffle(all_labels)
            
            # Try to get samples from different classes
            for label in all_labels:
                if len(batch) >= self.batch_size:
                    break
                if len(shuffled_indices[label]) > 0:
                    # Add samples from this class
                    samples_to_add = min(
                        self.batch_size - len(batch),
                        len(shuffled_indices[label])
                    )
                    for _ in range(samples_to_add):
                        if shuffled_indices[label]:
                            batch.append(shuffled_indices[label].pop(0))
                        if len(batch) >= self.batch_size:
                            break
            
            if len(batch) > 0:
                # Shuffle batch internally
                random.shuffle(batch)
                
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Shuffle batch order
        random.shuffle(batches)
        
        # Yield indices
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self) -> int:
        return self.num_samples


class StratifiedBatchSampler(Sampler):
    """
    Simpler stratified sampler that creates balanced batches.
    Each batch will have approximately equal representation from each class.
    
    Args:
        labels: List of labels for all samples
        batch_size: Target batch size
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(self, labels: List[int], batch_size: int, drop_last: bool = False):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        self.num_samples = len(labels)
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        shuffled_indices = {}
        for label, indices in self.label_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            shuffled_indices[label] = shuffled
        
        # Create batches with emphasis on forming positive pairs per class
        batches = []
        all_labels = list(shuffled_indices.keys())
        
        while any(len(v) > 0 for v in shuffled_indices.values()):
            batch = []
            random.shuffle(all_labels)

            # First pass: try to allocate pairs per class (SupCon friendly)
            labels_with_pairs = [
                label for label in all_labels if len(shuffled_indices[label]) >= 2
            ]
            random.shuffle(labels_with_pairs)

            for label in labels_with_pairs:
                if len(batch) >= self.batch_size:
                    break
                needed = self.batch_size - len(batch)
                take = min(2, len(shuffled_indices[label]), needed)
                for _ in range(take):
                    batch.append(shuffled_indices[label].pop(0))
                if len(batch) >= self.batch_size:
                    break

            # Second pass: fill any remaining slots with leftover samples
            if len(batch) < self.batch_size:
                for label in all_labels:
                    if len(batch) >= self.batch_size:
                        break
                    while shuffled_indices[label] and len(batch) < self.batch_size:
                        batch.append(shuffled_indices[label].pop(0))

            if not batch:
                break

            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                random.shuffle(batch)
                batches.append(batch)
        
        # Shuffle batch order
        random.shuffle(batches)
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size
