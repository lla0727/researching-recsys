"""
FeSAIL: Feature Staleness Aware Incremental Learning for CTR Prediction
Based on paper: http://arxiv.org/abs/2505.02844v1

Paper highlights:
- Problem: Feature staleness in incremental CTR model training
- Solution: SAS (Staleness Aware Sampling) + SAR (Staleness Aware Regularization)
- Result: ~4-6% AUC improvement on benchmark datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import random


# ============== Part 1: Feature Staleness Tracking ==============

class FeatureStalenessTracker:
    """
    Track feature staleness across time spans.
    From paper page 3, Equation (1):
        st_i = st-1_i + 1, if fi not in FDt
              0, otherwise

    Attributes:
        staleness: dict mapping feature_id -> staleness value
    """

    def __init__(self):
        self.staleness = defaultdict(int)

    def update(self, active_features: set):
        """
        Update staleness after each time span.

        Args:
            active_features: set of features appearing in current dataset
        """
        # Increment staleness for features not in current dataset
        for feat_id in list(self.staleness.keys()):
            if feat_id not in active_features:
                self.staleness[feat_id] += 1
            else:
                # Reset staleness if feature appears
                self.staleness[feat_id] = 0

    def get_staleness(self, feature_id: int) -> int:
        """Get staleness value for a feature."""
        return self.staleness.get(feature_id, 0)

    def get_staleness_batch(self, feature_ids: list) -> list:
        """Get staleness values for a batch of features."""
        return [self.get_staleness(fid) for fid in feature_ids]


# ============== Part 2: SAS - Staleness Aware Sampling ==============

class StalenessAwareSampling:
    """
    SAS Algorithm (from paper page 3-4, Algorithm 1)

    Goal: Select L samples covering maximum weight of stale features
    Formulation: Maximum Weighted Coverage Problem

    Theorem 1: SAS achieves 1 - 1/e approximation ratio

    Inverse correlation function (paper Eq 2):
        wi = func(st_i) + b
        where func can be: 1/si or exp(-si)
    """

    def __init__(self, reservoir_size: int = 10000, func_type: str = 'inverse',
                 bias: float = 0.1):
        """
        Args:
            reservoir_size: fixed size L of the reservoir
            func_type: 'inverse' (1/si) or 'exp' (exp(-si))
            bias: bias term b in Eq (2)
        """
        self.reservoir_size = reservoir_size
        self.func_type = func_type
        self.bias = bias
        self.reservoir = []

    def _compute_weight(self, staleness: int) -> float:
        """Compute weight from staleness using Eq (2)."""
        if self.func_type == 'inverse':
            return 1.0 / (staleness + 1) + self.bias
        elif self.func_type == 'exp':
            return np.exp(-staleness) + self.bias
        else:
            raise ValueError(f"Unknown func_type: {self.func_type}")

    def _greedy_select(self, samples: list, features_per_sample: list,
                       staleness_dict: dict) -> list:
        """
        Greedy algorithm for maximum weighted coverage.
        Time complexity: O(|Rt| * L * m)

        Returns:
            selected_indices: indices of selected samples
        """
        selected_indices = []
        covered_features = set()
        n_samples = len(samples)

        # Precompute weights for all features
        feature_weights = {
            feat: self._compute_weight(staleness_dict.get(feat, 0))
            for feat in staleness_dict
        }

        for _ in range(min(self.reservoir_size, n_samples)):
            best_sample_idx = -1
            best_incremental_weight = -1

            for i, (sample_id, features) in enumerate(zip(samples, features_per_sample)):
                if i in selected_indices:
                    continue

                # Calculate incremental weight (new features covered)
                incremental_weight = 0
                for feat in features:
                    if feat not in covered_features:
                        incremental_weight += feature_weights.get(feat, 0)

                if incremental_weight > best_incremental_weight:
                    best_incremental_weight = incremental_weight
                    best_sample_idx = i

            if best_sample_idx == -1:
                break

            selected_indices.append(best_sample_idx)
            # Update covered features
            for feat in features_per_sample[best_sample_idx]:
                covered_features.add(feat)

        return selected_indices

    def select(self, candidates: list, features_per_sample: list,
               staleness_dict: dict) -> list:
        """
        Select samples from candidates using SAS algorithm.

        Args:
            candidates: list of sample IDs
            features_per_sample: list of feature lists per sample
            staleness_dict: dict mapping feature_id -> staleness

        Returns:
            selected_samples: selected samples with maximum coverage
        """
        indices = self._greedy_select(candidates, features_per_sample, staleness_dict)
        return [candidates[i] for i in indices]


# ============== Part 3: SAR - Staleness Aware Regularization ==============

class StalenessAwareRegularization(nn.Module):
    """
    SAR Module (from paper Section 3.1)

    Purpose: Fine-grained control of feature embedding updates based on staleness

    The regularization restricts embedding updates for features with high staleness
    to prevent overfitting on stale features.
    """

    def __init__(self, embed_dim: int = 64, func_type: str = 'inverse',
                 bias: float = 0.1):
        """
        Args:
            embed_dim: embedding dimension
            func_type: 'inverse' or 'exp'
            bias: bias term
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.func_type = func_type
        self.bias = bias

    def _compute_reg_weight(self, staleness: int) -> float:
        """
        Compute regularization weight from staleness.
        Higher staleness -> lower update weight (to prevent overfitting)
        """
        if staleness == 0:
            return 1.0  # No regularization for active features

        if self.func_type == 'inverse':
            return 1.0 / (staleness + 1) + self.bias
        elif self.func_type == 'exp':
            return np.exp(-staleness) + self.bias
        return 1.0

    def compute_reg_loss(self, embeddings: torch.Tensor,
                        staleness_values: list) -> torch.Tensor:
        """
        Compute staleness-aware regularization loss.

        Args:
            embeddings: (batch_size, num_fields, embed_dim)
            staleness_values: list of staleness for each field position

        Returns:
            reg_loss: scalar regularization loss
        """
        reg_weights = torch.tensor(
            [self._compute_reg_weight(s) for s in staleness_values],
            device=embeddings.device,
            dtype=embeddings.dtype
        )

        # L2 regularization scaled by staleness weight
        # Higher staleness -> stronger regularization -> smaller update
        reg_loss = (reg_weights.unsqueeze(0).unsqueeze(-1) * embeddings ** 2).mean()
        return reg_loss


# ============== Part 4: Base CTR Model ==============

class BaseCTRModel(nn.Module):
    """
    Base CTR prediction model with embedding layer and interaction layers.
    Can be DeepFM, DCN, or similar architecture.
    """

    def __init__(self, num_features: int, embed_dim: int = 64,
                 hidden_dims: list = [256, 128, 64]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features

        # Embedding layer
        self.embedding = nn.Embedding(num_features, embed_dim)

        # Interaction layers (can be DeepFM, DCN, etc.)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, feature_ids: torch.Tensor):
        """
        Args:
            feature_ids: (batch_size, num_fields)

        Returns:
            logits: (batch_size,)
        """
        # Embedding lookup
        emb = self.embedding(feature_ids)  # (batch, num_fields, embed_dim)

        # Sum pooling for simplicity
        pooled = emb.sum(dim=1)  # (batch, embed_dim)

        # MLP
        output = self.fc(pooled).squeeze(-1)
        return output


# ============== Part 5: FeSAIL Full Framework ==============

class FeSAIL(nn.Module):
    """
    Full FeSAIL Framework integrating:
    1. Staleness tracking
    2. SAS for sample selection
    3. SAR for regularization
    4. Base CTR model
    """

    def __init__(self, num_features: int, embed_dim: int = 64,
                 hidden_dims: list = [256, 128, 64],
                 reservoir_size: int = 10000,
                 func_type: str = 'inverse', bias: float = 0.1):
        super().__init__()
        self.staleness_tracker = FeatureStalenessTracker()
        self.sas = StalenessAwareSampling(reservoir_size, func_type, bias)
        self.sar = StalenessAwareRegularization(embed_dim, func_type, bias)
        self.ctr_model = BaseCTRModel(num_features, embed_dim, hidden_dims)

    def update_staleness(self, active_features: set):
        """Update feature staleness after each time span."""
        self.staleness_tracker.update(active_features)

    def forward(self, feature_ids: torch.Tensor,
                staleness_positions: list = None):
        """
        Forward pass with optional SAR regularization.

        Args:
            feature_ids: (batch_size, num_fields)
            staleness_positions: list of field positions for regularization

        Returns:
            logits: (batch_size,)
        """
        embeddings = self.ctr_model.embedding(feature_ids)
        logits = self.ctr_model.fc(embeddings.sum(dim=1)).squeeze(-1)
        return logits

    def get_reg_loss(self, feature_ids: torch.Tensor) -> torch.Tensor:
        """Compute SAR regularization loss."""
        embeddings = self.ctr_model.embedding.weight  # (num_features, embed_dim)

        # For simplicity, use field-level staleness
        staleness_vals = [
            self.staleness_tracker.get_staleness(i)
            for i in range(feature_ids.shape[1])
        ]

        return self.sar.compute_reg_loss(embeddings, staleness_vals)


# ============== Part 6: Incremental Training Loop ==============

class IncrementalTrainer:
    """
    Incremental training loop for FeSAIL.
    From paper: joint training with reservoir Rt and current data Dt
    """

    def __init__(self, model: FeSAIL, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.reservoir = []

    def sample_reservoir(self, historical_samples: list,
                        features_per_sample: list) -> list:
        """
        Use SAS to sample from historical data.

        Args:
            historical_samples: list of historical sample IDs
            features_per_sample: list of feature lists

        Returns:
            selected_samples: selected samples using SAS
        """
        staleness_dict = dict(self.model.staleness_tracker.staleness)
        return self.model.sas.select(
            historical_samples,
            features_per_sample,
            staleness_dict
        )

    def train_step(self, current_batch, reservoir_batch=None):
        """
        Single training step.

        Args:
            current_batch: current time span data
            reservoir_batch: sampled reservoir data (optional)

        Returns:
            loss: total loss including SAR regularization
        """
        self.model.train()

        # Forward pass
        features, labels = current_batch
        features = features.to(self.device)
        labels = labels.to(self.device).float()

        logits = self.model(features)

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # SAR regularization
        reg_loss = self.model.get_reg_loss(features)

        # Total loss
        total_loss = bce_loss + 0.01 * reg_loss  # alpha=0.01

        return total_loss, bce_loss, reg_loss


# ============== Part 7: Demo Usage ==============

def demo():
    """Demo: Train FeSAIL on synthetic incremental data."""

    # Hyperparameters from paper
    NUM_FEATURES = 100000
    EMBED_DIM = 64
    HIDDEN_DIMS = [256, 128, 64]
    RESERVOIR_SIZE = 10000

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FeSAIL(
        num_features=NUM_FEATURES,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        reservoir_size=RESERVOIR_SIZE
    ).to(device)

    # Initialize trainer
    trainer = IncrementalTrainer(model, device)

    print("=" * 60)
    print("FeSAIL: Feature Staleness Aware Incremental Learning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Embedding dim: {EMBED_DIM}")
    print(f"Reservoir size: {RESERVOIR_SIZE}")
    print()

    # Simulate incremental training
    for time_span in range(1, 4):
        print(f"--- Time Span {time_span} ---")

        # Simulate active features in current dataset
        active_features = set(random.sample(range(NUM_FEATURES), 5000))

        # Update staleness
        model.update_staleness(active_features)

        # Sample reservoir using SAS
        historical = list(range(time_span * 1000))
        features_hist = [random.sample(range(NUM_FEATURES), 20) for _ in historical]
        selected = trainer.sample_reservoir(historical, features_hist)
        print(f"Selected {len(selected)} samples from reservoir")

        # Simulate training
        batch_size = 256
        seq_len = 20
        fake_features = torch.randint(0, NUM_FEATURES, (batch_size, seq_len))
        fake_labels = torch.randint(0, 2, (batch_size,)).float()

        loss, bce, reg = trainer.train_step((fake_features, fake_labels))
        print(f"Loss: {loss.item():.4f} (BCE: {bce.item():.4f}, Reg: {reg.item():.4f})")
        print()

    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
