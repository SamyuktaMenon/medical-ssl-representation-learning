"""Loss functions for self-supervised representation learning.

Implements Normalized Temperature-scaled Cross Entropy (NT-Xent) loss (Chen et al., 2020).
"""

from typing import Union, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss.

    Mathematical Intuition & Formulation:
    -------------------------------------
    Given a batch of N images, stochastic data augmentations yield two correlated views
    per image, forming a multi-view batch of 2N samples.

    Let z_k be the L2-normalized projection vector of augmented sample k in {0, ..., 2N-1}.
    For an anchor sample i, its corresponding view j(i) is the single positive pair:
        j(i) = i + N  if i < N
        j(i) = i - N  if i >= N

    The remaining 2N - 2 samples in the minibatch serve as negative pairs.
    Pairwise similarity is computed as scaled cosine similarity:
        s_{i, k} = (z_i^T z_k) / (||z_i|| * ||z_k|| * temperature)

    The contrastive loss for positive pair (i, j) is:
        l_{i, j} = - log ( exp(s_{i, j}) / sum_{k=0, k != i}^{2N-1} exp(s_{i, k}) )

    The total loss is the symmetric average across all 2N pairs:
        L = (1 / 2N) * sum_{i=0}^{2N-1} l_{i, j(i)}

    Numerical Stability:
    -------------------
    The self-similarity diagonal elements s_{i, i} are masked with -1e9 before
    evaluating log-softmax / cross-entropy, eliminating self-comparison while
    guaranteeing stability without numerical overflow or underflow.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the symmetric NT-Xent loss.

        Args:
            z_i: Tensor of projection embeddings from first view, shape (N, D),
                 or combined concatenated projections of shape (2N, D).
            z_j: Optional tensor of projection embeddings from second view, shape (N, D).
                 If None, z_i is assumed to already be concatenated (2N, D).

        Returns:
            Scalar tensor containing the mean contrastive loss.
        """
        if z_j is not None:
            if z_i.dim() != 2 or z_j.dim() != 2:
                raise ValueError(
                    f"Expected 2D tensors (N, D), got {z_i.shape} and {z_j.shape}"
                )
            if z_i.shape != z_j.shape:
                raise ValueError(
                    f"Shape mismatch: z_i {z_i.shape} != z_j {z_j.shape}"
                )
            batch_size = z_i.shape[0]
            if batch_size < 2:
                raise ValueError(
                    f"Batch size must be >= 2 for contrastive negatives, got {batch_size}"
                )
            # L2 normalize embeddings
            z_i = F.normalize(z_i, p=2, dim=1)
            z_j = F.normalize(z_j, p=2, dim=1)
            # Concatenate along batch dimension: shape (2N, D)
            z = torch.cat([z_i, z_j], dim=0)
        else:
            if z_i.dim() != 2:
                raise ValueError(f"Expected 2D tensor (2N, D), got {z_i.shape}")
            total_samples = z_i.shape[0]
            if total_samples % 2 != 0:
                raise ValueError(
                    f"Combined tensor must have an even number of samples, got {total_samples}"
                )
            batch_size = total_samples // 2
            if batch_size < 2:
                raise ValueError(
                    f"Batch size must be >= 2 for contrastive negatives, got {batch_size}"
                )
            z = F.normalize(z_i, p=2, dim=1)

        total_samples = 2 * batch_size

        # Pairwise cosine similarity matrix: shape (2N, 2N)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask self-similarity (diagonal entries) with large negative number
        diag_mask = torch.eye(total_samples, dtype=torch.bool, device=z.device)
        mask_val = -1e4 if sim_matrix.dtype in [torch.float16, torch.bfloat16] else -1e9
        sim_matrix = sim_matrix.masked_fill(diag_mask, mask_val)

        # Positive targets: for i in [0, N-1], target is i + N; for i in [N, 2N-1], target is i - N
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        # CrossEntropyLoss computes -log(exp(sim[i, target[i]]) / sum_{k!=i} exp(sim[i, k]))
        loss = F.cross_entropy(sim_matrix, targets)
        return loss
