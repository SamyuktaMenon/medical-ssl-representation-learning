"""Unit test suite for NT-Xent Contrastive Loss.

Mathematical validation of positive-pair indexing, self-similarity masking,
gradient flow, batch-size boundary conditions, and device portability.
"""

import unittest
import torch
import torch.nn.functional as F

from src.losses import NTXentLoss
from src.utils import get_device


class TestNTXentLoss(unittest.TestCase):
    """Test suite for Normalized Temperature-scaled Cross Entropy loss."""

    def setUp(self):
        self.device = get_device("auto")
        self.loss_fn = NTXentLoss(temperature=0.1)

    def test_output_shape_and_finite(self):
        """Test that loss output is a 0-dim scalar with finite real value."""
        N, D = 8, 128
        z1 = torch.randn(N, D, device=self.device)
        z2 = torch.randn(N, D, device=self.device)

        loss = self.loss_fn(z1, z2)

        self.assertEqual(loss.dim(), 0, "Loss must be a 0-dimensional scalar")
        self.assertTrue(torch.isfinite(loss).item(), "Loss must be finite")
        self.assertFalse(torch.isnan(loss).item(), "Loss must not be NaN")
        self.assertGreater(loss.item(), 0.0, "Loss must be strictly positive")

    def test_concatenated_input_support(self):
        """Test that single concatenated (2N, D) input matches (z1, z2) dual inputs."""
        N, D = 8, 128
        z1 = torch.randn(N, D, device=self.device)
        z2 = torch.randn(N, D, device=self.device)

        loss_dual = self.loss_fn(z1, z2)
        z_cat = torch.cat([z1, z2], dim=0)
        loss_cat = self.loss_fn(z_cat)

        torch.testing.assert_close(loss_dual, loss_cat, atol=1e-5, rtol=1e-5)

    def test_positive_pair_alignment_reduces_loss(self):
        """Verify that identical paired views yield minimal loss compared to orthogonal views."""
        N, D = 4, 64
        # Case 1: Identical positive pairs (cosine sim = 1.0) with mutually orthogonal negatives
        z1 = torch.eye(N, D, device=self.device)
        z2 = z1.clone()  # exact identical positive views
        loss_identical = self.loss_fn(z1, z2)

        # Case 2: Random vectors
        z1_rand = torch.randn(N, D, device=self.device)
        z2_rand = torch.randn(N, D, device=self.device)
        loss_random = self.loss_fn(z1_rand, z2_rand)

        self.assertLess(
            loss_identical.item(),
            loss_random.item(),
            "Loss on aligned positive pairs must be significantly lower than on random pairs",
        )

    def test_hand_calculated_analytical_example(self):
        """Verify NT-Xent loss against an exact independently hand-calculated analytical example.

        Let N=2, D=2, tau=0.5:
            z1 = [[1.0, 0.0], [0.0, 1.0]]
            z2 = [[0.8, 0.6], [-0.6, 0.8]]
        Both rows have unit L2 norm.
        Analytical cross-entropy over 4 masked similarity rows yields exactly 0.430190277.
        """
        loss_fn = NTXentLoss(temperature=0.5)
        z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device)
        z2 = torch.tensor([[0.8, 0.6], [-0.6, 0.8]], device=self.device)

        loss = loss_fn(z1, z2)

        import math
        e_pos = math.exp(0.8 / 0.5)
        e_0 = math.exp(0.0 / 0.5)
        e_neg1 = math.exp(-0.6 / 0.5)
        e_neg2 = math.exp(0.6 / 0.5)

        l0 = -math.log(e_pos / (e_0 + e_pos + e_neg1))
        l1 = -math.log(e_pos / (e_0 + e_neg2 + e_pos))
        l2 = -math.log(e_pos / (e_pos + e_neg2 + e_0))
        l3 = -math.log(e_pos / (e_neg1 + e_pos + e_0))
        expected_analytical = (l0 + l1 + l2 + l3) / 4.0

        self.assertAlmostEqual(loss.item(), expected_analytical, places=6)

    def test_self_similarity_masking(self):
        """Verify that self-similarity is masked and does not corrupt denominator or logits.

        For N=2, batch 2N=4:
        If self-similarity were not masked, s_{i,i} = 1.0/temp = 10.0 would dominate the softmax.
        We verify by manually computing the cross-entropy on masked similarity matrix.
        """
        N, D = 2, 8
        torch.manual_seed(42)
        z1 = torch.randn(N, D, device=self.device)
        z2 = torch.randn(N, D, device=self.device)
        loss = self.loss_fn(z1, z2)

        # Manual calculation
        z_norm1 = F.normalize(z1, dim=1)
        z_norm2 = F.normalize(z2, dim=1)
        z = torch.cat([z_norm1, z_norm2], dim=0)  # shape (4, D)
        sim = torch.matmul(z, z.T) / 0.1

        # Check positive pairs:
        # Row 0 target is 2; Row 1 target is 3; Row 2 target is 0; Row 3 target is 1
        targets = torch.tensor([2, 3, 0, 1], device=self.device)
        mask = torch.eye(4, dtype=torch.bool, device=self.device)
        mask_val = -1e4 if sim.dtype in [torch.float16, torch.bfloat16] else -1e9
        sim_masked = sim.masked_fill(mask, mask_val)
        expected_loss = F.cross_entropy(sim_masked, targets)

        torch.testing.assert_close(loss, expected_loss, atol=1e-5, rtol=1e-5)

    def test_gradient_flow_and_backprop(self):
        """Verify non-zero gradients propagate back to input embeddings."""
        N, D = 4, 32
        z1 = torch.randn(N, D, device=self.device, requires_grad=True)
        z2 = torch.randn(N, D, device=self.device, requires_grad=True)

        loss = self.loss_fn(z1, z2)
        loss.backward()

        self.assertIsNotNone(z1.grad)
        self.assertIsNotNone(z2.grad)
        self.assertTrue(torch.all(torch.isfinite(z1.grad)))
        self.assertTrue(torch.all(torch.isfinite(z2.grad)))
        self.assertGreater(torch.norm(z1.grad).item(), 0.0)
        self.assertGreater(torch.norm(z2.grad).item(), 0.0)

    def test_minimal_batch_size(self):
        """Test minimal valid batch size N=2 (producing 2N=4 samples, 1 positive and 2 negatives per anchor)."""
        z1 = torch.randn(2, 16, device=self.device)
        z2 = torch.randn(2, 16, device=self.device)
        loss = self.loss_fn(z1, z2)
        self.assertTrue(torch.isfinite(loss).item())

    def test_invalid_batch_size(self):
        """Verify that batch size N < 2 raises ValueError."""
        # N = 1 (no negatives available)
        z1 = torch.randn(1, 16, device=self.device)
        z2 = torch.randn(1, 16, device=self.device)
        with self.assertRaises(ValueError):
            self.loss_fn(z1, z2)

        # Odd number of samples in concatenated tensor
        z_odd = torch.randn(5, 16, device=self.device)
        with self.assertRaises(ValueError):
            self.loss_fn(z_odd)

        # Negative or zero temperature
        with self.assertRaises(ValueError):
            NTXentLoss(temperature=-0.1)
        with self.assertRaises(ValueError):
            NTXentLoss(temperature=0.0)


if __name__ == "__main__":
    unittest.main()
