"""Unit test suite for Phase 2: Datasets, Augmentations, Models, and Metrics."""

import os
import unittest
import numpy as np
import torch

from src.utils import set_seed, get_device
from src.datasets import PathMNISTDataset, UnlabeledPathMNISTDataset, get_stratified_train_subset
from src.augmentations import SimCLRAugmentation, get_supervised_transforms
from src.models import (
    ResNet18Encoder,
    ProjectionHead,
    SimCLRModel,
    SupervisedResNet18,
    LinearProbeClassifier,
    FineTuneClassifier,
)
from src.metrics import compute_metrics

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)

    def test_dataset_loading_and_shapes(self):
        """Test dataset loading and tensor conversion for train, val, and test splits."""
        for split in ["train", "val", "test"]:
            ds = PathMNISTDataset(split=split, root_dir=DATA_DIR)
            self.assertGreater(len(ds), 0)
            img, label = ds[0]
            self.assertEqual(img.shape, (3, 28, 28))
            self.assertIsInstance(label, int)
            self.assertTrue(0 <= label <= 8)

    def test_stratified_subsetting_reproducibility(self):
        """Verify stratified subsetting guarantees zero data leakage and strict reproducibility."""
        train_ds = PathMNISTDataset(split="train", root_dir=DATA_DIR)
        total_train = len(train_ds)
        self.assertEqual(total_train, 89996)

        # 1% subset
        sub_1pct, counts_1pct, idx_1pct = get_stratified_train_subset(train_ds, fraction=0.01, seed=42)
        expected_1pct = int(np.floor(total_train * 0.01))
        self.assertEqual(len(sub_1pct), expected_1pct)
        self.assertEqual(len(counts_1pct), 9, "All 9 classes must be represented in 1% stratified subset")
        # Indices must strictly come from train split
        self.assertTrue((idx_1pct >= 0).all() and (idx_1pct < total_train).all())

        # Parity check: identical seed produces identical indices
        _, _, idx_repeat = get_stratified_train_subset(train_ds, fraction=0.01, seed=42)
        np.testing.assert_array_equal(idx_1pct, idx_repeat)

        # Different seed produces different subset
        _, _, idx_diff = get_stratified_train_subset(train_ds, fraction=0.01, seed=123)
        self.assertFalse(np.array_equal(idx_1pct, idx_diff))

        # 5% subset
        sub_5pct, counts_5pct, _ = get_stratified_train_subset(train_ds, fraction=0.05, seed=42)
        expected_5pct = int(np.floor(total_train * 0.05))
        self.assertEqual(len(sub_5pct), expected_5pct)

        # 10% subset
        sub_10pct, counts_10pct, _ = get_stratified_train_subset(train_ds, fraction=0.10, seed=42)
        expected_10pct = int(np.floor(total_train * 0.10))
        self.assertEqual(len(sub_10pct), expected_10pct)

    def test_augmentations(self):
        """Test SimCLR dual-crop and supervised augmentations."""
        simclr_aug = SimCLRAugmentation(image_size=28)
        unlabeled_ds = UnlabeledPathMNISTDataset(root_dir=DATA_DIR, transform=simclr_aug)
        view1, view2 = unlabeled_ds[0]

        self.assertEqual(view1.shape, (3, 28, 28))
        self.assertEqual(view2.shape, (3, 28, 28))
        # Two views of the same image must be non-identical due to stochastic augmentations
        self.assertFalse(torch.allclose(view1, view2))

        # Supervised transforms
        train_tf = get_supervised_transforms(image_size=28, is_train=True)
        eval_tf = get_supervised_transforms(image_size=28, is_train=False)
        img, _ = PathMNISTDataset(split="train", root_dir=DATA_DIR, transform=train_tf)[0]
        self.assertEqual(img.shape, (3, 28, 28))
        img_eval, _ = PathMNISTDataset(split="val", root_dir=DATA_DIR, transform=eval_tf)[0]
        self.assertEqual(img_eval.shape, (3, 28, 28))

    def test_model_architectures_and_forward_shapes(self):
        """Test forward passes and output shapes for all model backbones and heads."""
        dummy_input = torch.randn(4, 3, 28, 28)

        # 1. ResNet18Encoder
        encoder = ResNet18Encoder(pretrained=False)
        h = encoder(dummy_input)
        self.assertEqual(h.shape, (4, 512))

        # 2. ProjectionHead
        projector = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
        z = projector(h)
        self.assertEqual(z.shape, (4, 128))

        # 3. SimCLRModel
        simclr = SimCLRModel(hidden_dim=512, projection_dim=128, pretrained=False)
        z_norm, h_rep = simclr(dummy_input)
        self.assertEqual(z_norm.shape, (4, 128))
        self.assertEqual(h_rep.shape, (4, 512))
        # Verify L2 normalization of projection vectors
        norms = torch.norm(z_norm, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

        # 4. Supervised ResNet-18 (Scratch & ImageNet)
        supervised_scratch = SupervisedResNet18(num_classes=9, pretrained=False)
        logits_scratch = supervised_scratch(dummy_input)
        self.assertEqual(logits_scratch.shape, (4, 9))

        supervised_imagenet = SupervisedResNet18(num_classes=9, pretrained=True)
        logits_imagenet = supervised_imagenet(dummy_input)
        self.assertEqual(logits_imagenet.shape, (4, 9))

        # 5. LinearProbeClassifier (encoder strictly frozen)
        probe = LinearProbeClassifier(encoder, num_classes=9)
        logits_probe = probe(dummy_input)
        self.assertEqual(logits_probe.shape, (4, 9))
        for p in probe.encoder.parameters():
            self.assertFalse(p.requires_grad)
        self.assertTrue(probe.fc.weight.requires_grad)

        # 6. FineTuneClassifier (encoder trainable)
        finetune = FineTuneClassifier(encoder, num_classes=9)
        logits_ft = finetune(dummy_input)
        self.assertEqual(logits_ft.shape, (4, 9))
        for p in finetune.encoder.parameters():
            self.assertTrue(p.requires_grad)
        self.assertTrue(finetune.fc.weight.requires_grad)

    def test_metrics_computation(self):
        """Test calculation of all required classification metrics and confusion matrix."""
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 2])  # 11 out of 12 correct
        # Mock probabilities
        y_prob = np.zeros((12, 9))
        for i, pred in enumerate(y_pred):
            y_prob[i, pred] = 0.8
            y_prob[i, (pred + 1) % 9] = 0.2

        metrics = compute_metrics(y_true, y_pred, y_prob)

        self.assertIn("accuracy", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 11 / 12, places=4)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("macro_roc_auc", metrics)
        self.assertIsNotNone(metrics["macro_roc_auc"])
        self.assertIn("confusion_matrix", metrics)
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (9, 9))


if __name__ == "__main__":
    unittest.main()
