"""Verification test suite for Phase 3: Contrastive Pretraining, Checkpointing, and Downstream Probing."""

import os
import copy
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils import set_seed, get_device, load_config
from src.models import ResNet18Encoder, SimCLRModel, LinearProbeClassifier, FineTuneClassifier
from src.losses import NTXentLoss
from src.pretrain_simclr import pretrain_simclr

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))


class TestPhase3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.device = get_device("auto")
        cls.config = load_config(CONFIG_PATH)

    def test_encoder_checkpoint_save_and_load(self):
        """Verify that encoder weights can be cleanly extracted, saved, and loaded independently."""
        simclr = SimCLRModel(hidden_dim=512, projection_dim=128, pretrained=False)
        test_ckpt_path = os.path.join(CHECKPOINTS_DIR, "test_encoder_export.pth")

        # Save encoder
        simclr.save_encoder(test_ckpt_path)
        self.assertTrue(os.path.exists(test_ckpt_path))

        # Load into fresh standalone ResNet18Encoder
        fresh_encoder = ResNet18Encoder(pretrained=False)
        state_dict = torch.load(test_ckpt_path, map_location="cpu", weights_only=True)
        fresh_encoder.load_state_dict(state_dict)

        # Check weights are identical
        dummy_input = torch.randn(2, 3, 28, 28)
        simclr.eval()
        fresh_encoder.eval()
        with torch.no_grad():
            _, h1 = simclr(dummy_input)
            h2 = fresh_encoder(dummy_input)
        torch.testing.assert_close(h1, h2, atol=1e-5, rtol=1e-5)

        # Clean up
        if os.path.exists(test_ckpt_path):
            os.remove(test_ckpt_path)

    def test_linear_probe_freezes_encoder(self):
        """Provide concrete evidence that LinearProbeClassifier strictly freezes encoder weights."""
        encoder = ResNet18Encoder(pretrained=False).to(self.device)
        model = LinearProbeClassifier(encoder=encoder, num_classes=9).to(self.device)

        # Snapshot initial encoder weights
        init_encoder_weights = {
            name: param.clone() for name, param in model.encoder.named_parameters()
        }
        init_head_weight = model.fc.weight.clone()

        # Optimizer only for linear head
        optimizer = optim.AdamW(model.fc.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        dummy_x = torch.randn(8, 3, 28, 28, device=self.device)
        dummy_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=self.device)

        # Optimization step
        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

        # 1. Check that all encoder gradients are None or zero
        for name, param in model.encoder.named_parameters():
            self.assertFalse(param.requires_grad, f"Param {name} should have requires_grad=False")
            self.assertIsNone(param.grad, f"Param {name} gradient should be None")
            # 2. Check encoder weights are strictly identical
            torch.testing.assert_close(
                param,
                init_encoder_weights[name],
                msg=f"Encoder weights in {name} changed during linear probing!",
            )

        # 3. Check linear head was updated
        self.assertFalse(
            torch.equal(model.fc.weight, init_head_weight),
            "Linear probe head weights must update after training step",
        )

    def test_finetune_updates_encoder(self):
        """Provide concrete evidence that FineTuneClassifier updates both encoder and classification head."""
        encoder = ResNet18Encoder(pretrained=False).to(self.device)
        model = FineTuneClassifier(encoder=encoder, num_classes=9).to(self.device)

        init_encoder_conv1 = model.encoder.conv1.weight.clone()
        init_head_weight = model.fc.weight.clone()

        optimizer = optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        dummy_x = torch.randn(8, 3, 28, 28, device=self.device)
        dummy_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=self.device)

        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

        # Check encoder conv1 gradients exist and weights updated
        self.assertIsNotNone(model.encoder.conv1.weight.grad)
        self.assertGreater(torch.norm(model.encoder.conv1.weight.grad).item(), 0.0)
        self.assertFalse(
            torch.equal(model.encoder.conv1.weight, init_encoder_conv1),
            "Encoder weights must update during fine-tuning!",
        )

        # Check head weights updated
        self.assertFalse(
            torch.equal(model.fc.weight, init_head_weight),
            "Classification head weights must update during fine-tuning!",
        )

    def test_simclr_small_batch_smoke_test(self):
        """Smoke-test SimCLR pretraining loop on small batch of real data."""
        # Use isolated test checkpoint directory so production weights are never clobbered
        test_config = copy.deepcopy(self.config)
        test_ckpt_dir = os.path.join(CHECKPOINTS_DIR, "test_tmp")
        test_config["project"]["checkpoints_dir"] = test_ckpt_dir
        os.makedirs(test_ckpt_dir, exist_ok=True)

        model, history = pretrain_simclr(
            config=test_config,
            max_epochs=2,
            max_batches_per_epoch=3,
        )

        self.assertEqual(len(history["epoch"]), 2)
        self.assertEqual(len(history["loss"]), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in history["loss"]))

        # Verify best and final encoder checkpoints were created in test dir
        best_encoder_path = os.path.join(test_ckpt_dir, "simclr_encoder_best.pth")
        final_encoder_path = os.path.join(test_ckpt_dir, "simclr_encoder_final.pth")
        self.assertTrue(os.path.exists(best_encoder_path), f"Missing {best_encoder_path}")
        self.assertTrue(os.path.exists(final_encoder_path), f"Missing {final_encoder_path}")

        # Clean up test files
        for p in [best_encoder_path, final_encoder_path, os.path.join(test_ckpt_dir, "simclr_best.pth"), os.path.join(test_ckpt_dir, "simclr_final.pth")]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(test_ckpt_dir):
            os.rmdir(test_ckpt_dir)


if __name__ == "__main__":
    unittest.main()

