"""Environment and hardware verification test suite.

Validates:
1. Python environment and core package imports
2. Device availability (MPS, CUDA, or CPU fallback)
3. MedMNIST installation and configuration accessibility
4. Directory structure and required file existence
"""

import os
import sys
import unittest
import yaml
import torch
import medmnist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestEnvironment(unittest.TestCase):
    def test_core_dependencies(self):
        """Verify that all scientific and ML packages import correctly."""
        import numpy as np
        import pandas as pd
        import sklearn
        import torchvision

        self.assertIsNotNone(np.__version__)
        self.assertIsNotNone(pd.__version__)
        self.assertIsNotNone(sklearn.__version__)
        self.assertIsNotNone(torchvision.__version__)
        self.assertIsNotNone(torch.__version__)
        self.assertIsNotNone(medmnist.__version__)

    def test_device_acceleration(self):
        """Verify compute device availability."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            x = torch.ones((2, 2), device=device)
            y = x * 2
            self.assertEqual(y.sum().item(), 8.0)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            x = torch.ones((2, 2), device=device)
            y = x * 2
            self.assertEqual(y.sum().item(), 8.0)
        else:
            device = torch.device("cpu")
            x = torch.ones((2, 2), device=device)
            self.assertEqual(x.sum().item(), 4.0)

    def test_config_structure(self):
        """Verify config.yaml can be parsed and contains required keys."""
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
        self.assertTrue(os.path.exists(config_path), f"Missing {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        required_sections = [
            "project",
            "dataset",
            "hardware",
            "label_efficiency",
            "augmentations",
            "model",
            "simclr_pretrain",
            "downstream",
            "evaluation",
        ]
        for section in required_sections:
            self.assertIn(section, config, f"Missing section '{section}' in config.yaml")

        self.assertEqual(config["dataset"]["num_classes"], 9)
        self.assertEqual(
            config["label_efficiency"]["proportions"], [0.01, 0.05, 0.10, 1.00]
        )

    def test_directory_structure(self):
        """Verify all project directories exist."""
        expected_dirs = [
            "configs",
            "data",
            "notebooks",
            "src",
            "scripts",
            "tests",
            "checkpoints",
            "results/metrics",
            "results/plots",
            "results/experiment_logs",
        ]
        for d in expected_dirs:
            dir_path = os.path.join(PROJECT_ROOT, d)
            self.assertTrue(os.path.isdir(dir_path), f"Directory {dir_path} does not exist")

    def test_dataset_present(self):
        """Verify that pathmnist.npz exists in data/ and has expected splits."""
        data_path = os.path.join(PROJECT_ROOT, "data", "pathmnist.npz")
        self.assertTrue(os.path.exists(data_path), f"Missing {data_path}")

        import numpy as np

        data = np.load(data_path)
        self.assertIn("train_images", data)
        self.assertIn("train_labels", data)
        self.assertIn("val_images", data)
        self.assertIn("val_labels", data)
        self.assertIn("test_images", data)
        self.assertIn("test_labels", data)
        self.assertEqual(data["train_images"].shape, (89996, 28, 28, 3))
        self.assertEqual(data["val_images"].shape, (10004, 28, 28, 3))
        self.assertEqual(data["test_images"].shape, (7180, 28, 28, 3))


if __name__ == "__main__":
    unittest.main()
