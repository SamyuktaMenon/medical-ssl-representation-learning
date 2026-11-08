"""Dataset loading, stratified subset generation, and leak-proof data loaders."""

import os
from typing import Tuple, Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from src.augmentations import get_supervised_transforms, SimCLRAugmentation


def _resolve_dir(path: str) -> str:
    """Resolves relative paths relative to the project root directory."""
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(project_root, path))
    return path


class PathMNISTDataset(Dataset):
    """PathMNIST dataset loading directly from the official .npz format.

    Guarantees isolation between training, validation, and test splits.
    """

    def __init__(
        self,
        split: str = "train",
        root_dir: str = "./data",
        transform=None,
    ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.root_dir = _resolve_dir(root_dir)
        self.transform = transform

        npz_path = os.path.join(self.root_dir, "pathmnist.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"PathMNIST file not found at {npz_path}. Run data download first."
            )

        data = np.load(npz_path)
        self.images = data[f"{split}_images"]  # shape: (N, 28, 28, 3), uint8
        self.labels = data[f"{split}_labels"].squeeze().astype(np.int64)  # shape: (N,)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = int(self.labels[idx])

        # Convert numpy uint8 (H, W, C) to PIL Image for torchvision transforms
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            # Fallback to ToTensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, label


class UnlabeledPathMNISTDataset(Dataset):
    """Unlabeled dataset for self-supervised pretraining (SimCLR).

    Strictly uses ONLY images from the training split.
    Validation and test splits are never accessed.
    Returns two independently augmented views (x_i, x_j) of each training image.
    """

    def __init__(
        self,
        root_dir: str = "./data",
        transform: Optional[SimCLRAugmentation] = None,
    ):
        self.root_dir = _resolve_dir(root_dir)
        npz_path = os.path.join(self.root_dir, "pathmnist.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"PathMNIST file not found at {npz_path}")

        data = np.load(npz_path)
        # ONLY training split is used for representation pretraining
        self.images = data["train_images"]  # shape: (89996, 28, 28, 3)
        self.transform = transform if transform is not None else SimCLRAugmentation()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        pil_img = Image.fromarray(img)
        view1, view2 = self.transform(pil_img)
        return view1, view2


def get_stratified_train_subset(
    train_dataset: PathMNISTDataset,
    fraction: float,
    seed: int = 42,
) -> Tuple[Subset, Dict[int, int], np.ndarray]:
    """Generates a deterministic, stratified subset of the training dataset.

    Args:
        train_dataset: PathMNISTDataset configured with split='train'.
        fraction: Proportion of training data (e.g. 0.01, 0.05, 0.10, 1.00).
        seed: Random seed for reproducibility.

    Returns:
        subset: torch.utils.data.Subset wrapping the selected indices.
        class_counts: Exact count of samples for each class in the subset.
        selected_indices: Array of selected training dataset indices.
    """
    assert train_dataset.split == "train", "Subsetting must strictly be performed on train split."
    assert 0.0 < fraction <= 1.0, f"Fraction must be in (0, 1], got {fraction}"

    total_samples = len(train_dataset)
    labels = train_dataset.labels

    if fraction >= 0.9999:
        selected_indices = np.arange(total_samples)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
        selected_indices, _ = next(sss.split(np.zeros(total_samples), labels))
        selected_indices = np.sort(selected_indices)

    subset_labels = labels[selected_indices]
    unique_classes, counts = np.unique(subset_labels, return_counts=True)
    class_counts = {int(cls_id): int(cnt) for cls_id, cnt in zip(unique_classes, counts)}

    subset = Subset(train_dataset, selected_indices)
    return subset, class_counts, selected_indices


def get_downstream_loaders(
    config: dict,
    label_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, int]]:
    """Creates DataLoader objects for training, validation, and test splits.

    Guarantees:
    - Train loader uses deterministic stratified subset of train split.
    - Validation loader uses untouched official validation split.
    - Test loader uses untouched official test split.
    """
    root_dir = config["dataset"]["root_dir"]
    image_size = config["dataset"].get("image_size", 28)
    batch_size = config["downstream"].get("batch_size", 64)
    num_workers = config.get("hardware", {}).get("num_workers", 0)
    pin_memory = config.get("hardware", {}).get("pin_memory", False)

    norm_cfg = config.get("augmentations", {}).get("normalization", {})
    mean = norm_cfg.get("mean", [0.7405, 0.5330, 0.7058])
    std = norm_cfg.get("std", [0.1235, 0.1768, 0.1244])

    train_transform = get_supervised_transforms(
        image_size=image_size, is_train=True, mean=mean, std=std
    )
    eval_transform = get_supervised_transforms(
        image_size=image_size, is_train=False, mean=mean, std=std
    )

    full_train_dataset = PathMNISTDataset(split="train", root_dir=root_dir, transform=train_transform)
    val_dataset = PathMNISTDataset(split="val", root_dir=root_dir, transform=eval_transform)
    test_dataset = PathMNISTDataset(split="test", root_dir=root_dir, transform=eval_transform)

    train_subset, class_counts, _ = get_stratified_train_subset(
        full_train_dataset, fraction=label_fraction, seed=seed
    )

    # Drop trailing incomplete batch during training if dataset is larger than batch size
    # This prevents BatchNorm instability on micro-batches (e.g. 3 samples in 1% subset)
    drop_last = len(train_subset) > batch_size

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_counts


def get_simclr_pretrain_loader(config: dict) -> DataLoader:
    """Creates DataLoader for self-supervised SimCLR pretraining.

    Strictly draws unlabeled images from the training split.
    """
    root_dir = config["dataset"]["root_dir"]
    image_size = config["dataset"].get("image_size", 28)
    batch_size = config["simclr_pretrain"].get("batch_size", 256)
    num_workers = config.get("hardware", {}).get("num_workers", 0)
    pin_memory = config.get("hardware", {}).get("pin_memory", False)

    aug_cfg = config.get("augmentations", {}).get("simclr", {})
    norm_cfg = config.get("augmentations", {}).get("normalization", {})

    simclr_transform = SimCLRAugmentation(
        image_size=image_size,
        crop_scale=tuple(aug_cfg.get("crop_scale", [0.6, 1.0])),
        horizontal_flip_p=aug_cfg.get("horizontal_flip_p", 0.5),
        vertical_flip_p=aug_cfg.get("vertical_flip_p", 0.5),
        color_jitter_p=aug_cfg.get("color_jitter", {}).get("p", 0.8),
        brightness=aug_cfg.get("color_jitter", {}).get("brightness", 0.4),
        contrast=aug_cfg.get("color_jitter", {}).get("contrast", 0.4),
        saturation=aug_cfg.get("color_jitter", {}).get("saturation", 0.4),
        hue=aug_cfg.get("color_jitter", {}).get("hue", 0.1),
        random_gray_p=aug_cfg.get("random_gray_p", 0.1),
        gaussian_blur_p=aug_cfg.get("gaussian_blur", {}).get("p", 0.5),
        kernel_size=aug_cfg.get("gaussian_blur", {}).get("kernel_size", 3),
        mean=norm_cfg.get("mean", [0.7405, 0.5330, 0.7058]),
        std=norm_cfg.get("std", [0.1235, 0.1768, 0.1244]),
    )

    unlabeled_dataset = UnlabeledPathMNISTDataset(root_dir=root_dir, transform=simclr_transform)

    pretrain_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # drop last incomplete batch for contrastive loss stability
    )
    return pretrain_loader

