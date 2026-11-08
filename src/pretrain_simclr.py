"""Self-Supervised Learning (SimCLR) pretraining module.

Pretrains ResNet-18 encoder using Normalized Temperature-scaled Cross Entropy (NT-Xent)
loss on unlabeled histopathological images strictly from the training split.
"""

import os
import time
import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import SimCLRModel
from src.losses import NTXentLoss
from src.datasets import get_simclr_pretrain_loader
from src.utils import setup_logger, save_checkpoint, load_checkpoint, get_device


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scaled decay between base_lr and min_lr
        decayed = (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine_decay
        return decayed

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_simclr_epoch(
    model: SimCLRModel,
    loader: DataLoader,
    criterion: NTXentLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
) -> float:
    """Trains SimCLR for one epoch across minibatches of paired views."""
    model.train()
    total_loss = 0.0
    batch_count = 0

    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    # Autocast on MPS or CUDA
    amp_enabled = use_amp and device_type in ["cuda", "mps"]

    for batch_idx, (view1, view2) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        if amp_enabled:
            with torch.amp.autocast(device_type=device_type):
                z1, _ = model(view1)
                z2, _ = model(view2)
                loss = criterion(z1, z2)
        else:
            z1, _ = model(view1)
            z2, _ = model(view2)
            loss = criterion(z1, z2)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


def pretrain_simclr(
    config: dict,
    resume_checkpoint: Optional[str] = None,
    max_epochs: Optional[int] = None,
    max_batches_per_epoch: Optional[int] = None,
    logger=None,
) -> Tuple[SimCLRModel, Dict[str, list]]:
    """Executes full SimCLR contrastive pretraining pipeline."""
    if logger is None:
        logger = setup_logger("SimCLR-Pretrain")

    device = get_device(config.get("hardware", {}).get("device", "auto"))
    logger.info(f"Pretraining SimCLR on device: {device}")

    # Hyperparameters
    simclr_cfg = config.get("simclr_pretrain", {})
    epochs = max_epochs if max_epochs is not None else simclr_cfg.get("epochs", 30)
    warmup_epochs = simclr_cfg.get("warmup_epochs", 3)
    temperature = simclr_cfg.get("temperature", 0.1)
    lr = simclr_cfg.get("learning_rate", 1e-3)
    weight_decay = simclr_cfg.get("weight_decay", 1e-4)
    hidden_dim = config.get("model", {}).get("projection_head", {}).get("hidden_dim", 512)
    projection_dim = config.get("model", {}).get("projection_head", {}).get("output_dim", 128)

    checkpoints_dir = config.get("project", {}).get("checkpoints_dir", "checkpoints")
    if not os.path.isabs(checkpoints_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoints_dir = os.path.normpath(os.path.join(project_root, checkpoints_dir))
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize model
    model = SimCLRModel(
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        pretrained=False,
    ).to(device)

    # Loss and optimizer
    criterion = NTXentLoss(temperature=temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=lr,
    )

    # Checkpoint resuming
    start_epoch = 1
    best_loss = float("inf")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Resuming pretraining from checkpoint: {resume_checkpoint}")
        ckpt = load_checkpoint(resume_checkpoint, device=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))

    # Data loader (ONLY training split)
    loader = get_simclr_pretrain_loader(config)
    logger.info(
        f"SimCLR dataset loaded: {len(loader.dataset)} unlabeled training images. "
        f"Batch size: {loader.batch_size}, Total batches: {len(loader)}"
    )

    history = {"epoch": [], "loss": [], "lr": []}

    use_amp = device.type in ["cuda", "mps"]
    scaler = torch.amp.GradScaler(device_type=device.type) if device.type == "cuda" else None

    logger.info(f"Beginning contrastive pretraining for {epochs} epochs (warmup: {warmup_epochs})")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        loss = train_simclr_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            max_batches=max_batches_per_epoch,
        )

        scheduler.step()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["loss"].append(loss)
        history["lr"].append(current_lr)

        logger.info(
            f"Pretrain Epoch {epoch:02d}/{epochs:02d} | "
            f"NT-Xent Loss: {loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Periodic checkpoint
        if loss < best_loss:
            best_loss = loss
            # Save full SimCLR state
            best_simclr_path = os.path.join(checkpoints_dir, "simclr_best.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "config": config,
                },
                best_simclr_path,
            )
            # Crucially: save standalone encoder checkpoint
            encoder_path = os.path.join(checkpoints_dir, "simclr_encoder_best.pth")
            model.save_encoder(encoder_path)
            logger.info(f"Saved best encoder checkpoint to {encoder_path}")

    # Save final model and encoder
    final_simclr_path = os.path.join(checkpoints_dir, "simclr_final.pth")
    save_checkpoint(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "loss": loss,
            "config": config,
        },
        final_simclr_path,
    )
    final_encoder_path = os.path.join(checkpoints_dir, "simclr_encoder_final.pth")
    model.save_encoder(final_encoder_path)
    logger.info(f"Pretraining complete. Final encoder saved to {final_encoder_path}")

    return model, history
