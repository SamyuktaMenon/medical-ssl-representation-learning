"""Neural network architectures: ResNet-18 encoder, SimCLR projection head, and classification heads."""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18Encoder(nn.Module):
    """ResNet-18 feature extractor that outputs 512-dimensional representations."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)

        # Remove the classification fc layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dimensional feature embedding h from image tensor x."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = torch.flatten(x, 1)  # shape: (B, 512)
        return h


class ProjectionHead(nn.Module):
    """Non-linear 2-layer MLP projection head g(.) mapping representation h to latent space z.

    Architecture: Linear(512 -> hidden_dim) -> BatchNorm1d -> ReLU -> Linear(hidden_dim -> output_dim)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLRModel(nn.Module):
    """Full SimCLR framework combining ResNet-18 encoder f(.) and projection head g(.)."""

    def __init__(
        self,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.projector = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns normalized projection vector z and feature representation h."""
        h = self.encoder(x)
        z = self.projector(h)
        z_norm = F.normalize(z, dim=1)
        return z_norm, h

    def save_encoder(self, filepath: str) -> None:
        """Saves only the trained encoder weights, stripping the projection head."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.encoder.state_dict(), filepath)

    def load_encoder(self, filepath: str, map_location: Optional[torch.device] = None) -> None:
        """Loads trained encoder weights from a saved checkpoint."""
        state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
        self.encoder.load_state_dict(state_dict)


class SupervisedResNet18(nn.Module):
    """ResNet-18 classifier for supervised baseline and ImageNet transfer learning."""

    def __init__(self, num_classes: int = 9, pretrained: bool = False):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.fc = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        logits = self.fc(h)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts representations h before the linear classification head."""
        return self.encoder(x)


class LinearProbeClassifier(nn.Module):
    """Downstream classifier with frozen encoder weights.

    Only the linear classification head is trained.
    """

    def __init__(self, encoder: ResNet18Encoder, num_classes: int = 9):
        super().__init__()
        self.encoder = encoder
        # Strictly freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder is kept in evaluation mode during linear probing
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x)
        logits = self.fc(h)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


class FineTuneClassifier(nn.Module):
    """Downstream classifier with fully trainable encoder and linear head."""

    def __init__(self, encoder: ResNet18Encoder, num_classes: int = 9):
        super().__init__()
        self.encoder = encoder
        # All encoder parameters are trainable
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        logits = self.fc(h)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

