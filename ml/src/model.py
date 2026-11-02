"""EfficientNet-B0 Transfer Learning Architecture for Food Classification."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger("nutrition_ai.model")


class FoodClassifier(nn.Module):
    """Production food recognition model using transfer learning with EfficientNet-B0."""

    def __init__(
        self,
        num_classes: int = 1051,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize EfficientNet-B0 with custom food classification head.

        Args:
            num_classes: Number of distinct food target classes.
            pretrained: Whether to load ImageNet pre-trained weights.
            dropout: Dropout probability before the final linear layer.
            freeze_backbone: Whether to freeze feature extractor layers for initial training.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Load EfficientNet-B0 backbone
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
                logger.info("Loaded pre-trained ImageNet weights for EfficientNet-B0.")
            except Exception as e:
                logger.warning(
                    f"Could not download pre-trained weights ({e}). Initializing random weights."
                )
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            logger.info("Initialized EfficientNet-B0 with random weights.")

        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # Initialize new classifier layer with He / Kaiming normal
        nn.init.kaiming_normal_(self.backbone.classifier[1].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.backbone.classifier[1].bias)

        if freeze_backbone:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone feature extractor layers."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = not freeze
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Backbone feature extractor parameters are now {status}.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for end-to-end fine-tuning."""
        self.freeze_backbone(False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vector before the classifier layer."""
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper returning top-1 class indices and softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        return predictions, confidences

    @torch.no_grad()
    def predict_topk(self, x: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper returning top-K class indices and softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, min(k, self.num_classes), dim=-1)
        return topk_indices, topk_probs


def build_model(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> FoodClassifier:
    """Factory creating configured FoodClassifier model."""
    return FoodClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

