"""
model.py
--------
Model definitions:
  - BaseCNN:    Lightweight CNN trained from scratch (baseline)
  - ResNetModel: ResNet-18/34/50 with a custom classification head (main model)
"""

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────
# 1. Baseline CNN (scratch)
# ─────────────────────────────────────────────

class BaseCNN(nn.Module):
    """
    Simple 4-block CNN for binary classification.
    Used as baseline to measure transfer learning gains.
    
    Input:  (B, 3, 224, 224)
    Output: (B, 2) logits
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # → 112×112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # → 56×56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # → 28×28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # → 14×14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → (B, 256, 1, 1)
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# 2. ResNet with custom head (main model)
# ─────────────────────────────────────────────

class ResNetModel(nn.Module):
    """
    ResNet-18/34/50 fine-tuned for binary deforestation classification.

    Two-phase training strategy:
      Phase 1: Freeze backbone, train head only (fast convergence)
      Phase 2: Unfreeze all, end-to-end fine-tuning (best accuracy)

    Args:
        backbone:    'resnet18' | 'resnet34' | 'resnet50'
        num_classes: 2 for binary classification
        dropout:     Dropout rate in classifier head
        pretrained:  Use ImageNet weights
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        dropout: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()

        weights_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        }
        model_map = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }

        assert backbone in model_map, f"backbone must be one of {list(model_map.keys())}"

        weights = weights_map[backbone] if pretrained else None
        base = model_map[backbone](weights=weights)

        # Extract feature layers (everything except the final FC)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)
        in_features = base.fc.in_features  # 512 for resnet18/34, 2048 for resnet50

        # Custom classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.backbone_name = backbone

    def freeze_backbone(self):
        """Phase 1: freeze backbone, only train the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[Model] Backbone frozen. Training head only.")

    def unfreeze_backbone(self):
        """Phase 2: unfreeze all layers for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"[Model] Backbone unfrozen. End-to-end fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(model_type: str = "resnet18", **kwargs) -> nn.Module:
    """
    Build a model by name.
    
    Args:
        model_type: 'baseline' | 'resnet18' | 'resnet34' | 'resnet50'
    """
    if model_type == "baseline":
        model = BaseCNN(**kwargs)
        print(f"[Model] Built BaseCNN | Params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model = ResNetModel(backbone=model_type, **kwargs)
        info = model.count_parameters()
        print(f"[Model] Built ResNetModel({model_type}) | Total: {info['total']:,} | Trainable: {info['trainable']:,}")

    return model


if __name__ == "__main__":
    # Sanity check
    x = torch.randn(4, 3, 224, 224)

    baseline = build_model("baseline")
    print("BaseCNN output:", baseline(x).shape)

    resnet = build_model("resnet18")
    resnet.freeze_backbone()
    print("ResNet output:", resnet(x).shape)
