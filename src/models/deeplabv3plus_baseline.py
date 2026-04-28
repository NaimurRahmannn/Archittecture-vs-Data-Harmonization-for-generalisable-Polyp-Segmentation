import torch.nn as nn


class DeepLabV3PlusBaseline(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for DeepLabV3+ baseline. Install it with: pip install torchvision"
            ) from exc

        weights = None
        weights_backbone = None
        if pretrained:
            # Prefer full segmentation weights; fallback to backbone weights if unavailable.
            try:
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
            except Exception:
                weights = None
                weights_backbone = ResNet50_Weights.DEFAULT

        self.model = deeplabv3_resnet50(weights=weights, weights_backbone=weights_backbone)

        if hasattr(self.model, "classifier") and self.model.classifier is not None:
            in_ch = self.model.classifier[-1].in_channels
            self.model.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)

        if hasattr(self.model, "aux_classifier") and self.model.aux_classifier is not None:
            in_ch_aux = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 1, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return out["out"]
