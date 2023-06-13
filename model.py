import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp


# ! Definition of Semantic Segmentator
class BaseModel(nn.Module):
    def __init__(self, classes):
        super(BaseModel, self).__init__()

        self.backbone = models.segmentation.fcn_resnet50(pretrained=True)
        self.backbone.classifier[4] = nn.Conv2d(512, len(classes), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x


class fcn_resnet101(nn.Module):
    def __init__(self, classes):
        super(fcn_resnet101, self).__init__()

        self.backbone = models.segmentation.fcn_resnet101(pretrained=True)
        self.backbone.classifier[4] = nn.Conv2d(512, len(classes), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x


class DeepLabV3_resnet50(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_resnet50, self).__init__()

        self.backbone = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.backbone.classifier[-1] = nn.Conv2d(256, len(num_classes), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x


class DeepLabV3_resnet101(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_resnet101, self).__init__()

        self.backbone = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.backbone.classifier[-1] = nn.Conv2d(256, len(num_classes), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Unet_resnet101(nn.Module):
    def __init__(self, classes):
        super(Unet_resnet101, self).__init__()

        self.model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MAnet_resnet50(nn.Module):
    def __init__(self, classes):
        super(MAnet_resnet50, self).__init__()

        self.model = smp.MAnet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MAnet_resnet101(nn.Module):
    def __init__(self, classes):
        super(MAnet_resnet101, self).__init__()

        self.model = smp.MAnet(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabV3Plus_resnet50(nn.Module):
    def __init__(self, classes):
        super(DeepLabV3Plus_resnet50, self).__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class PAN_resnet101(nn.Module):
    def __init__(self, classes):
        super(PAN_resnet101, self).__init__()

        self.model = smp.PAN(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class PAN_efficientnet_b5(nn.Module):
    def __init__(self, classes):
        super(PAN_efficientnet_b5, self).__init__()

        self.model = smp.PAN(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x
