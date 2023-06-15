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


class fcn_resnet101_NotPretrained(nn.Module):
    def __init__(self, classes):
        super(fcn_resnet101_NotPretrained, self).__init__()

        self.backbone = models.segmentation.fcn_resnet101(pretrained=False)
        self.backbone.classifier[4] = nn.Conv2d(512, len(classes), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class fcn_resnet101_NotPretrained_1024(nn.Module):
    def __init__(self, classes):
        super(fcn_resnet101_NotPretrained_1024, self).__init__()

        self.backbone = models.segmentation.fcn_resnet101(pretrained=False)
        self.backbone.classifier[0] = nn.Conv2d(2048, 1024, kernel_size=(3,3), stride=(1, 1), padding=1, bias=False)
        self.backbone.classifier[1] = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.backbone.classifier[2] = nn.SELU()
        self.backbone.classifier[3] = nn.Dropout(p=0.1, inplace=False)
        self.backbone.classifier[4] = nn.Conv2d(1024, len(classes), kernel_size=(1, 1), stride=(1, 1))

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


# pip install --upgrade torch torchvision
class DeepLabV3_MobileNet_v3_Large(nn.Module):
    def __init__(self, classes):
        super(DeepLabV3_MobileNet_v3_Large,self).__init__()

        self.backbone = models.segmentation.deeplabv3.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.backbone.classifier[-1]=nn.Conv2d(256, len(classes), kernel_size=1)
    
    def forward(self,x):
        x = self.backbone(x)
        return x


class SMP_FPN(nn.Module):
    def __init__(self, classes):
        super(SMP_FPN,self).__init__()

        self.backbone = smp.FPN(classes = len(classes))

    def forward(self,x):
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
