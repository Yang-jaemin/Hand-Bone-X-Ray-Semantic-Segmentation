import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]


# ! Definition of Semantic Segmentator
class BaseModel(nn.Module):
    def __init__(self, classes=CLASSES):
        super(BaseModel, self).__init__()

        self.backbone = models.segmentation.fcn_resnet50(pretrained=True)
        self.backbone.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x
