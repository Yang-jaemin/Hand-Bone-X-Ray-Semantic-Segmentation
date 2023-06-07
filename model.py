import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
