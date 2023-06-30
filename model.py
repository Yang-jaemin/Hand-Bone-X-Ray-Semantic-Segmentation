import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp


# FCN
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
        self.backbone.classifier[0] = nn.Conv2d(
            2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.backbone.classifier[1] = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.backbone.classifier[2] = nn.SELU()
        self.backbone.classifier[3] = nn.Dropout(p=0.1, inplace=False)
        self.backbone.classifier[4] = nn.Conv2d(
            1024, len(classes), kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


# DeepLabV3
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


class DeepLabV3_MobileNet_v3_Large(nn.Module):
    def __init__(self, classes):
        super(DeepLabV3_MobileNet_v3_Large, self).__init__()

        self.backbone = models.segmentation.deeplabv3.deeplabv3_mobilenet_v3_large(
            pretrained=True
        )
        self.backbone.classifier[-1] = nn.Conv2d(256, len(classes), kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return x


class DeepLabV3_resnet50_custom(nn.Module):
    def __init__(self, classes):
        super(DeepLabV3_resnet50_custom, self).__init__()

        self.backbone = models.segmentation.deeplabv3_resnet50(pretrained=False)

        self.backbone.classifier[-5].project[0] = nn.Conv2d(
            1280, 1024, kernel_size=1, stride=1, bias=False
        )
        self.backbone.classifier[-5].project[1] = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.backbone.classifier[-5].project[2] = nn.SELU()

        self.backbone.classifier[-4] = nn.Conv2d(
            1024, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.classifier[-3] = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.backbone.classifier[-2] = nn.SELU()
        self.backbone.classifier[-1] = nn.Conv2d(512, 29, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
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


# FPN
class SMP_FPN(nn.Module):
    def __init__(self, classes):
        super(SMP_FPN, self).__init__()

        self.backbone = smp.FPN(classes=len(classes))

    def forward(self, x):
        x = self.backbone(x)
        return x


# Unet
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


# MANet
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


# PAN
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


# HRNet models (timm)
class HRNetV2_W48(nn.Module):
    def __init__(self, classes):
        super(HRNetV2_W48, self).__init__()

        self.model = timm.create_model("hrnet_w48", pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabV3plus_hrnet(nn.Module):
    def __init__(self, classes):
        super(DeepLabV3plus_hrnet, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=("tu-hrnet_w48"),
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class PAN_hrnet(nn.Module):
    def __init__(self, classes):
        super(PAN_hrnet, self).__init__()
        self.model = smp.PAN(
            encoder_name=("tu-hrnet_w48"),
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Unetplusplus_hrnet(nn.Module):
    def __init__(self, classes):
        super(Unetplusplus_hrnet, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=("tu-hrnet_w48"),
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# ! Basic U-Net scratch version
class Basic_UNet(nn.Module):
    def __init__(self, classes):
        super(Basic_UNet, self).__init__()

        self.A_encoder = nn.Sequential(  # ! Output = 64 X 512 X 512
            BasicConv(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        self.B_encoder = nn.Sequential(  # ! Output = 128 X 256 X 256
            BasicConv(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        self.C_encoder = nn.Sequential(  # ! Output = 256 X 128 X 128
            BasicConv(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        self.D_encoder = nn.Sequential(  # ! Output = 512 X 64 X 64
            BasicConv(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        self.Enc_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Enc_Dec_Inter = nn.Sequential(  # ! Output = 512 X 128 X 128
            BasicConv(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=2, stride=2, bias=True
            ),
        )

        self.D_decoder = nn.Sequential(  # ! Output = 256 X 256 X 256
            BasicConv(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=2, stride=2, bias=True
            ),
        )
        self.C_decoder = nn.Sequential(  # ! Output = 128 X 512 X 512
            BasicConv(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2, bias=True
            ),
        )
        self.B_decoder = nn.Sequential(  # ! Output = 64 X 1024 X 1024
            BasicConv(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=True
            ),
        )
        self.A_decoder = nn.Sequential(  # ! Output = 29 X 1024 X 1024
            BasicConv(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            BasicConv(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=len(classes),
                kernel_size=1,
                stride=1,
                bias=True,
            ),
        )

    def forward(self, input):
        A_enc_out = self.A_encoder(input)
        A_pool_out = self.Enc_pool(A_enc_out)

        B_enc_out = self.B_encoder(A_pool_out)
        B_pool_out = self.Enc_pool(B_enc_out)

        C_enc_out = self.C_encoder(B_pool_out)
        C_pool_out = self.Enc_pool(C_enc_out)

        D_enc_out = self.D_encoder(C_pool_out)
        D_pool_out = self.Enc_pool(D_enc_out)

        middle = self.Enc_Dec_Inter(D_pool_out)

        D_dec_out = self.D_decoder(torch.cat((D_enc_out, middle), dim=1))
        C_dec_out = self.C_decoder(torch.cat((C_enc_out, D_dec_out), dim=1))
        B_dec_out = self.B_decoder(torch.cat((B_enc_out, C_dec_out), dim=1))
        A_dec_out = self.A_decoder(torch.cat((A_enc_out, B_dec_out), dim=1))

        output = A_dec_out
        return output


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(BasicConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
        )

    def forward(self, input):
        output = self.conv_block(input)
        return output


# ! HRNet v2 scratch version
class HRNetv2(nn.Module):
    def __init__(self, classes, C=48):
        super(HRNetv2, self).__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.SELU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.SELU(),
        )

        self.A1_Block = nn.Sequential()
        for idx1 in range(4):
            if idx1 == 0:
                self.A1_Block.add_module(
                    f"A1_Block_{idx1+1}",
                    Stage1(in_channels=64, mid_channels=64, out_channels=256),
                )
            else:
                self.A1_Block.add_module(
                    f"A1_Block_{idx1+1}",
                    Stage1(in_channels=256, mid_channels=64, out_channels=256),
                )
        self.Stage1_to_Stage2 = Stage1toStage2_Stream(
            in_channels=256, Hchannels=C, MHchannels=2 * C
        )

        self.A2_Block = nn.Sequential()
        self.B1_Block = nn.Sequential()
        for idx2 in range(4):
            self.A2_Block.add_module(f"A2_Block_{idx2+1}", Stage2(channels=C))
            self.B1_Block.add_module(f"B1_Block_{idx2+1}", Stage2(channels=2 * C))
        self.Stage2_to_Stage3 = Stage2toStage3_Stream(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C
        )

        self.A3_B2_C1_Block1 = Alpha_Stage3(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C
        )
        self.A3_B2_C1_Block2 = Alpha_Stage3(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C
        )
        self.A3_B2_C1_Block3 = Alpha_Stage3(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C
        )
        self.Stage3_to_Stage4 = Beta_Stage3(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C, Lchannels=8 * C
        )

        self.A4_B3_C2_D1_Block1 = Stage4(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C, Lchannels=8 * C
        )
        self.A4_B3_C2_D1_Block2 = Stage4(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C, Lchannels=8 * C
        )
        self.A4_B3_C2_D1_Block3 = Stage4(
            Hchannels=C, MHchannels=2 * C, MLchannels=4 * C, Lchannels=8 * C
        )

        self.total_channels = 48 + 96 + 192 + 384
        self.output_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.total_channels,
                out_channels=self.total_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.total_channels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=self.total_channels,
                out_channels=len(classes),
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, input):
        after_input = self.input_block(input)

        A1_out = self.A1_Block(after_input)
        A1_in, B1_in = self.Stage1_to_Stage2(A1_out)

        A2_out = self.A2_Block(A1_in)
        B1_out = self.B1_Block(B1_in)
        A3_in, B2_in, C1_in = self.Stage2_to_Stage3(A2_out, B1_out)

        A3_out, B2_out, C1_out = self.A3_B2_C1_Block1(A3_in, B2_in, C1_in)
        A3_out, B2_out, C1_out = self.A3_B2_C1_Block2(A3_out, B2_out, C1_out)
        A3_out, B2_out, C1_out = self.A3_B2_C1_Block3(A3_out, B2_out, C1_out)
        A4_in, B3_in, C2_in, D1_in = self.Stage3_to_Stage4(A3_out, B2_out, C1_out)

        A4_out, B3_out, C2_out, D1_out = self.A4_B3_C2_D1_Block1(
            A4_in, B3_in, C2_in, D1_in
        )
        A4_out, B3_out, C2_out, D1_out = self.A4_B3_C2_D1_Block2(
            A4_out, B3_out, C2_out, D1_out
        )
        A4_out, B3_out, C2_out, D1_out = self.A4_B3_C2_D1_Block3(
            A4_out, B3_out, C2_out, D1_out
        )
        align_size = (A4_out.size(2), A4_out.size(3))

        MH_to_H_map = F.interpolate(
            B3_out, size=align_size, mode="bilinear", align_corners=True
        )
        ML_to_H_map = F.interpolate(
            C2_out, size=align_size, mode="bilinear", align_corners=True
        )
        L_to_H_map = F.interpolate(
            D1_out, size=align_size, mode="bilinear", align_corners=True
        )

        origin_size = (input.size(2), input.size(3))
        temp1 = torch.cat([A4_out, MH_to_H_map, ML_to_H_map, L_to_H_map], dim=1)
        temp2 = self.output_block(temp1)
        output = F.interpolate(
            temp2, size=origin_size, mode="bilinear", align_corners=True
        )

        return output


# ! 1st Stage Bottle Neck Architecture
class Stage1(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Stage1, self).__init__()

        self.stage1_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.adjust_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = nn.SELU()

    def forward(self, input):
        temp1 = self.stage1_bottle_neck(input)

        if input.size(1) != temp1.size(1):
            input = self.adjust_block(input)

        temp2 = temp1 + input
        output = self.activation(temp2)
        return output


# ! 1st Stage to 2nd Stage Stream Block Architecture for Fusion
class Stage1toStage2_Stream(nn.Module):
    def __init__(self, in_channels, Hchannels, MHchannels):
        super(Stage1toStage2_Stream, self).__init__()

        self.H_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
        )

        self.H_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=MHchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
            nn.SELU(),
        )

    def forward(self, input):
        same_out = self.H_to_H(input)
        down_out = self.H_to_MH(input)
        return same_out, down_out


# ! 2nd Stage Bottle Neck Architecture
class Stage2(nn.Module):
    def __init__(self, channels):
        super(Stage2, self).__init__()

        self.stage2_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.SELU()

    def forward(self, input):
        temp1 = self.stage2_bottle_neck(input)
        temp2 = temp1 + input

        output = self.activation(temp2)
        return output


# ! 2nd Stage to 3rd Stage Stream Block Architecture for Fusion
class Stage2toStage3_Stream(nn.Module):
    def __init__(self, Hchannels, MHchannels, MLchannels):
        super(Stage2toStage3_Stream, self).__init__()

        self.H_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MHchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
            nn.SELU(),
        )

        self.MH_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )

        self.MH_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
            nn.SELU(),
        )

        self.activation = nn.SELU()

    def forward(self, before_high, before_medium_high):
        high_size = (before_high.size(2), before_high.size(3))

        after_high1 = before_high
        after_high2 = F.interpolate(
            before_medium_high, size=high_size, mode="bilinear", align_corners=True
        )
        after_high2 = self.MH_to_H(after_high2)
        after_high = after_high1 + after_high2

        after_medium_high1 = before_medium_high
        after_medium_high2 = self.H_to_MH(before_high)
        after_medium_high = after_medium_high1 + after_medium_high2

        high_out = self.activation(after_high)
        medium_high_out = self.activation(after_medium_high)
        medium_low_out = self.MH_to_ML(medium_high_out)

        return high_out, medium_high_out, medium_low_out


# ! 3rd Stage Before - Bottle Neck Architecture
class Alpha_Stage3(nn.Module):
    def __init__(self, Hchannels, MHchannels, MLchannels):
        super(Alpha_Stage3, self).__init__()

        self.H_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.MH_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.ML_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MLchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MLchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.activation = nn.SELU()

        self.H_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MHchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.H_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.MH_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.MH_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.ML_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.ML_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MHchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )

    def forward(self, before_high, before_medium_high, before_medium_low):
        high_size = (before_high.size(2), before_high.size(3))
        medium_high_size = (before_medium_high.size(2), before_medium_high.size(3))

        after_high1 = before_high
        after_medium_high1 = before_medium_high
        after_medium_low1 = before_medium_low
        for _ in range(4):
            after_high1 = self.H_bottle_neck(after_high1)
            after_medium_high1 = self.MH_bottle_neck(after_medium_high1)
            after_medium_low1 = self.ML_bottle_neck(after_medium_low1)

        after_high2 = F.interpolate(
            after_medium_high1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high2 = self.MH_to_H(after_high2)
        after_high3 = F.interpolate(
            after_medium_low1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high3 = self.ML_to_H(after_high3)
        after_high = after_high1 + after_high2 + after_high3

        after_medium_high2 = self.H_to_MH(after_high1)
        after_medium_high3 = F.interpolate(
            after_medium_low1,
            size=medium_high_size,
            mode="bilinear",
            align_corners=True,
        )
        after_medium_high3 = self.ML_to_MH(after_medium_high3)
        after_medium_high = after_medium_high1 + after_medium_high2 + after_medium_high3

        after_medium_low2 = self.H_to_ML(after_high1)
        after_medium_low3 = self.MH_to_ML(after_medium_high1)
        after_medium_low = after_medium_low1 + after_medium_low2 + after_medium_low3

        high_out = self.activation(after_high)
        medium_high_out = self.activation(after_medium_high)
        medium_low_out = self.activation(after_medium_low)

        return high_out, medium_high_out, medium_low_out


# ! 3rd Stage After - Bottle Neck Architecture
class Beta_Stage3(nn.Module):
    def __init__(self, Hchannels, MHchannels, MLchannels, Lchannels):
        super(Beta_Stage3, self).__init__()

        self.H_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.MH_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.ML_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MLchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MLchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.activation = nn.SELU()

        self.H_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MHchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.H_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.MH_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.MH_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.ML_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.ML_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MHchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.ML_to_L = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=Lchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Lchannels),
            nn.SELU(),
        )

    def forward(self, before_high, before_medium_high, before_medium_low):
        high_size = (before_high.size(2), before_high.size(3))
        medium_high_size = (before_medium_high.size(2), before_medium_high.size(3))
        medium_low_size = (before_medium_low.size(2), before_medium_low.size(3))

        after_high1 = before_high
        after_medium_high1 = before_medium_high
        after_medium_low1 = before_medium_low
        for _ in range(4):
            after_high1 = self.H_bottle_neck(after_high1)
            after_medium_high1 = self.MH_bottle_neck(after_medium_high1)
            after_medium_low1 = self.ML_bottle_neck(after_medium_low1)

        after_high2 = F.interpolate(
            after_medium_high1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high2 = self.MH_to_H(after_high2)
        after_high3 = F.interpolate(
            after_medium_low1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high3 = self.ML_to_H(after_high3)
        after_high = after_high1 + after_high2 + after_high3

        after_medium_high2 = self.H_to_MH(after_high1)
        after_medium_high3 = F.interpolate(
            after_medium_low1,
            size=medium_high_size,
            mode="bilinear",
            align_corners=True,
        )
        after_medium_high3 = self.ML_to_MH(after_medium_high3)
        after_medium_high = after_medium_high1 + after_medium_high2 + after_medium_high3

        after_medium_low2 = self.H_to_ML(after_high1)
        after_medium_low3 = self.MH_to_ML(after_medium_high1)
        after_medium_low = after_medium_low1 + after_medium_low2 + after_medium_low3

        high_out = self.activation(after_high)
        medium_high_out = self.activation(after_medium_high)
        medium_low_out = self.activation(after_medium_low)
        low_out = self.ML_to_L(medium_low_out)

        return high_out, medium_high_out, medium_low_out, low_out


# ! 4th Stage - Bottle Neck Architecture
class Stage4(nn.Module):
    def __init__(self, Hchannels, MHchannels, MLchannels, Lchannels):
        super(Stage4, self).__init__()

        self.H_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.MH_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.ML_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MLchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MLchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.L_bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=Lchannels,
                out_channels=Lchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Lchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Lchannels,
                out_channels=Lchannels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Lchannels),
        )
        self.activation = nn.SELU()

        self.H_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MHchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.H_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.H_to_L = nn.Sequential(
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Hchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=Hchannels,
                out_channels=Lchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Lchannels),
        )
        self.MH_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.MH_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MLchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )
        self.MH_to_L = nn.Sequential(
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=MHchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=MHchannels,
                out_channels=Lchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Lchannels),
        )
        self.ML_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=Hchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.ML_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=MHchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.ML_to_L = nn.Sequential(
            nn.Conv2d(
                in_channels=MLchannels,
                out_channels=Lchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Lchannels),
        )
        self.L_to_H = nn.Sequential(
            nn.Conv2d(
                in_channels=Lchannels, out_channels=Hchannels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(Hchannels),
        )
        self.L_to_MH = nn.Sequential(
            nn.Conv2d(
                in_channels=Lchannels,
                out_channels=MHchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(MHchannels),
        )
        self.L_to_ML = nn.Sequential(
            nn.Conv2d(
                in_channels=Lchannels,
                out_channels=MLchannels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(MLchannels),
        )

    def forward(self, before_high, before_medium_high, before_medium_low, before_low):
        high_size = (before_high.size(2), before_high.size(3))
        medium_high_size = (before_medium_high.size(2), before_medium_high.size(3))
        medium_low_size = (before_medium_low.size(2), before_medium_low.size(3))
        low_size = (before_low.size(2), before_low.size(3))

        after_high1 = before_high
        after_medium_high1 = before_medium_high
        after_medium_low1 = before_medium_low
        after_low1 = before_low
        for _ in range(4):
            after_high1 = self.H_bottle_neck(after_high1)
            after_medium_high1 = self.MH_bottle_neck(after_medium_high1)
            after_medium_low1 = self.ML_bottle_neck(after_medium_low1)
            after_low1 = self.L_bottle_neck(after_low1)

        after_high2 = F.interpolate(
            after_medium_high1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high2 = self.MH_to_H(after_high2)
        after_high3 = F.interpolate(
            after_medium_low1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high3 = self.ML_to_H(after_high3)
        after_high4 = F.interpolate(
            after_low1, size=high_size, mode="bilinear", align_corners=True
        )
        after_high4 = self.L_to_H(after_high4)
        after_high = after_high1 + after_high2 + after_high3 + after_high4

        after_medium_high2 = self.H_to_MH(after_high1)
        after_medium_high3 = F.interpolate(
            after_medium_low1,
            size=medium_high_size,
            mode="bilinear",
            align_corners=True,
        )
        after_medium_high3 = self.ML_to_MH(after_medium_high3)
        after_medium_high4 = F.interpolate(
            after_low1,
            size=medium_high_size,
            mode="bilinear",
            align_corners=True,
        )
        after_medium_high4 = self.L_to_MH(after_medium_high4)
        after_medium_high = (
            after_medium_high1
            + after_medium_high2
            + after_medium_high3
            + after_medium_high4
        )

        after_medium_low2 = self.H_to_ML(after_high1)
        after_medium_low3 = self.MH_to_ML(after_medium_high1)
        after_medium_low4 = F.interpolate(
            after_low1,
            size=medium_low_size,
            mode="bilinear",
            align_corners=True,
        )
        after_medium_low4 = self.L_to_ML(after_medium_low4)
        after_medium_low = (
            after_medium_low1
            + after_medium_low2
            + after_medium_low3
            + after_medium_low4
        )

        after_low2 = self.H_to_L(after_high1)
        after_low3 = self.MH_to_L(after_medium_high1)
        after_low4 = self.ML_to_L(after_medium_low1)
        after_low = after_low1 + after_low2 + after_low3 + after_low4

        high_out = self.activation(after_high)
        medium_high_out = self.activation(after_medium_high)
        medium_low_out = self.activation(after_medium_low)
        low_out = self.activation(after_low)

        return high_out, medium_high_out, medium_low_out, low_out
