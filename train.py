import datetime
import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from dataset import XRayDataset
from model import BaseModel

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
BATCH_SIZE = 8
RANDOM_SEED = 127
LR = 1e-4
NUM_EPOCHS = 50
VAL_EVERY = 1
SAVED_DIR = "/opt/ml/input/code/best_models"
if not os.path.isdir(SAVED_DIR):
    os.mkdir(SAVED_DIR)

# ! Albumentation Transforms & Generation of Train/Valid Dataset
album_transform = A.Resize(512, 512)
train_dataset = XRayDataset(is_train=True, transforms=album_transform)
valid_dataset = XRayDataset(is_train=False, transforms=album_transform)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=False
)


# ! Definitions of Optionable Training functions & Wandb Configuration
def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2.0 * intersection + eps) / (
        torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
    )


def save_model(model, file_name="fcn_resnet50_best_model.pt"):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)


wandb.init(
    project="HandBoneSeg",
    notes="Baseline Code Test",
    config={
        "model": "BaseModel",
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "random_seed": RANDOM_SEED,
    },
    tags=["Resize"],
)


# ! Validation Process
def validation(epoch, model, data_loader, criterion, thr=0.5):
    print()
    print(f"Start Validation #{epoch:2d}")
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        valid_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)["out"]

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # ! restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            valid_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.4f}" for c, d in zip(CLASSES, dices_per_class)]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    wandb.log(
        {"valid_loss": valid_loss / len(data_loader), "avg_dice": avg_dice}, step=epoch
    )

    return avg_dice


# ! Training Process
def train(model, data_loader, val_loader, criterion, optimizer):
    print(f"Start Training...")

    n_class = len(CLASSES)
    best_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ! step 주기에 따른 Train Loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1} / {NUM_EPOCHS}], "
                    f"Step [{step+1} / {len(train_loader)}], "
                    f"Loss: {round(loss.item(), 4)}"
                )

        wandb.log({"train_loss": train_loss / len(data_loader)}, step=epoch)

        # ! validation 주기에 따른 Valid Loss 출력 및 Best Model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                print(
                    f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}"
                )
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model, file_name="baseline_FCN_ResNet50.pt")


# ! Model Importation & Loss function and Optimizer
model = BaseModel(classes=CLASSES)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

set_seed()
train(model, train_loader, valid_loader, criterion, optimizer)
