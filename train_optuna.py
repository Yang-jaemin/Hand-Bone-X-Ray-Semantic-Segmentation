import datetime
import os
import random
import argparse

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from importlib import import_module
from collections import OrderedDict

import wandb
from dataset import XRayDataset
from model import *
from alarm import send_message_slack
from loss import DiceLoss, IoULoss	
from adamp import AdamP
import joblib


# ! Definitions of Optionable Training functions & Wandb Configuration
def set_seed(RANDOM_SEED):
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


def save_model(model, saved_dir, file_name="fcn_resnet50_best_model.pt"):
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


# ! Validation Process
def validation(epoch, model, data_loader, criterion1, criterion2, criterion3, classes, _wandb, thr=0.5):
    print()
    print(f"Start Validation #{epoch:2d}")
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(classes)
        valid_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)
            if isinstance(outputs, OrderedDict):
                outputs = outputs["out"]

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # ! restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = (
                (args.loss[0] * criterion1(outputs, masks)) # BCE
                + (args.loss[1] * criterion2(outputs, masks)) # Dice
                + (args.loss[2] * criterion3(outputs, masks)) # IoU
            )
            valid_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.4f}" for c, d in zip(classes, dices_per_class)]
    dice_str = "\n".join(dice_str)
    print("\n" + dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    if _wandb:
        wandb.log(
            {"valid_loss": valid_loss / len(data_loader), "avg_dice": avg_dice},
            step=epoch,
        )

    return avg_dice


def get_data(c_size):
    album_transform1 = A.Compose([
            A.Resize(c_size, c_size, p=1.0),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2), p=0.25),
            A.Compose([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=1.0),
            ], p=0.25),
    ])

    ## Augmentations for Valid Dataset, Resize & Normalize만 실시
    album_transform2 = A.Compose([
            A.Resize(c_size, c_size, p=1.0),
    ])

    train_dataset = XRayDataset(is_train=True, transforms=album_transform1)
    valid_dataset = XRayDataset(is_train=False, transforms=album_transform2)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, valid_loader

# ! Training Process
def train(model, criterion1, criterion2, criterion3, optimizer, args):
    torch.cuda.empty_cache()

    data_loader, val_loader = get_data(cfg['resize'])
    print(f"Start Training...")

    best_dice = 0.0

    for epoch in range(cfg['n_epochs']):
        model.train()
        train_loss = 0

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)
            if isinstance(outputs, OrderedDict):
                outputs = outputs["out"]

            loss = (	
                (args.loss[0] * criterion1(outputs, masks)) # BCE	
                + (args.loss[1] * criterion2(outputs, masks)) # Dice	
                + (args.loss[2] * criterion3(outputs, masks)) # IoU	
            )
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ! step 주기에 따른 Train Loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1} / {cfg['n_epochs']}], "
                    f"Step [{step+1} / {len(data_loader)}], "
                    f"Loss: {round(loss.item(), 4)}"
                )
        if args.wandb:
            wandb.log({"train_loss": train_loss / len(data_loader)}, step=epoch)
            wandb.log({"learning_rate" : cfg["lr"]})
            wandb.log({"epochs" : cfg["n_epochs"]})

        # ! validation 주기에 따른 Valid Loss 출력 및 Best Model 저장
        if (epoch + 1) % args.val_every == 0:
            dice = validation(
                epoch + 1, model, val_loader, criterion1, criterion2, criterion3, args.classes, args.wandb
            )

            if best_dice < dice:
                print(
                    f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}"
                )
                print(f"Save model in {args.saved_dir}")
                best_dice = dice
                save_model(model, args.saved_dir, file_name=str(args.model) + ".pt")

    return best_dice


def make_cfg(trial):
    m_cfg = {
          'n_epochs' : trial.suggest_int('n_epochs', 10, 40, 5),
          'seed' : 127,
          'resize' : trial.suggest_categorical('resize', [1024,512,256]),
          'lr' : trial.suggest_loguniform('lr', 1e-3, 1e-2),
        #   'momentum': trial.suggest_uniform('momentum', 0.4, 0.99),
        #   'optimizer': trial.suggest_categorical('optimizer',[optim.Adam, optim.AdamW, optim.RAdam])
          }
    return m_cfg


def main(trial):
    global args
    global cfg
    
    set_seed(args.seed)
    cfg = make_cfg(trial)


    # ! Model Importation & Loss function and Optimizer
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(classes=args.classes)
    print(model)

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    criterion3 = IoULoss()

    # optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer = AdamP(params=model.parameters(), lr=cfg['lr'], weight_decay=1e-2)

    dice_val = train(model, criterion1,criterion2, criterion3, optimizer, args)
    return dice_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=127, help="random seed (default: 127)"
    )  # RANDOM SEED
    parser.add_argument(
        "--epochs", type=int, default=40, help="number of epochs to train (default: 40)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--wandb",
        type=int,
        default=1,
        help="1 : save in wandb, 0 : do not save in wandb",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
    )
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument(
        "--saved_dir",
        type=str,
        default="/opt/ml/input/code/best_models",
        help="model save at {saved_dir}",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--loss", type=float, nargs="+", default=[1.0, 1.0, 1.0]
    )

    args = parser.parse_args()

    # print(args)

    args.classes = [
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

    # wandb init
    if args.wandb:
        wandb.init(
            project="HandBoneSeg",
            notes="Optuna Test",
            config={
                "model": args.model,
                # "epochs": args.epochs,
                "batch_size": args.batch_size,
                # "learning_rate": args.lr,
                "random_seed": args.seed,
                "BCE_weight" : args.loss[0],
                "Dice_weight" : args.loss[1],
                "IoU_weight" : args.loss[2],
                "optimizer" : "AdamP",
            },
            tags=["optuna"],
        )

    # make saved dir
    if not os.path.isdir(args.saved_dir):
        os.mkdir(args.saved_dir)


    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(main, n_trials=30)
        joblib.dump(study, './optuna_output.pkl')
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        send_message_slack(text="Model Learning Completed")
    except:
        send_message_slack(text="Model Learning Failed")
