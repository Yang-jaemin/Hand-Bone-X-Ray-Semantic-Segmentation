import csv
import os
import argparse

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from collections import OrderedDict
from inference import encode_mask_to_rle, decode_rle_to_mask

# ensemble csv files
submission_files = [
    "/opt/ml/submissions/output_fcn_r101_sharpen_rbc.csv",
    "/opt/ml/submissions/output_MAnet_resnet101_preFalse_5e-4_epoch80.csv",
    "/opt/ml/submissions/output_fcn_r50_sharpen_adamP.csv",
]


classes = [
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

def encode_per_img(zeros_mask, threshold, rles):
    zeros_mask = zeros_mask >= threshold  # encode and save
    for segm in zeros_mask:
        rle = encode_mask_to_rle(segm)
        rles.append(rle)
    zeros_mask = np.zeros((29, 2048, 2048), dtype=np.float32)
    return zeros_mask

def ensemble(submission_files):
    submission_df = [pd.read_csv(file) for file in submission_files]

    threshold = int(len(submission_df) * 0.7)
    img_name = submission_df[0]["image_name"].to_list()

    zeros_mask = np.zeros((29, 2048, 2048), dtype=np.float32)
    new_classes = classes * len(set(img_name))

    rles = []
    for idx in tqdm(
        range(len(submission_df[0].to_numpy())), desc="Ensemble Submissions.."
    ):  # 모든 line에 대해
        class_idx = idx % 29
        if class_idx == 0:  # finished with 1 image
            if idx != 0:
                zeros_mask = encode_per_img(zeros_mask, threshold, rles)

        for submission in submission_df:  # 모든 submission에서 해당 line을 decode
            submission = submission.to_numpy()
            line = submission[idx] # class line

            if isinstance(line[2], str):
                class_mask = decode_rle_to_mask(line[2], 2048, 2048)
                zeros_mask[class_idx] += class_mask

    # for last img
    zeros_mask = encode_per_img(zeros_mask, threshold, rles)

    df = pd.DataFrame(
        {
            "image_name": img_name,
            "class": new_classes,
            "rle": rles,
        }
    )

    df.to_csv("ensemble_output.csv", index=False)


if __name__ == "__main__":
    ensemble(submission_files)
