import json
import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

IMAGE_ROOT = "/opt/ml/input/data/train/DCM"
LABEL_ROOT = "/opt/ml/input/data/train/outputs_json"
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
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
LR = 1e-4
NUM_EPOCHS = 50
VAL_EVERY = 1


# ! Images & Annotations Importation
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
pngs = sorted(pngs)
jsons = sorted(jsons)


# ! Definition of Train/Valid Dataset with Group-KFold
class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, val_k=0):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        groups = [
            os.path.dirname(fname) for fname in _filenames
        ]  # split Train/Valid set
        ys = [0 for fname in _filenames]  # dummy label

        groud_kfold = GroupKFold(n_splits=5)  # Group KFold with K=5
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(groud_kfold.split(_filenames, ys, groups)):
            if is_train:
                if i == val_k:  # ! 0번을 Valid set으로 사용
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                if i == val_k:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
        print(f"IS_TRAIN={is_train}, fold{val_k} : {filenames}")  #####temp

        self.filenames = filenames  # ! 각 Iamge의 path가 담긴 list
        self.labelnames = labelnames  # ! 각 Label(Annotation file)의 path가 담긴 list
        self.is_train = is_train  # ! True이면 Train-Set으로, False이면 Valid-Set으로 Split 수행
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)

        label_shape = tuple(image.shape[:2]) + (
            len(CLASSES),
        )  # ! label shape = (H, W, NC)
        label = np.zeros(
            label_shape, dtype=np.uint8
        )  # ! shape = Height X Width X 29(=num of class labels)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:  # ! 각각의 class label을 확인하며
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)  # ! Polygon 형태의 point를 Mask 형식으로 변환
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = (
                {"image": image, "mask": label} if self.is_train else {"image": image}
            )
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
