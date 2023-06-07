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

# ! Definition of Test Dataset
class XRayInferenceDataset(Dataset):
    def __init__(self, pngs, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        image = torch.from_numpy(image).float()

        return image, image_name


# ! Mask Map으로 나오는 Inference Result를 RLE로 encoding
def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# ! encoded RLE Result를 Mask Map으로 decoding
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


# ! Inference Process
def test(model, data_loader, classes, ind2class, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(classes)

        for step, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.cuda()
            outputs = model(images)["out"]

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{ind2class[c]}_{image_name}")

    return rles, filename_and_class



def main(args):
    CLASS2IND = {v: i for i, v in enumerate(args.classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    # ! Best Trained Model Importation
    model = torch.load(os.path.join(args.saved_dir, args.model + ".pt"))
    
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }


    # ! Albumentation Transforms & Generation of Test Dataset
    infer_transform = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(pngs, transforms=infer_transform)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=False
    )

    rles, filename_and_class = test(model, test_loader, args.classes, IND2CLASS)


    # ! Save CSV file for Submission
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    df.to_csv("output.csv", index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--saved_dir', type=str, default="/opt/ml/input/code/best_models", help='model save at {saved_dir}')
    parser.add_argument('--model', type=str, default="BaseModel", help='model type (default: BaseModel)')
    args = parser.parse_args()
    
    args.classes = [
        "finger-1","finger-2","finger-3","finger-4","finger-5","finger-6","finger-7","finger-8","finger-9",
        "finger-10","finger-11","finger-12","finger-13","finger-14","finger-15","finger-16","finger-17",
        "finger-18","finger-19","Trapezium","Trapezoid","Capitate","Hamate","Scaphoid","Lunate",
        "Triquetrum","Pisiform","Radius","Ulna",
        ]
    
    # for XRayInferenceDataset __getitem__
    global IMAGE_ROOT
    IMAGE_ROOT = "/opt/ml/input/data/test/DCM"

    main(args)