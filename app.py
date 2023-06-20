import json
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from app_utils import *

st.set_page_config(initial_sidebar_state="expanded")
st.title("Hand Bone X-Ray Semantic Segmentation")
st.subheader("CV-10 : Bro3Sis1 Team Prediction")
st.divider()


def main():
    global CAM_READY

    mode = st.selectbox("Visualization Mode", ("Prediction Output", "Compare Loss"))

    box_size = st.selectbox("Input Image Size", (512, 1024))
    img_type = st.selectbox("Input Image Type", ("Gray", "RGB"))
    if img_type == "Gray":
        img_type = "L"

    st.divider()

    st.error("Import Your Own Best Semantic Segmentation Model(.pth or .pt)")
    uploaded_model = st.file_uploader(
        "Best Model Importation",
        accept_multiple_files=False,
        type=["pth", "pt"],
        label_visibility="collapsed",
    )
    if uploaded_model is not None:
        model = torch.load(uploaded_model, map_location="cpu")
        model.to("cuda")
        model.eval()
    st.warning("Input Your Own Sample Image File(.png or .jpg) for Visualization")
    uploaded_image = st.file_uploader(
        "Input Sample Image",
        accept_multiple_files=False,
        type=["png", "jpg"],
        label_visibility="collapsed",
    )
    st.success("Input Your Own Sample JSON File(.json) for Visualization")
    uploaded_json = st.file_uploader(
        "Input Sample JSON File",
        accept_multiple_files=False,
        type=["json"],
        label_visibility="collapsed",
    )

    if uploaded_json is not None:
        annotations = json.load(uploaded_json)
        annotations = annotations["annotations"]

    if mode != "Prediction Output":
        class_idx = st.radio("Select Class Label ID", [num for num in range(29)])

    button = st.button("Start Inference")

    st.divider()

    with st.spinner("Inference In Progress"):
        if button:
            button = False

            if uploaded_model is not None and uploaded_image is not None:
                image = Image.open(uploaded_image)
                image = np.array(image.convert(img_type))
                origin_image = image.copy()

                album_transform = A.Resize(box_size, box_size)

                if uploaded_json is not None:
                    label = np.zeros(
                        (image.shape[0], image.shape[1], 29), dtype=np.uint8
                    )
                    mask = label2mask(annotations, label)

                    augment = album_transform(image=image, mask=mask)
                    image = augment["image"]
                    mask = augment["mask"]
                    mask = mask.transpose(2, 0, 1)
                    mask = torch.from_numpy(mask).float().unsqueeze(0)
                    mask = mask.to("cuda")
                else:
                    image = album_transform(image=image)["image"]

                image = image / 255.0
                rgb_image = np.float32(image)
                image = image.transpose(2, 0, 1)
                image = torch.from_numpy(image).float().unsqueeze(0)
                image = image.to("cuda")

                output = model(image)
                if isinstance(output, OrderedDict):
                    output = output["out"]
                if image.shape[-2:] != output.shape[-2:]:
                    output = F.interpolate(
                        output, size=image.shape[-2:], mode="bilinear"
                    )

                if mode == "Prediction Output":
                    output_image = plot_pred(output)
                    ground_truths, diff = gt_pred_diff(mask, output)

                    col1, col2 = st.columns(2)
                    col1.image(origin_image, caption="Original")
                    col2.image(output_image, caption=mode)

                    col3, col4 = st.columns(2)
                    col3.image(ground_truths, caption="Ground Truth")
                    col4.image(diff, caption="Ground Truth VS Prediction Diff")
                    st.text(f"# of diff pixel {np.count_nonzero(diff)}")

                elif mode == "Compare Loss":
                    ground_truth_maps, loss_maps = compare_loss(rgb_image, mask, output)

                    col1, col2 = st.columns(2)
                    col1.image(ground_truth_maps[class_idx], caption="Ground Truth")
                    col2.image(loss_maps[class_idx], caption=mode)

            else:
                st.text("Please Input or Upload Necessary Data")


if __name__ == "__main__":
    main()
