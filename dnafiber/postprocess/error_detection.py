import timm
import torch
import cv2
import numpy as np
from dnafiber.postprocess.fiber import FiberProps
from skimage.segmentation import expand_labels
import streamlit as st
import albumentations as A
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class MyModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "seresnet50.a1_in1k",
            pretrained=False,
            num_classes=1,
            in_chans=4,
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_model():
    return MyModel.from_pretrained("ClementP/dnafiber-error-detection")


def generate_thumbnail(image, fibers: list[FiberProps], verbose=False):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406, 0.229],
        std=[0.229, 0.224, 0.225, 0.225],
        max_pixel_value=255.0,
    )
    thumbnails = np.zeros((128, 128, 4, len(fibers)), dtype=np.float32)
    for i, fiber in enumerate(fibers):
        bbox = fiber.bbox
        x, y, w, h = bbox
        data = fiber.data

        seg = expand_labels(data, distance=2)

        offsetX = int(w * 0.5)
        offsetY = int(h * 0.5)
        # Compute the offset so that the bounding is square
        if offsetX < offsetY:
            offsetX = offsetY
        else:
            offsetY = offsetX

        img = image[y - offsetY : y + h + offsetY, x - offsetX : x + w + offsetX]
        img_with_bbox = img.copy()
        # Ensure img_with_bbox is contiguous in memory for OpenCV compatibility
        img_with_bbox = np.ascontiguousarray(img_with_bbox)
        # Draw the bounding box on the image
        try:
            cv2.rectangle(
                img_with_bbox,
                (offsetX, offsetY),
                (offsetX + w, offsetY + h),
                (0, 0, 255),
                2,
            )
        except Exception as e:
            # print(f"Error drawing rectangle: {e}")
            continue

        # Pad the segmentation mask to match the extended bounding box
        padded_seg = np.zeros_like(img_with_bbox[:, :, 0], dtype=np.uint8)
        try:
            padded_seg[offsetY : offsetY + h, offsetX : offsetX + w] = seg
        except ValueError as e:
            if verbose:
                print("Failed to extract bbox from fiber data:", e)
        # Resize all the images to a fixed size of 128x128
        img_with_bbox = cv2.resize(img_with_bbox, (128, 128))
        img_with_bbox = normalize(image=img_with_bbox)["image"]
        padded_seg = cv2.resize(
            padded_seg, (128, 128), interpolation=cv2.INTER_NEAREST_EXACT
        )[:, :, np.newaxis]
        thumbnails[..., i] = np.concatenate([img_with_bbox, padded_seg], axis=-1)

    return thumbnails


def error_inference_thumbnails(model, thumbnails: np.ndarray, device="cpu"):
    """
    Perform inference on the input image using the provided model.
    """
    thumbnails = (
        torch.from_numpy(thumbnails).float().permute(3, 2, 0, 1)
    )  # Convert to tensor and rearrange dimensions
    model.eval()
    model = model.to(device)
    thumbnails = thumbnails.to(device)
    with torch.no_grad():
        thumbnails = thumbnails.to(device)
        output = model(thumbnails) > 0
    return output.cpu().numpy()


def correct_fibers(
    fibers: list[FiberProps], image: np.ndarray, correction_model=None, device=None, verbose=False
):
    """
    Correct the fibers using the provided correction model.
    """
    if correction_model is None:
        return fibers

    thumbnails = generate_thumbnail(image, fibers)
    error_maps = error_inference_thumbnails(correction_model, thumbnails, device=device)

    for fiber, em in zip(fibers, error_maps):
        fiber.is_an_error = em

    return fibers
