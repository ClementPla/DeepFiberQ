import PIL.Image
import streamlit as st
from dnafiber.data.utils import read_czi
import cv2
import numpy as np
import math
from dnafiber.deployment import _get_model
import PIL

MAX_WIDTH = 512
MAX_HEIGHT = 512

MODELS_ZOO = {
    "SegFormer MiT-B4": "segformer_mit_b4",
    "segformer MiT-B2": "segformer_mit_b2",
    "U-Net++ SE-ResNet50": "unetplusplus_se_resnet50",
    "U-Net SE-ResNet50": "unet_se_resnet50",
    "U-Net MiT-B0": None,
}

TYPE_MAPPING = {
    0: "BG",
    1: "SINGLE",
    2: "BILATERAL",
    3: "TRICOLOR",
    4: "MULTICOLOR",
}


@st.cache_data
def get_image(_filepath, id):
    filename = str(_filepath.name)
    if filename.endswith(".czi"):
        return read_czi(_filepath)
    elif filename.endswith(".tif") or filename.endswith(".tiff"):
        raise NotImplementedError("Tiff files are not supported yet")
    elif (
        filename.endswith(".png")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
    ):
        image = PIL.Image.open(_filepath)
        return np.array(image)
    else:
        raise NotImplementedError(f"File type {filename} is not supported yet")


@st.cache_data
def get_resized_image(_image, id):
    h, w = _image.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(_image, new_size, interpolation=cv2.INTER_NEAREST)
    else:
        resized_image = _image
    if h > MAX_HEIGHT:
        scale = MAX_HEIGHT / h
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(
            resized_image, new_size, interpolation=cv2.INTER_NEAREST
        )
    else:
        resized_image = resized_image
    return resized_image


def bokeh_imshow(fig, image):
    # image is a numpy array of shape (h, w, 3) or (h, w) of type uint8
    if len(image.shape) == 2:
        # grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert to h*w with uint32
    img = np.empty((image.shape[0], image.shape[1]), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((image.shape[0], image.shape[1], 4))  # RGBA
    view[:, :, 0] = image[:, :, 0]
    view[:, :, 1] = image[:, :, 1]
    view[:, :, 2] = image[:, :, 2]
    view[:, :, 3] = 255  # Alpha channel
    fig.image_rgba(image=[img], x=0, y=0, dw=image.shape[1], dh=image.shape[0])


@st.cache_resource
def get_model(device, revision=None):
    return _get_model(revision=revision, device=device)


def pad_image_to_croppable(_image, bx, by, uid=None):
    # Pad the image to be divisible by bx and by
    h, w = _image.shape[:2]
    if h % bx != 0:
        pad_h = bx - (h % bx)
    else:
        pad_h = 0
    if w % by != 0:
        pad_w = by - (w % by)
    else:
        pad_w = 0
    _image = cv2.copyMakeBorder(
        _image,
        math.ceil(pad_h / 2),
        math.floor(pad_h / 2),
        math.ceil(pad_w / 2),
        math.floor(pad_w / 2),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return _image
