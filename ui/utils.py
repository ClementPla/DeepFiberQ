import streamlit as st
from dnafiber.data.utils import read_czi
import cv2
import numpy as np
from dnafiber.trainee import Trainee


MAX_WIDTH = 512
MAX_HEIGHT = 512


@st.cache_data
def get_image(_filepath, id):
    return read_czi(_filepath)


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
def get_model(device):
    model = (
        Trainee.load_from_checkpoint(
            "/home/clement/Documents/Projets/DNAFiber/checkpoints/breezy-sponge-14/last.ckpt",
            arch="segformer",
            encoder_name="mit_b2",
            training_config=dict(weight_decay=0.01, learning_rate=1e-4),
        )
        .eval()
        .to(device)
    )
    return model
