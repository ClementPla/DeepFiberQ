import PIL.Image
import streamlit as st
from dnafiber.data.utils import read_czi, read_tiff, preprocess
import cv2
import numpy as np
import math
from dnafiber.deployment import _get_model
import PIL
from PIL import Image
import io
import base64

MAX_WIDTH = 1024
MAX_HEIGHT = 1024


TYPE_MAPPING = {
    0: "BG",
    1: "SINGLE",
    2: "BILATERAL",
    3: "TRICOLOR",
    4: "MULTICOLOR",
}


def load_image(_filepath, id=None):
    filename = str(_filepath.name)
    if filename.endswith(".czi"):
        return read_czi(_filepath)
    elif filename.endswith(".tif") or filename.endswith(".tiff"):
        return read_tiff(_filepath)
    elif (
        filename.endswith(".png")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
    ):
        image = PIL.Image.open(_filepath)
        image = np.array(image)
        return image
    else:
        raise NotImplementedError(f"File type {filename} is not supported yet")


@st.cache_data
def get_image(_filepath, reverse_channel, id):
    return get_image_cacheless(_filepath, reverse_channel, id)


def get_image_cacheless(filepath, reverse_channel, id):
    """
    A cacheless version of the get_image function.
    This function does not use caching and is intended for use in scenarios where caching is not desired.
    """
    filename = str(filepath.name)
    image = load_image(filepath, id)
    if (
        filename.endswith(".czi")
        or filename.endswith(".tif")
        or filename.endswith(".tiff")
    ):
        image = preprocess(image, reverse_channel)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return image


def get_multifile_image(_filepaths):
    result = None

    if _filepaths[0] is not None:
        chan1 = get_image(_filepaths[0], False, _filepaths[0].file_id)
        chan1 = cv2.cvtColor(chan1, cv2.COLOR_RGB2GRAY)
        h, w = chan1.shape[:2]
    else:
        chan1 = None
    if _filepaths[1] is not None:
        chan2 = get_image(_filepaths[1], False, _filepaths[1].file_id)
        chan2 = cv2.cvtColor(chan2, cv2.COLOR_RGB2GRAY)
        h, w = chan2.shape[:2]
    else:
        chan2 = None

    result = np.zeros((h, w, 3), dtype=np.uint8)

    if chan1 is not None:
        result[:, :, 0] = chan1
    else:
        result[:, :, 0] = chan2

    if chan2 is not None:
        result[:, :, 1] = chan2
    else:
        result[:, :, 1] = chan1

    return result


def numpy_to_base64_png(image_array):
    """
    Encodes a NumPy image array to a base64 string (PNG format).

    Args:
        image_array: A NumPy array representing the image.

    Returns:
        A base64 string representing the PNG image.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)

    # Create an in-memory binary stream
    buffer = io.BytesIO()

    # Save the image to the buffer in PNG format
    image.save(buffer, format="jpeg")

    # Get the byte data from the buffer
    png_data = buffer.getvalue()

    # Encode the byte data to base64
    base64_encoded = base64.b64encode(png_data).decode()

    return f"data:image/jpeg;base64,{base64_encoded}"


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
