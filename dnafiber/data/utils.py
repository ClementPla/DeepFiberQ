from PIL import Image
import io
import base64
from xml.dom import minidom
import cv2
import numpy as np
from czifile import CziFile
from tifffile import imread
import math
import streamlit as st
from matplotlib.colors import ListedColormap


CMAP = ListedColormap(["#000000", "#ff0000", "#00ff00"])

def read_svg(svg_path):
    doc = minidom.parse(str(svg_path))
    img_strings = {
        path.getAttribute("id"): path.getAttribute("href")
        for path in doc.getElementsByTagName("image")
    }
    doc.unlink()

    red = img_strings["Red"]
    green = img_strings["Green"]
    red = base64.b64decode(red.split(",")[1])
    green = base64.b64decode(green.split(",")[1])
    red = cv2.imdecode(np.frombuffer(red, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    green = cv2.imdecode(np.frombuffer(green, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    red = cv2.cvtColor(red, cv2.COLOR_BGRA2GRAY)
    green = cv2.cvtColor(green, cv2.COLOR_BGRA2GRAY)
    mask = np.zeros_like(red)
    mask[red > 0] = 1
    mask[green > 0] = 2
    return mask


def extract_bboxes(mask):
    mask = np.array(mask)
    mask = mask.astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        bboxes.append([x, y, x + w, y + h])
    return bboxes


def preprocess(raw_data, reverse_channels=False, bit_depth=14):
    MAX_VALUE = 2**bit_depth - 1
    if raw_data.ndim == 2:
        raw_data = raw_data[np.newaxis, :, :]
    h, w = raw_data.shape[1:3]
    orders = np.arange(raw_data.shape[0])[::-1]  # Reverse channel order by default
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for i, chan in enumerate(raw_data):
        hist, bins = np.histogram(chan.ravel(), MAX_VALUE + 1, (0, MAX_VALUE + 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        bmax = np.searchsorted(cdf_normalized, 0.99, side="left")
        clip = np.clip(chan, 0, bmax).astype(np.float32)
        clip = (clip - clip.min()) / (bmax - clip.min()) * 255
        result[:, :, orders[i]] = clip
    if reverse_channels:
        # Reverse channels 0 and 1
        result = result[:, :, [1, 0, 2]]
    return result


def read_czi(filepath):
    with CziFile(filepath) as czi:
        data = czi.asarray().squeeze()
    return data


def read_tiff(filepath):
    data = imread(filepath).squeeze()
    return data


def read_dv(filepath):
    from mrc import DVFile

    with DVFile(filepath) as dv:
        data = dv.asarray().squeeze()[:2]

    return data

def convert_rgb_to_mask(image, threshold=200):

    output = np.zeros(image.shape[:2], dtype=np.uint8)
    output[image[:, :, 0] > threshold] = 1
    output[image[:, :, 1] > threshold] = 2
    return output


def numpy_to_base64_jpeg(image_array):
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
    jpeg_data = buffer.getvalue()

    # Encode the byte data to base64
    base64_encoded = base64.b64encode(jpeg_data).decode()

    return f"data:image/jpeg;base64,{base64_encoded}"


@st.cache_data
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
