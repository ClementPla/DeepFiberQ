import base64

from xml.dom import minidom
import cv2
import numpy as np
from czifile import CziFile
from tifffile import imread


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


def preprocess(raw_data, reverse_channels=False):
    MAX_VALUE = 2**16 - 1
    if raw_data.ndim == 2:
        raw_data = raw_data[np.newaxis, :, :]
    h, w = raw_data.shape[1:3]
    orders = np.arange(raw_data.shape[0])[::-1]  # Reverse channel order
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for i, chan in enumerate(raw_data):
        hist, bins = np.histogram(chan.ravel(), MAX_VALUE + 1, (0, MAX_VALUE + 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        bmax = np.searchsorted(cdf_normalized, 0.99, side="left")
        clip = np.clip(chan, 0, bmax).astype(np.float32)
        clip =  (clip - clip.min()) / (bmax - clip.min()) * 255
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