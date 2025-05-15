import base64

from xml.dom import minidom
import cv2
import numpy as np
from czifile import CziFile


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


def preprocess_tiff(raw_data):
    MAX_VALUE = 2**16 - 1
    h, w = raw_data.shape[1:3]
    orders = [1, 0]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i, chan in enumerate(raw_data):
        hist, bins = np.histogram(chan.ravel(), MAX_VALUE + 1, (0, MAX_VALUE + 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        bmax = np.searchsorted(cdf_normalized, 0.99, side="left")
        clip = np.clip(chan, 0, bmax)
        result[:, :, orders[i]] = (
            (clip - clip.min()) / (bmax - clip.min()) * 255
        ).astype(np.uint8)
    return result


def read_czi(filepath):
    data = CziFile(filepath).asarray().squeeze()
    return preprocess_tiff(data)


def read_tiff(filepath):
    from tifffile import imread

    data = imread(filepath)
    return preprocess_tiff(data)
