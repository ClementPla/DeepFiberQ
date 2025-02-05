
import base64

from xml.dom import minidom
import cv2
import numpy as np

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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        bboxes.append([x, y, x + w, y + h])
    return bboxes

