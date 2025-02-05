import torch.nn.functional as F
from monai.inferers import sliding_window_inference
import numpy as np
from czifile import CziFile
import torch
from torchvision.transforms._functional_tensor import normalize
from dnafiber.post_process import PostProcessor
from tifffile import imread
from pathlib import Path


def load_image(path: Path):
    path = str(path)
    if path.endswith(".czi"):
        data: np.ndarray = CziFile(path).asarray()
    elif path.endswith(".tif") or path.endswith(".tiff"):
        data = imread(path)
    else:
        raise ValueError("Invalid file format")

    data = data.squeeze()

    r = data[1]

    g = data[0]

    r = np.clip(r, 0, 1320)
    g = np.clip(g, 0, 4288)

    g = g / g.max()
    r = r / r.max()
    img = np.stack([r, g, np.zeros_like(r)], axis=-1).astype(np.float32)
    return img


def preprocess_image(image):
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


@torch.inference_mode()
def inference(model, path, use_cuda=False):
    image = load_image(path)
    image = preprocess_image(image)
    if use_cuda:
        image = image.cuda()
    if use_cuda:
        model = model.cuda()

    scale_factor = 2048 / 4512
    rescaled = F.interpolate(image, scale_factor=scale_factor, mode="bilinear")
    pred = sliding_window_inference(
        rescaled, roi_size=(1024, 1024), sw_batch_size=16, predictor=model, overlap=0.15
    )
    pred = F.interpolate(pred, size=image.shape[-2:], mode="nearest")
    pred = pred.argmax(1).cpu().numpy().squeeze()
    pp = PostProcessor(pred)
    mask = pp.apply()
    counts = pp.count_ratio()

    return mask, counts
