import torch.nn.functional as F
import numpy as np
import torch
from torchvision.transforms._functional_tensor import normalize
import pandas as pd
from skimage.segmentation import expand_labels
import albumentations as A
from monai.inferers import SlidingWindowInferer
from dnafiber.ui.utils import _get_model
from dnafiber.model.autopadDPT import AutoPad
import torch.nn as nn
import ttach as tta

transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)


def preprocess_image(image):
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def convert_to_dataset(counts):
    data = {"index": [], "red": [], "green": [], "ratio": []}
    for k, v in counts.items():
        data["index"].append(k)
        data["green"].append(v["green"])
        data["red"].append(v["red"])
        if v["red"] == 0:
            data["ratio"].append(np.nan)
        else:
            data["ratio"].append(v["green"] / (v["red"]))
    df = pd.DataFrame(data)
    return df


def convert_mask_to_image(mask, expand=False):
    if expand:
        mask = expand_labels(mask, distance=expand)
    h, w = mask.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    GREEN = np.array([0, 255, 0])
    RED = np.array([255, 0, 0])

    image[mask == 1] = RED
    image[mask == 2] = GREEN

    return image


class Inferer(nn.Module):
    def __init__(self, model, sliding_window_inferer=None, use_tta=False):
        super().__init__()

        self.model = AutoPad(nn.Sequential(model, nn.Softmax(dim=1)), 32)
        self.model.eval()

        self.sliding_window_inferer = sliding_window_inferer

        if use_tta:
            transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 90, 180]),
                ]
            )
            self.model = tta.SegmentationTTAWrapper(
                self.model, transforms, merge_mode="gmean"
            )

    def forward(self, image):
        if self.sliding_window_inferer is not None:
            output = self.sliding_window_inferer(image, self.model)
        else:
            output = self.model(image)
        return output


@torch.inference_mode()
def infer(
    model,
    image,
    device,
    scale=0.13,
    use_tta=False,
    to_numpy=True,
    only_probabilities=False,
    verbose=False,
):
    if isinstance(model, str):
        model = _get_model(device=device, revision=model)
    model_pixel_size = 0.26

    scale = scale / model_pixel_size
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    h, w = tensor.shape[2], tensor.shape[3]
    device = torch.device(device)
    sliding_window = None

    if int(h * scale) > 1024 or int(w * scale) > 1024:
        sliding_window = SlidingWindowInferer(
            roi_size=(1024, 1024),
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian",
            device=device,
            progress=verbose,
        )

    inferer = Inferer(
        model=model, sliding_window_inferer=sliding_window, use_tta=use_tta
    )
    with torch.autocast(device_type=device.type):
        tensor = F.interpolate(
            tensor,
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
        )
        probabilities = inferer(tensor)
        probabilities = probabilities.cpu()

        probabilities = F.interpolate(
            probabilities,
            size=(h, w),
            mode="bilinear",
        )
        return probabilities
